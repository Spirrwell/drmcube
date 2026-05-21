/*
 * drmcube — DRM/KMS + GBM + EGL + GLESv2 rotating-cube display test tool
 * ----------------------------------------------------------------------
 * Designed to exercise an embedded device with two outputs (e.g. an
 * internal LVDS/eDP/DSI panel and an external HDMI display) in three modes:
 *
 *   clone-encoder    One CRTC drives BOTH connectors from a single
 *                    framebuffer. The mode is necessarily identical on both
 *                    outputs (they share the CRTC scanout timing). This is
 *                    "hardware" cloning and depends on the two connectors'
 *                    encoders sharing a possible CRTC.
 *
 *   clone-compositor Two CRTCs, two framebuffers, the SAME scene rendered
 *                    to each. Each output may use a different mode/resolution.
 *                    This is "software/compositor" cloning via EGL/GLES.
 *
 *   independent      Two CRTCs, two framebuffers, different mode per output,
 *                    and the cube is drawn with a different colour/rotation
 *                    on each so it is obvious the displays are independent.
 *
 * Modes can be picked by name (e.g. 1920x1080@60) or supplied as a raw
 * X11-style Modeline, which is handed straight to the kernel via the legacy
 * drmModeSetCrtc() path — so it works even if the display never advertised it.
 *
 * BUILD (needs libdrm, gbm, EGL, GLESv2 dev packages):
 *   cc -O2 -Wall -Wextra main.c -o drmcube \
 *      $(pkg-config --cflags --libs libdrm gbm egl glesv2) -lm
 *   (or: -I/usr/include/libdrm -ldrm -lgbm -lEGL -lGLESv2 -lm)
 *
 * RUN (must be DRM master: from a bare text console, usually as root, with
 * no X/Wayland compositor holding the device):
 *   ./drmcube --list
 *   ./drmcube --mode clone-encoder    --mode0 1920x1080@60
 *   ./drmcube --mode clone-compositor --connector0 LVDS-1  --mode0 1280x800 \
 *                                     --connector1 HDMI-A-1 --mode1 1920x1080@60
 *   ./drmcube --mode independent      --mode1 'modeline:148.50 1920 2008 2052 2200 1080 1084 1089 1125 +hsync +vsync'
 *
 * Ctrl-C exits and restores the original CRTC configuration (best effort).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <signal.h>
#include <time.h>
#include <math.h>
#include <getopt.h>

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>

#ifndef EGL_PLATFORM_GBM_KHR
#define EGL_PLATFORM_GBM_KHR 0x31D7
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_CONNS_PER_OUTPUT 4

/* ------------------------------------------------------------------ */
/* Small column-major 4x4 matrix helpers (OpenGL convention).          */
/* ------------------------------------------------------------------ */

static void mat4_identity(float *m)
{
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

/* r = a * b  (column-major) */
static void mat4_mul(float *r, const float *a, const float *b)
{
    float t[16];
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++)
            t[col * 4 + row] =
                a[0 * 4 + row] * b[col * 4 + 0] +
                a[1 * 4 + row] * b[col * 4 + 1] +
                a[2 * 4 + row] * b[col * 4 + 2] +
                a[3 * 4 + row] * b[col * 4 + 3];
    memcpy(r, t, sizeof(t));
}

static void mat4_perspective(float *m, float fovy, float aspect,
                             float znear, float zfar)
{
    float f = 1.0f / tanf(fovy * 0.5f);
    memset(m, 0, 16 * sizeof(float));
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = (zfar + znear) / (znear - zfar);
    m[11] = -1.0f;
    m[14] = (2.0f * zfar * znear) / (znear - zfar);
}

static void mat4_translate(float *m, float x, float y, float z)
{
    mat4_identity(m);
    m[12] = x;
    m[13] = y;
    m[14] = z;
}

/* Rotation about an arbitrary axis, matching glRotate's column-major layout. */
static void mat4_rotate(float *m, float angle, float x, float y, float z)
{
    float len = sqrtf(x * x + y * y + z * z);
    if (len < 1e-6f) { mat4_identity(m); return; }
    x /= len; y /= len; z /= len;
    float c = cosf(angle), s = sinf(angle), o = 1.0f - c;

    mat4_identity(m);
    m[0]  = x * x * o + c;
    m[1]  = y * x * o + z * s;
    m[2]  = z * x * o - y * s;
    m[4]  = x * y * o - z * s;
    m[5]  = y * y * o + c;
    m[6]  = z * y * o + x * s;
    m[8]  = x * z * o + y * s;
    m[9]  = y * z * o - x * s;
    m[10] = z * z * o + c;
}

/* ------------------------------------------------------------------ */
/* Cube geometry: 36 vertices, each pos(3) + normal(3) + colour(3).    */
/* ------------------------------------------------------------------ */

static float g_cube[36 * 9];

static void build_cube(void)
{
    static const float corner[8][3] = {
        {-1,-1,-1},{ 1,-1,-1},{ 1, 1,-1},{-1, 1,-1},
        {-1,-1, 1},{ 1,-1, 1},{ 1, 1, 1},{-1, 1, 1},
    };
    /* Four corners per face (winding irrelevant: culling is disabled). */
    static const int face[6][4] = {
        {4,5,6,7}, /* +Z */ {1,0,3,2}, /* -Z */
        {5,1,2,6}, /* +X */ {0,4,7,3}, /* -X */
        {3,2,6,7}, /* +Y */ {0,1,5,4}, /* -Y */
    };
    static const float normal[6][3] = {
        {0,0,1},{0,0,-1},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},
    };
    static const float colour[6][3] = {
        {0.90f,0.30f,0.30f},{0.30f,0.90f,0.35f},{0.30f,0.50f,0.95f},
        {0.95f,0.85f,0.25f},{0.85f,0.35f,0.85f},{0.30f,0.85f,0.90f},
    };
    static const int tri[6] = {0,1,2, 0,2,3}; /* two triangles per quad */

    float *p = g_cube;
    for (int f = 0; f < 6; f++) {
        for (int v = 0; v < 6; v++) {
            int ci = face[f][tri[v]];
            *p++ = corner[ci][0] * 0.8f;
            *p++ = corner[ci][1] * 0.8f;
            *p++ = corner[ci][2] * 0.8f;
            *p++ = normal[f][0]; *p++ = normal[f][1]; *p++ = normal[f][2];
            *p++ = colour[f][0]; *p++ = colour[f][1]; *p++ = colour[f][2];
        }
    }
}

/* ------------------------------------------------------------------ */
/* GLES2 shaders.                                                      */
/* ------------------------------------------------------------------ */

static const char *VS =
    "attribute vec3 a_position;\n"
    "attribute vec3 a_normal;\n"
    "attribute vec3 a_color;\n"
    "uniform mat4 u_mvp;\n"
    "uniform mat4 u_model;\n"
    "varying vec3 v_color;\n"
    "varying vec3 v_normal;\n"
    "void main() {\n"
    "    gl_Position = u_mvp * vec4(a_position, 1.0);\n"
    "    v_normal = mat3(u_model) * a_normal;\n"
    "    v_color = a_color;\n"
    "}\n";

static const char *FS =
    "precision mediump float;\n"
    "varying vec3 v_color;\n"
    "varying vec3 v_normal;\n"
    "uniform vec3 u_tint;\n"
    "void main() {\n"
    "    vec3 N = normalize(v_normal);\n"
    "    vec3 L = normalize(vec3(0.4, 0.6, 1.0));\n"
    "    float d = max(dot(N, L), 0.0);\n"
    "    vec3 c = v_color * u_tint * (0.35 + 0.65 * d);\n"
    "    gl_FragColor = vec4(c, 1.0);\n"
    "}\n";

/* ------------------------------------------------------------------ */
/* Per-output state and global context.                                */
/* ------------------------------------------------------------------ */

struct output {
    uint32_t           crtc_id;
    drmModeConnector  *conns[MAX_CONNS_PER_OUTPUT];
    uint32_t           connector_ids[MAX_CONNS_PER_OUTPUT];
    int                num_connectors;
    drmModeModeInfo    mode;
    drmModeCrtc       *saved_crtc;       /* original config, for restore */

    struct gbm_surface *gs;
    EGLSurface          surface;
    int                 width, height;

    struct gbm_bo      *prev_bo;
    struct gbm_bo      *next_bo;
    int                 flip_pending;

    int                 variant;         /* 0 or 1 — drives colour/spin */

    /* Refresh-rate / missed-vblank statistics. */
    char                label[40];       /* connector name(s) for reports */
    double              target_hz;       /* exact rate from mode timings   */
    uint64_t            target_period_ns;/* exact vblank period (ns)        */
    uint64_t            last_flip_ns;    /* vblank time of previous flip    */
    unsigned            flips_window;    /* flips counted this report cycle */
    unsigned            missed_window;   /* skipped vblanks this cycle      */
};

static int                 g_drm_fd = -1;
static struct gbm_device  *g_gbm     = NULL;
static EGLDisplay          g_egl_dpy = EGL_NO_DISPLAY;
static EGLContext          g_egl_ctx = EGL_NO_CONTEXT;
static EGLConfig           g_egl_cfg = NULL;

static struct output       g_out[2];
static int                 g_num_out = 0;
static int                 g_stats   = 0;   /* print per-output fps each sec */

static GLuint              g_prog, g_vbo;
static GLint               g_a_pos, g_a_nrm, g_a_col;
static GLint               g_u_mvp, g_u_model, g_u_tint;

static volatile sig_atomic_t g_running = 1;
static void on_sigint(int s) { (void)s; g_running = 0; }

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Safe printf-append onto a NUL-terminated buffer. */
static void appendf(char *buf, size_t cap, const char *fmt, ...)
{
    size_t len = strlen(buf);
    if (len >= cap) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf + len, cap - len, fmt, ap);
    va_end(ap);
}

/* ------------------------------------------------------------------ */
/* DRM connector helpers.                                              */
/* ------------------------------------------------------------------ */

static const char *conn_type_name(uint32_t t)
{
    switch (t) {
    case DRM_MODE_CONNECTOR_VGA:         return "VGA";
    case DRM_MODE_CONNECTOR_DVII:        return "DVI-I";
    case DRM_MODE_CONNECTOR_DVID:        return "DVI-D";
    case DRM_MODE_CONNECTOR_DVIA:        return "DVI-A";
    case DRM_MODE_CONNECTOR_Composite:   return "Composite";
    case DRM_MODE_CONNECTOR_SVIDEO:      return "SVIDEO";
    case DRM_MODE_CONNECTOR_LVDS:        return "LVDS";
    case DRM_MODE_CONNECTOR_Component:   return "Component";
    case DRM_MODE_CONNECTOR_9PinDIN:     return "DIN";
    case DRM_MODE_CONNECTOR_DisplayPort: return "DP";
    case DRM_MODE_CONNECTOR_HDMIA:       return "HDMI-A";
    case DRM_MODE_CONNECTOR_HDMIB:       return "HDMI-B";
    case DRM_MODE_CONNECTOR_TV:          return "TV";
    case DRM_MODE_CONNECTOR_eDP:         return "eDP";
    case DRM_MODE_CONNECTOR_VIRTUAL:     return "Virtual";
    case DRM_MODE_CONNECTOR_DSI:         return "DSI";
#ifdef DRM_MODE_CONNECTOR_DPI
    case DRM_MODE_CONNECTOR_DPI:         return "DPI";
#endif
    default:                             return "Unknown";
    }
}

static int conn_is_internal(uint32_t t)
{
    return t == DRM_MODE_CONNECTOR_LVDS ||
           t == DRM_MODE_CONNECTOR_eDP  ||
           t == DRM_MODE_CONNECTOR_DSI
#ifdef DRM_MODE_CONNECTOR_DPI
           || t == DRM_MODE_CONNECTOR_DPI
#endif
           ;
}

static void connector_name(const drmModeConnector *c, char *buf, size_t n)
{
    snprintf(buf, n, "%s-%u", conn_type_name(c->connector_type),
             c->connector_type_id);
}

static uint32_t possible_crtcs_for_connector(int fd, drmModeConnector *c)
{
    uint32_t mask = 0;
    for (int e = 0; e < c->count_encoders; e++) {
        drmModeEncoder *enc = drmModeGetEncoder(fd, c->encoders[e]);
        if (enc) { mask |= enc->possible_crtcs; drmModeFreeEncoder(enc); }
    }
    return mask;
}

/* Pick a free CRTC for a connector, marking it used in *used_idx_mask. */
static int pick_crtc(int fd, drmModeRes *res, drmModeConnector *c,
                     uint32_t *used_idx_mask, uint32_t *crtc_id_out)
{
    uint32_t pc = possible_crtcs_for_connector(fd, c);
    for (int i = 0; i < res->count_crtcs; i++) {
        if (!(pc & (1u << i)))            continue;
        if (*used_idx_mask & (1u << i))   continue;
        *used_idx_mask |= (1u << i);
        *crtc_id_out = res->crtcs[i];
        return 0;
    }
    return -1;
}

/* Pick a single CRTC that can drive BOTH connectors (encoder-level clone). */
static int pick_common_crtc(int fd, drmModeRes *res,
                            drmModeConnector *a, drmModeConnector *b,
                            uint32_t *crtc_id_out)
{
    uint32_t common = possible_crtcs_for_connector(fd, a) &
                      possible_crtcs_for_connector(fd, b);
    for (int i = 0; i < res->count_crtcs; i++)
        if (common & (1u << i)) { *crtc_id_out = res->crtcs[i]; return 0; }
    return -1;
}

static drmModeCrtc *get_connector_crtc(int fd, drmModeConnector *c)
{
    if (c->encoder_id) {
        drmModeEncoder *e = drmModeGetEncoder(fd, c->encoder_id);
        if (e) {
            uint32_t cid = e->crtc_id;
            drmModeFreeEncoder(e);
            if (cid) return drmModeGetCrtc(fd, cid);
        }
    }
    return NULL;
}

/* Locate a connector by name ("HDMI-A-1"), numeric id, or auto-pick.
 * In auto mode (want == NULL), prefer internal/external panels as requested,
 * falling back to any other connected connector. */
static drmModeConnector *find_connector(int fd, drmModeRes *res,
                                        const char *want, int prefer_internal)
{
    uint32_t want_id = 0;
    if (want) {
        char *end = NULL;
        long id = strtol(want, &end, 10);
        if (end != want && *end == '\0') want_id = (uint32_t)id;
    }

    uint32_t fallback_id = 0;
    for (int i = 0; i < res->count_connectors; i++) {
        drmModeConnector *c = drmModeGetConnector(fd, res->connectors[i]);
        if (!c) continue;

        char name[64];
        connector_name(c, name, sizeof(name));

        if (want) {
            if ((want_id && c->connector_id == want_id) ||
                strcasecmp(name, want) == 0) {
                return c;                 /* explicit: honour even if disconnected */
            }
        } else if (c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
            if (conn_is_internal(c->connector_type) == prefer_internal)
                return c;                 /* exact preference match */
            if (!fallback_id) fallback_id = c->connector_id;
        }
        drmModeFreeConnector(c);
    }

    if (!want && fallback_id)
        return drmModeGetConnector(fd, fallback_id);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Modeline parsing and mode selection.                                */
/* ------------------------------------------------------------------ */

/* Accepts e.g.  "173.00 1920 2048 2248 2576 1080 1083 1088 1120 -hsync +vsync"
 * or a full     'Modeline "name" 173.00 ... -hsync +vsync'                    */
static int parse_modeline(const char *in, drmModeModeInfo *mode)
{
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", in);

    char *tok[64];
    int nt = 0;
    char *save = NULL;
    for (char *t = strtok_r(buf, " \t", &save); t && nt < 64;
         t = strtok_r(NULL, " \t", &save))
        tok[nt++] = t;

    int i = 0;
    char name[32] = "";

    if (i < nt && strcasecmp(tok[i], "Modeline") == 0) i++;
    if (i < nt && tok[i][0] == '"') {                 /* quoted name */
        snprintf(name, sizeof(name), "%s", tok[i] + 1);
        char *q = strchr(name, '"');
        if (q) *q = '\0';
        i++;
    }
    if (i >= nt) { fprintf(stderr, "modeline: missing pixel clock\n"); return -1; }

    char *end = NULL;
    double mhz = strtod(tok[i], &end);
    if (end == tok[i] || *end != '\0') {
        fprintf(stderr, "modeline: bad pixel clock '%s'\n", tok[i]);
        return -1;
    }
    i++;

    if (nt - i < 8) {
        fprintf(stderr, "modeline: need 8 timing values "
                        "(hdisp hss hse htot vdisp vss vse vtot)\n");
        return -1;
    }
    long v[8];
    for (int k = 0; k < 8; k++) {
        char *e = NULL;
        v[k] = strtol(tok[i + k], &e, 10);
        if (e == tok[i + k] || *e != '\0') {
            fprintf(stderr, "modeline: bad timing value '%s'\n", tok[i + k]);
            return -1;
        }
    }
    i += 8;

    uint32_t flags = 0;
    int have_h = 0, have_v = 0, interlace = 0, dblscan = 0;
    for (; i < nt; i++) {
        if      (!strcasecmp(tok[i], "+hsync")) { flags |= DRM_MODE_FLAG_PHSYNC; have_h = 1; }
        else if (!strcasecmp(tok[i], "-hsync")) { flags |= DRM_MODE_FLAG_NHSYNC; have_h = 1; }
        else if (!strcasecmp(tok[i], "+vsync")) { flags |= DRM_MODE_FLAG_PVSYNC; have_v = 1; }
        else if (!strcasecmp(tok[i], "-vsync")) { flags |= DRM_MODE_FLAG_NVSYNC; have_v = 1; }
        else if (!strcasecmp(tok[i], "interlace") ||
                 !strcasecmp(tok[i], "+interlace")) { flags |= DRM_MODE_FLAG_INTERLACE; interlace = 1; }
        else if (!strcasecmp(tok[i], "doublescan"))  { flags |= DRM_MODE_FLAG_DBLSCAN;   dblscan   = 1; }
        /* csync / unknown flags ignored */
    }
    if (!have_h) flags |= DRM_MODE_FLAG_PHSYNC;
    if (!have_v) flags |= DRM_MODE_FLAG_PVSYNC;

    memset(mode, 0, sizeof(*mode));
    mode->clock       = (uint32_t)(mhz * 1000.0 + 0.5);   /* MHz -> kHz */
    mode->hdisplay    = (uint16_t)v[0];
    mode->hsync_start = (uint16_t)v[1];
    mode->hsync_end   = (uint16_t)v[2];
    mode->htotal      = (uint16_t)v[3];
    mode->vdisplay    = (uint16_t)v[4];
    mode->vsync_start = (uint16_t)v[5];
    mode->vsync_end   = (uint16_t)v[6];
    mode->vtotal      = (uint16_t)v[7];
    mode->flags       = flags;
    mode->type        = DRM_MODE_TYPE_USERDEF;

    double vref = (double)mode->clock * 1000.0 /
                  ((double)mode->htotal * (double)mode->vtotal);
    if (interlace) vref *= 2.0;
    if (dblscan)   vref /= 2.0;
    mode->vrefresh = (uint32_t)(vref + 0.5);

    if (name[0])
        snprintf(mode->name, sizeof(mode->name), "%s", name);
    else
        snprintf(mode->name, sizeof(mode->name), "%dx%d",
                 mode->hdisplay, mode->vdisplay);
    return 0;
}

/* spec:  NULL/"preferred"  | "WxH" | "WxH@R" | "modeline:<...>" */
static int select_mode(drmModeConnector *c, const char *spec,
                       drmModeModeInfo *out)
{
    if (!spec || !*spec || !strcasecmp(spec, "preferred")) {
        for (int i = 0; i < c->count_modes; i++)
            if (c->modes[i].type & DRM_MODE_TYPE_PREFERRED) {
                *out = c->modes[i];
                return 0;
            }
        if (c->count_modes > 0) { *out = c->modes[0]; return 0; }
        fprintf(stderr, "connector has no modes; supply 'modeline:...'\n");
        return -1;
    }

    if (!strncasecmp(spec, "modeline:", 9))
        return parse_modeline(spec + 9, out);

    int w = 0, h = 0;
    double r = 0;
    if (sscanf(spec, "%dx%d@%lf", &w, &h, &r) >= 2) {
        drmModeModeInfo *best = NULL;
        for (int i = 0; i < c->count_modes; i++) {
            drmModeModeInfo *m = &c->modes[i];
            if (m->hdisplay != w || m->vdisplay != h) continue;
            if (r <= 0) {
                if (!best || (m->type & DRM_MODE_TYPE_PREFERRED)) best = m;
            } else if (fabs((double)m->vrefresh - r) < 1.0) {
                best = m;
                break;
            }
        }
        if (best) { *out = *best; return 0; }
        fprintf(stderr, "mode '%s' not advertised by connector; "
                        "use 'modeline:...' to force a custom mode\n", spec);
        return -1;
    }

    fprintf(stderr, "unrecognised mode spec '%s'\n", spec);
    return -1;
}

/* Exact refresh rate and vblank period derived from the mode's timings
 * (more precise than the integer drmModeModeInfo.vrefresh field). */
static void compute_timing(struct output *o)
{
    double pixels = (double)o->mode.htotal * (double)o->mode.vtotal;
    double hz = (pixels > 0.0)
              ? (double)o->mode.clock * 1000.0 / pixels   /* clock is kHz */
              : 0.0;
    if (o->mode.flags & DRM_MODE_FLAG_INTERLACE) hz *= 2.0;
    if (o->mode.flags & DRM_MODE_FLAG_DBLSCAN)   hz /= 2.0;
    o->target_hz        = hz;
    o->target_period_ns = (hz > 0.0) ? (uint64_t)(1e9 / hz + 0.5) : 0;
}

/* Human-readable label, e.g. "LVDS-1" or "LVDS-1+HDMI-A-1" (encoder clone). */
static void output_set_label(struct output *o)
{
    o->label[0] = '\0';
    for (int k = 0; k < o->num_connectors; k++) {
        char nm[32];
        connector_name(o->conns[k], nm, sizeof(nm));
        if (k) appendf(o->label, sizeof(o->label), "+");
        appendf(o->label, sizeof(o->label), "%s", nm);
    }
}

/* ------------------------------------------------------------------ */
/* Framebuffer wrapping for GBM buffer objects.                        */
/* ------------------------------------------------------------------ */

struct fb_wrap { uint32_t fb_id; int fd; };

static void bo_destroy_cb(struct gbm_bo *bo, void *data)
{
    (void)bo;
    struct fb_wrap *fw = data;
    if (fw) {
        if (fw->fb_id) drmModeRmFB(fw->fd, fw->fb_id);
        free(fw);
    }
}

static uint32_t fb_for_bo(int fd, struct gbm_bo *bo)
{
    struct fb_wrap *fw = gbm_bo_get_user_data(bo);
    if (fw) return fw->fb_id;

    fw = calloc(1, sizeof(*fw));
    fw->fd = fd;

    uint32_t w   = gbm_bo_get_width(bo);
    uint32_t h   = gbm_bo_get_height(bo);
    uint32_t fmt = gbm_bo_get_format(bo);
    uint32_t handles[4] = {0}, strides[4] = {0}, offsets[4] = {0};
    handles[0] = gbm_bo_get_handle(bo).u32;
    strides[0] = gbm_bo_get_stride(bo);

    if (drmModeAddFB2(fd, w, h, fmt, handles, strides, offsets,
                      &fw->fb_id, 0) != 0) {
        /* Fallback for older drivers / formats. */
        if (drmModeAddFB(fd, w, h, 24, 32, strides[0], handles[0],
                         &fw->fb_id) != 0) {
            fprintf(stderr, "drmModeAddFB(2) failed: %s\n", strerror(errno));
            free(fw);
            return 0;
        }
    }
    gbm_bo_set_user_data(bo, fw, bo_destroy_cb);
    return fw->fb_id;
}

/* ------------------------------------------------------------------ */
/* GL program setup.                                                   */
/* ------------------------------------------------------------------ */

static GLuint compile_shader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "shader compile error: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static int gl_setup(void)
{
    GLuint vs = compile_shader(GL_VERTEX_SHADER, VS);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FS);
    if (!vs || !fs) return -1;

    g_prog = glCreateProgram();
    glAttachShader(g_prog, vs);
    glAttachShader(g_prog, fs);
    glLinkProgram(g_prog);
    GLint ok = 0;
    glGetProgramiv(g_prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(g_prog, sizeof(log), NULL, log);
        fprintf(stderr, "program link error: %s\n", log);
        return -1;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    g_a_pos   = glGetAttribLocation(g_prog, "a_position");
    g_a_nrm   = glGetAttribLocation(g_prog, "a_normal");
    g_a_col   = glGetAttribLocation(g_prog, "a_color");
    g_u_mvp   = glGetUniformLocation(g_prog, "u_mvp");
    g_u_model = glGetUniformLocation(g_prog, "u_model");
    g_u_tint  = glGetUniformLocation(g_prog, "u_tint");

    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_cube), g_cube, GL_STATIC_DRAW);

    const GLsizei stride = 9 * sizeof(float);
    glVertexAttribPointer(g_a_pos, 3, GL_FLOAT, GL_FALSE, stride,
                          (void *)(0 * sizeof(float)));
    glVertexAttribPointer(g_a_nrm, 3, GL_FLOAT, GL_FALSE, stride,
                          (void *)(3 * sizeof(float)));
    glVertexAttribPointer(g_a_col, 3, GL_FLOAT, GL_FALSE, stride,
                          (void *)(6 * sizeof(float)));
    glEnableVertexAttribArray(g_a_pos);
    glEnableVertexAttribArray(g_a_nrm);
    glEnableVertexAttribArray(g_a_col);

    glEnable(GL_DEPTH_TEST);
    /* Back-face culling deliberately left off so winding never matters. */
    return 0;
}

/* ------------------------------------------------------------------ */
/* Per-output rendering.                                               */
/* ------------------------------------------------------------------ */

static void draw_output(struct output *o, double t)
{
    eglMakeCurrent(g_egl_dpy, o->surface, o->surface, g_egl_ctx);
    glViewport(0, 0, o->width, o->height);

    if (o->variant == 1) glClearColor(0.12f, 0.04f, 0.05f, 1.0f); /* warm */
    else                 glClearColor(0.04f, 0.05f, 0.12f, 1.0f); /* cool */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float ax, ay, az, speed, phase;
    float tint[3];
    if (o->variant == 1) {
        ax = 1.0f; ay = 0.3f; az = 0.2f; speed = -0.9f; phase = 1.0f;
        tint[0] = 1.0f; tint[1] = 0.7f; tint[2] = 0.55f;
    } else {
        ax = 0.3f; ay = 1.0f; az = 0.0f; speed = 0.7f; phase = 0.0f;
        tint[0] = tint[1] = tint[2] = 1.0f;
    }
    float angle = (float)t * speed + phase;

    float proj[16], view[16], model[16], pv[16], mvp[16];
    float aspect = (o->height > 0) ? (float)o->width / (float)o->height : 1.0f;
    mat4_perspective(proj, 50.0f * (float)M_PI / 180.0f, aspect, 0.1f, 100.0f);
    mat4_translate(view, 0.0f, 0.0f, -4.0f);
    mat4_rotate(model, angle, ax, ay, az);
    mat4_mul(pv, proj, view);
    mat4_mul(mvp, pv, model);

    glUseProgram(g_prog);
    glUniformMatrix4fv(g_u_mvp,   1, GL_FALSE, mvp);
    glUniformMatrix4fv(g_u_model, 1, GL_FALSE, model);
    glUniform3fv(g_u_tint, 1, tint);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    eglSwapBuffers(g_egl_dpy, o->surface);
}

/* ------------------------------------------------------------------ */
/* Page-flip handling.                                                 */
/* ------------------------------------------------------------------ */

/* Record a completed flip and, from the gap since the previous one,
 * estimate how many of this output's vblanks were skipped. The timestamp
 * is the kernel's vblank time (page-flip event) or now_ns() on the
 * non-event fallback path; only deltas matter, so the clock domain is
 * irrelevant. */
static void account_flip(struct output *o, uint64_t ts_ns)
{
    if (o->last_flip_ns && o->target_period_ns) {
        uint64_t interval = ts_ns - o->last_flip_ns;
        int frames = (int)((double)interval / (double)o->target_period_ns + 0.5);
        if (frames < 1) frames = 1;
        o->missed_window += (unsigned)(frames - 1);
    }
    o->last_flip_ns = ts_ns;
    o->flips_window++;
}

static void page_flip_handler(int fd, unsigned int frame, unsigned int sec,
                              unsigned int usec, void *data)
{
    (void)fd; (void)frame;
    struct output *o = data;
    o->flip_pending = 0;
    account_flip(o, (uint64_t)sec * 1000000000ull + (uint64_t)usec * 1000ull);
    if (o->prev_bo) gbm_surface_release_buffer(o->gs, o->prev_bo);
    o->prev_bo = o->next_bo;
    o->next_bo = NULL;
}

static int any_flip_pending(void)
{
    for (int i = 0; i < g_num_out; i++)
        if (g_out[i].flip_pending) return 1;
    return 0;
}

static int wait_for_flips(void)
{
    drmEventContext ev = { 0 };
    ev.version = 2;
    ev.page_flip_handler = page_flip_handler;

    while (any_flip_pending() && g_running) {
        struct pollfd pfd = { .fd = g_drm_fd, .events = POLLIN };
        int r = poll(&pfd, 1, 1000);
        if (r < 0) {
            if (errno == EINTR) return -1;     /* signal (Ctrl-C) */
            perror("poll");
            return -1;
        }
        if (r == 0) continue;                  /* timeout: keep waiting */
        if (pfd.revents & POLLIN)
            drmHandleEvent(g_drm_fd, &ev);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* EGL/GBM bring-up.                                                   */
/* ------------------------------------------------------------------ */

static EGLConfig choose_egl_config(EGLDisplay dpy, uint32_t gbm_format)
{
    static const EGLint attrs[] = {
        EGL_SURFACE_TYPE,    EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE,        8,
        EGL_GREEN_SIZE,      8,
        EGL_BLUE_SIZE,       8,
        EGL_ALPHA_SIZE,      0,
        EGL_DEPTH_SIZE,      16,
        EGL_NONE
    };
    EGLint count = 0;
    eglChooseConfig(dpy, attrs, NULL, 0, &count);
    if (count <= 0) return NULL;

    EGLConfig *cfgs = calloc(count, sizeof(EGLConfig));
    eglChooseConfig(dpy, attrs, cfgs, count, &count);

    EGLConfig chosen = cfgs[0];           /* sane default */
    for (int i = 0; i < count; i++) {
        EGLint vid = 0;
        if (eglGetConfigAttrib(dpy, cfgs[i], EGL_NATIVE_VISUAL_ID, &vid) &&
            (uint32_t)vid == gbm_format) {
            chosen = cfgs[i];
            break;
        }
    }
    free(cfgs);
    return chosen;
}

static int egl_init(void)
{
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_pd =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)
        eglGetProcAddress("eglGetPlatformDisplayEXT");

    if (get_pd)
        g_egl_dpy = get_pd(EGL_PLATFORM_GBM_KHR, g_gbm, NULL);
    else
        g_egl_dpy = eglGetDisplay((EGLNativeDisplayType)g_gbm);

    if (g_egl_dpy == EGL_NO_DISPLAY) {
        fprintf(stderr, "eglGetDisplay failed\n");
        return -1;
    }
    if (!eglInitialize(g_egl_dpy, NULL, NULL)) {
        fprintf(stderr, "eglInitialize failed\n");
        return -1;
    }
    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
        fprintf(stderr, "eglBindAPI failed\n");
        return -1;
    }

    g_egl_cfg = choose_egl_config(g_egl_dpy, GBM_FORMAT_XRGB8888);
    if (!g_egl_cfg) {
        fprintf(stderr, "no suitable EGL config\n");
        return -1;
    }

    static const EGLint ctx_attr[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE
    };
    g_egl_ctx = eglCreateContext(g_egl_dpy, g_egl_cfg, EGL_NO_CONTEXT, ctx_attr);
    if (g_egl_ctx == EGL_NO_CONTEXT) {
        fprintf(stderr, "eglCreateContext failed\n");
        return -1;
    }
    return 0;
}

static int output_create_surface(struct output *o)
{
    o->gs = gbm_surface_create(g_gbm, o->width, o->height,
                               GBM_FORMAT_XRGB8888,
                               GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
    if (!o->gs) {
        fprintf(stderr, "gbm_surface_create failed (%dx%d)\n",
                o->width, o->height);
        return -1;
    }
    o->surface = eglCreateWindowSurface(g_egl_dpy, g_egl_cfg,
                                        (EGLNativeWindowType)o->gs, NULL);
    if (o->surface == EGL_NO_SURFACE) {
        fprintf(stderr, "eglCreateWindowSurface failed\n");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Listing / cleanup.                                                  */
/* ------------------------------------------------------------------ */

static void list_connectors(int fd, drmModeRes *res)
{
    printf("Connectors:\n");
    for (int i = 0; i < res->count_connectors; i++) {
        drmModeConnector *c = drmModeGetConnector(fd, res->connectors[i]);
        if (!c) continue;
        char name[64];
        connector_name(c, name, sizeof(name));
        printf("  %-12s id=%-3u %-13s %s\n",
               name, c->connector_id,
               c->connection == DRM_MODE_CONNECTED ? "connected" :
               c->connection == DRM_MODE_DISCONNECTED ? "disconnected" : "unknown",
               conn_is_internal(c->connector_type) ? "(internal)" : "(external)");
        for (int m = 0; m < c->count_modes; m++) {
            drmModeModeInfo *mo = &c->modes[m];
            printf("        %-14s %4ux%-4u @ %3uHz%s\n",
                   mo->name, mo->hdisplay, mo->vdisplay, mo->vrefresh,
                   (mo->type & DRM_MODE_TYPE_PREFERRED) ? "  *preferred" : "");
        }
        drmModeFreeConnector(c);
    }
}

static void cleanup(void)
{
    /* Best-effort restore of the original CRTC config. */
    for (int i = 0; i < g_num_out; i++) {
        struct output *o = &g_out[i];
        if (o->saved_crtc && g_drm_fd >= 0) {
            drmModeCrtc *s = o->saved_crtc;
            drmModeSetCrtc(g_drm_fd, s->crtc_id, s->buffer_id, s->x, s->y,
                           o->connector_ids, o->num_connectors, &s->mode);
        }
    }

    if (g_egl_dpy != EGL_NO_DISPLAY)
        eglMakeCurrent(g_egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

    for (int i = 0; i < g_num_out; i++) {
        struct output *o = &g_out[i];
        if (o->prev_bo) gbm_surface_release_buffer(o->gs, o->prev_bo);
        if (o->next_bo) gbm_surface_release_buffer(o->gs, o->next_bo);
        if (o->surface != EGL_NO_SURFACE && g_egl_dpy != EGL_NO_DISPLAY)
            eglDestroySurface(g_egl_dpy, o->surface);
        if (o->gs) gbm_surface_destroy(o->gs);
        if (o->saved_crtc) drmModeFreeCrtc(o->saved_crtc);
        for (int k = 0; k < o->num_connectors; k++)
            if (o->conns[k]) drmModeFreeConnector(o->conns[k]);
    }

    if (g_egl_ctx != EGL_NO_CONTEXT) eglDestroyContext(g_egl_dpy, g_egl_ctx);
    if (g_egl_dpy != EGL_NO_DISPLAY) eglTerminate(g_egl_dpy);
    if (g_gbm)        gbm_device_destroy(g_gbm);
    if (g_drm_fd >= 0) close(g_drm_fd);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

enum clone_mode { MODE_CLONE_ENC, MODE_CLONE_COMP, MODE_INDEP };

static void usage(const char *argv0)
{
    fprintf(stderr,
"Usage: %s [options]\n"
"  -D, --device PATH      DRM device (default /dev/dri/card0)\n"
"  -m, --mode MODE        clone-encoder | clone-compositor | independent\n"
"      --connector0 NAME  output 0 connector (name or id; default auto internal)\n"
"      --connector1 NAME  output 1 connector (name or id; default auto external)\n"
"      --mode0 SPEC       mode for output 0\n"
"      --mode1 SPEC       mode for output 1\n"
"  -f, --frames N         render N frames then exit (0 = run forever)\n"
"  -s, --stats            print measured refresh + missed vblanks once a second\n"
"  -l, --list             list connectors and modes, then exit\n"
"  -h, --help             this help\n"
"\n"
"SPEC: 'preferred' | WxH | WxH@Refresh | 'modeline:<clk h hss hse ht v vss vse vt [flags]>'\n"
"For clone-encoder, --mode0 is applied to both connectors (they share a CRTC).\n",
        argv0);
}

int main(int argc, char **argv)
{
    const char *device      = "/dev/dri/card0";
    const char *conn0_name  = NULL;
    const char *conn1_name  = NULL;
    const char *mode0_spec  = NULL;
    const char *mode1_spec  = NULL;
    enum clone_mode kind    = MODE_CLONE_COMP;
    long max_frames         = 0;
    int  do_list            = 0;

    static struct option opts[] = {
        { "device",     required_argument, 0, 'D' },
        { "mode",       required_argument, 0, 'm' },
        { "connector0", required_argument, 0, 1000 },
        { "connector1", required_argument, 0, 1001 },
        { "mode0",      required_argument, 0, 1002 },
        { "mode1",      required_argument, 0, 1003 },
        { "frames",     required_argument, 0, 'f' },
        { "list",       no_argument,       0, 'l' },
        { "stats",      no_argument,       0, 's' },
        { "help",       no_argument,       0, 'h' },
        { 0, 0, 0, 0 }
    };

    int c;
    while ((c = getopt_long(argc, argv, "D:m:f:lhs", opts, NULL)) != -1) {
        switch (c) {
        case 'D': device = optarg; break;
        case 'm':
            if      (!strcmp(optarg, "clone-encoder"))    kind = MODE_CLONE_ENC;
            else if (!strcmp(optarg, "clone-compositor")) kind = MODE_CLONE_COMP;
            else if (!strcmp(optarg, "independent"))      kind = MODE_INDEP;
            else { fprintf(stderr, "unknown mode '%s'\n", optarg); return 1; }
            break;
        case 1000: conn0_name = optarg; break;
        case 1001: conn1_name = optarg; break;
        case 1002: mode0_spec = optarg; break;
        case 1003: mode1_spec = optarg; break;
        case 'f':  max_frames = strtol(optarg, NULL, 10); break;
        case 'l':  do_list = 1; break;
        case 's':  g_stats = 1; break;
        case 'h':  usage(argv[0]); return 0;
        default:   usage(argv[0]); return 1;
        }
    }

    signal(SIGINT,  on_sigint);
    signal(SIGTERM, on_sigint);

    g_drm_fd = open(device, O_RDWR | O_CLOEXEC);
    if (g_drm_fd < 0) {
        fprintf(stderr, "open %s: %s\n", device, strerror(errno));
        return 1;
    }
    /* Try to become DRM master (harmless if we already are). */
    drmSetMaster(g_drm_fd);

    drmModeRes *res = drmModeGetResources(g_drm_fd);
    if (!res) {
        fprintf(stderr, "drmModeGetResources failed: %s "
                        "(is this a KMS-capable card node?)\n", strerror(errno));
        close(g_drm_fd);
        return 1;
    }

    if (do_list) {
        list_connectors(g_drm_fd, res);
        drmModeFreeResources(res);
        close(g_drm_fd);
        return 0;
    }

    /* --- Find the two connectors. --- */
    drmModeConnector *c0 = find_connector(g_drm_fd, res, conn0_name, 1);
    drmModeConnector *c1 = find_connector(g_drm_fd, res, conn1_name, 0);
    if (!c0 || !c1) {
        fprintf(stderr, "could not find %s connector "
                        "(use --list to inspect, --connectorN to choose)\n",
                !c0 ? "output 0" : "output 1");
        if (c0) drmModeFreeConnector(c0);
        if (c1) drmModeFreeConnector(c1);
        drmModeFreeResources(res);
        close(g_drm_fd);
        return 1;
    }
    {
        char n0[64], n1[64];
        connector_name(c0, n0, sizeof(n0));
        connector_name(c1, n1, sizeof(n1));
        printf("output 0: %s (id %u)\noutput 1: %s (id %u)\n",
               n0, c0->connector_id, n1, c1->connector_id);
    }

    /* --- Build outputs and assign CRTCs. --- */
    if (kind == MODE_CLONE_ENC) {
        /* Single output, one CRTC, two connectors, one shared mode. */
        g_num_out = 1;
        struct output *o = &g_out[0];
        o->conns[0] = c0; o->connector_ids[0] = c0->connector_id;
        o->conns[1] = c1; o->connector_ids[1] = c1->connector_id;
        o->num_connectors = 2;
        o->variant = 0;

        if (pick_common_crtc(g_drm_fd, res, c0, c1, &o->crtc_id) != 0) {
            fprintf(stderr, "no CRTC can drive both connectors at once; "
                            "encoder-level clone unsupported by this hardware. "
                            "Try --mode clone-compositor.\n");
            goto fail;
        }
        if (select_mode(c0, mode0_spec, &o->mode) != 0) goto fail;
        o->width  = o->mode.hdisplay;
        o->height = o->mode.vdisplay;
        o->saved_crtc = get_connector_crtc(g_drm_fd, c0);
        compute_timing(o);
        output_set_label(o);

        printf("clone-encoder: CRTC %u -> both connectors, mode %s %dx%d@%.2fHz\n",
               o->crtc_id, o->mode.name, o->width, o->height, o->target_hz);
    } else {
        /* Two independent outputs, distinct CRTCs. */
        g_num_out = 2;
        uint32_t used = 0;

        struct output *o0 = &g_out[0];
        o0->conns[0] = c0; o0->connector_ids[0] = c0->connector_id;
        o0->num_connectors = 1;
        o0->variant = 0;
        if (pick_crtc(g_drm_fd, res, c0, &used, &o0->crtc_id) != 0) {
            fprintf(stderr, "no CRTC available for output 0\n"); goto fail;
        }
        if (select_mode(c0, mode0_spec, &o0->mode) != 0) goto fail;
        o0->width  = o0->mode.hdisplay;
        o0->height = o0->mode.vdisplay;
        o0->saved_crtc = get_connector_crtc(g_drm_fd, c0);
        compute_timing(o0);
        output_set_label(o0);

        struct output *o1 = &g_out[1];
        o1->conns[0] = c1; o1->connector_ids[0] = c1->connector_id;
        o1->num_connectors = 1;
        o1->variant = (kind == MODE_INDEP) ? 1 : 0;  /* different cube if indep */
        if (pick_crtc(g_drm_fd, res, c1, &used, &o1->crtc_id) != 0) {
            fprintf(stderr, "no CRTC available for output 1\n"); goto fail;
        }
        if (select_mode(c1, mode1_spec, &o1->mode) != 0) goto fail;
        o1->width  = o1->mode.hdisplay;
        o1->height = o1->mode.vdisplay;
        o1->saved_crtc = get_connector_crtc(g_drm_fd, c1);
        compute_timing(o1);
        output_set_label(o1);

        printf("%s: CRTC %u %s %dx%d@%.2fHz | CRTC %u %s %dx%d@%.2fHz\n",
               kind == MODE_INDEP ? "independent" : "clone-compositor",
               o0->crtc_id, o0->mode.name, o0->width, o0->height, o0->target_hz,
               o1->crtc_id, o1->mode.name, o1->width, o1->height, o1->target_hz);
    }

    /* c0/c1 are now owned by the outputs; don't free them separately. */
    drmModeFreeResources(res);
    res = NULL;

    /* --- GBM / EGL / GLES bring-up. --- */
    g_gbm = gbm_create_device(g_drm_fd);
    if (!g_gbm) { fprintf(stderr, "gbm_create_device failed\n"); goto fail; }
    if (egl_init() != 0) goto fail;

    for (int i = 0; i < g_num_out; i++)
        if (output_create_surface(&g_out[i]) != 0) goto fail;

    /* Make a surface current so we can build GL objects. */
    if (!eglMakeCurrent(g_egl_dpy, g_out[0].surface, g_out[0].surface,
                        g_egl_ctx)) {
        fprintf(stderr, "eglMakeCurrent failed\n");
        goto fail;
    }
    build_cube();
    if (gl_setup() != 0) goto fail;

    /* --- Initial modeset for each output. --- */
    for (int i = 0; i < g_num_out; i++) {
        struct output *o = &g_out[i];
        draw_output(o, 0.0);
        o->prev_bo = gbm_surface_lock_front_buffer(o->gs);
        if (!o->prev_bo) { fprintf(stderr, "lock_front_buffer failed\n"); goto fail; }
        uint32_t fb = fb_for_bo(g_drm_fd, o->prev_bo);
        if (!fb) goto fail;
        if (drmModeSetCrtc(g_drm_fd, o->crtc_id, fb, 0, 0,
                           o->connector_ids, o->num_connectors, &o->mode) != 0) {
            fprintf(stderr, "drmModeSetCrtc(crtc %u, %s) failed: %s\n",
                    o->crtc_id, o->mode.name, strerror(errno));
            goto fail;
        }
    }

    /* --- Main render / page-flip loop. --- */
    double t0 = now_sec();
    uint64_t last_report_ns = g_stats ? now_ns() : 0;
    long frame = 0;
    while (g_running && (max_frames == 0 || frame < max_frames)) {
        double t = now_sec() - t0;

        for (int i = 0; i < g_num_out; i++) {
            struct output *o = &g_out[i];
            draw_output(o, t);
            o->next_bo = gbm_surface_lock_front_buffer(o->gs);
            if (!o->next_bo) { fprintf(stderr, "lock_front_buffer failed\n"); goto fail; }
            uint32_t fb = fb_for_bo(g_drm_fd, o->next_bo);
            if (drmModePageFlip(g_drm_fd, o->crtc_id, fb,
                                DRM_MODE_PAGE_FLIP_EVENT, o) != 0) {
                /* Some drivers reject flips; fall back to a blocking modeset. */
                drmModeSetCrtc(g_drm_fd, o->crtc_id, fb, 0, 0,
                               o->connector_ids, o->num_connectors, &o->mode);
                account_flip(o, now_ns());
                if (o->prev_bo) gbm_surface_release_buffer(o->gs, o->prev_bo);
                o->prev_bo = o->next_bo;
                o->next_bo = NULL;
            } else {
                o->flip_pending = 1;
            }
        }

        if (wait_for_flips() != 0) break;  /* interrupted */
        frame++;

        /* Once per second, report measured refresh and skipped vblanks. */
        if (g_stats) {
            uint64_t tns = now_ns();
            if (tns - last_report_ns >= 1000000000ull) {
                double elapsed = (double)(tns - last_report_ns) / 1e9;
                char line[256];
                line[0] = '\0';
                appendf(line, sizeof(line), "[fps]");
                for (int i = 0; i < g_num_out; i++) {
                    struct output *o = &g_out[i];
                    double hz = (elapsed > 0.0) ? o->flips_window / elapsed : 0.0;
                    appendf(line, sizeof(line), " %s %6.2fHz (target %.2f",
                            o->label, hz, o->target_hz);
                    if (o->missed_window)
                        appendf(line, sizeof(line), ", %u missed", o->missed_window);
                    appendf(line, sizeof(line), ")");
                    if (i + 1 < g_num_out) appendf(line, sizeof(line), " |");
                    o->flips_window  = 0;
                    o->missed_window = 0;
                }
                printf("%s\n", line);
                fflush(stdout);
                last_report_ns = tns;
            }
        }
    }

    cleanup();
    return 0;

fail:
    if (res) drmModeFreeResources(res);
    cleanup();
    return 1;
}
