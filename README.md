# drmcube

This is a utility kind of like kmscube, but with more of a focus on testing two display outputs.

It uses a combination of libdrm, gbm, EGL, and GLESv2 to render a rotating 3D cube. As such, this should be run without any kind of compositor running. You must be the DRM master.

It has 3 main modes of operation:

1. Display outputs may be cloned at the hardware level using a shared encoder. This requires the displays share the exact same mode.
2. Display outputs may be "cloned" by simply rendering the same scene twice to each output. This allows the mixing display modes and refresh rates.
3. Display outputs may be independent. In this case, the scene is rendered with different colour/rotation on each output so that it is obvious each display is operating independently.

You may also launch the program with the `--stats` option to see if you're missing vblank intervals.

Additionally, manual modelines may be specified so that you can pick modes your display doesn't necessarily support.

For a more detailed explanation, see the comments at the start of main.c.

I had Claude write this program so I could do some testing. It seems to do pretty much what I want. Maybe it's helpful for you too.

As such, I'm throwing the Unlicense on here. Do with this software what you will brave one.

For those that are in a hurry:

```
cc -O2 -Wall -Wextra main.c -o drmcube \
    $(pkg-config --cflags --libs libdrm gbm egl glesv2) -lm
```
