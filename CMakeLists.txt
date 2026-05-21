cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(drmcube)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBDRM REQUIRED IMPORTED_TARGET libdrm)
pkg_check_modules(GBM REQUIRED IMPORTED_TARGET gbm)

find_package(OpenGL REQUIRED COMPONENTS EGL GLES2)

add_executable(drmcube main.c)

target_link_libraries(drmcube PRIVATE
  PkgConfig::LIBDRM
  PkgConfig::GBM
  OpenGL::EGL
  OpenGL::GLES2
  m
)
