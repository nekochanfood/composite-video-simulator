# toolchain-mingw64.cmake — Cross-compile from Linux/WSL targeting Windows x86_64
#
# Usage from WSL/Linux:
#   cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-mingw64.cmake \
#         -DFFMPEG_ROOT=/path/to/win64-ffmpeg \
#         ..
#
# Prerequisites (Ubuntu/Debian):
#   sudo apt install mingw-w64 g++-mingw-w64-x86-64
#
# For Spout2 (DirectX): cross-compilation of spout_ntsc requires the
# Windows SDK DirectX headers. These are included with mingw-w64.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Compiler
set(CMAKE_C_COMPILER   x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_RC_COMPILER  x86_64-w64-mingw32-windres)

# Search paths: only look in cross-compile sysroot, not host
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Static linking of libgcc/libstdc++ so the binary is self-contained
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++" CACHE STRING "" FORCE)
