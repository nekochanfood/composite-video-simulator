cmake_minimum_required(VERSION 3.18)
project(composite-video-simulator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Optional CUDA support ───────────────────────────────────────────────────
option(ENABLE_CUDA "Enable CUDA GPU acceleration" ON)

if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(HAVE_CUDA TRUE)
        # Generate native SASS for the target GPU architecture.
        # sm_89 = Ada Lovelace (RTX 4070 Super). This avoids PTX JIT
        # compatibility issues when toolkit version > driver CUDA version.
        # Add more architectures here for broader GPU support.
        set(CMAKE_CUDA_ARCHITECTURES 89)
        message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER_VERSION}")
    else()
        message(WARNING "CUDA compiler not found. Building CPU-only version.")
        set(HAVE_CUDA FALSE)
    endif()
else()
    set(HAVE_CUDA FALSE)
    message(STATUS "CUDA disabled by user option.")
endif()

# ─── Find FFmpeg via pkg-config ──────────────────────────────────────────────
# Use custom FFmpeg build at /usr/local with NVENC support (shared libraries).
# This build includes h264_nvenc, hevc_nvenc, av1_nvenc encoders.
find_package(PkgConfig REQUIRED)

set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig")

pkg_check_modules(AVCODEC    REQUIRED IMPORTED_TARGET libavcodec)
pkg_check_modules(AVFORMAT   REQUIRED IMPORTED_TARGET libavformat)
pkg_check_modules(AVUTIL     REQUIRED IMPORTED_TARGET libavutil)
pkg_check_modules(SWSCALE    REQUIRED IMPORTED_TARGET libswscale)
pkg_check_modules(SWRESAMPLE REQUIRED IMPORTED_TARGET libswresample)

# ─── Common FFmpeg link helper ───────────────────────────────────────────────
# Use custom FFmpeg at /usr/local (built with NVENC support).
# Set RPATH so the binary finds /usr/local/lib at runtime.
set(CMAKE_INSTALL_RPATH "/usr/local/lib")
set(CMAKE_BUILD_RPATH "/usr/local/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library(ffmpeg_deps INTERFACE)
target_link_libraries(ffmpeg_deps INTERFACE
    PkgConfig::AVCODEC
    PkgConfig::AVFORMAT
    PkgConfig::AVUTIL
    PkgConfig::SWSCALE
    PkgConfig::SWRESAMPLE
)

# ─── Compile definitions ─────────────────────────────────────────────────────
if(HAVE_CUDA)
    add_compile_definitions(HAVE_CUDA=1)
endif()

# ─── Active programs ─────────────────────────────────────────────────────────

# ffmpeg_ntsc - the flagship NTSC composite video simulator
add_executable(ffmpeg_ntsc ffmpeg_ntsc.cpp)
target_link_libraries(ffmpeg_ntsc PRIVATE ffmpeg_deps)

# ffmpeg_vhsled - VHS black border detection/correction
add_executable(ffmpeg_vhsled ffmpeg_vhsled.cpp)
target_link_libraries(ffmpeg_vhsled PRIVATE ffmpeg_deps)

# frameblend - multi-frame weighted blending
add_executable(frameblend frameblend.cpp)
target_link_libraries(frameblend PRIVATE ffmpeg_deps)

# filmac - film auto-contrast/levels
add_executable(filmac filmac.cpp)
target_link_libraries(filmac PRIVATE ffmpeg_deps)

# ─── CUDA kernel libraries ───────────────────────────────────────────────────
if(HAVE_CUDA)
    # frameblend CUDA kernels
    add_library(frameblend_cuda STATIC frameblend_cuda.cu)
    target_link_libraries(frameblend_cuda PRIVATE CUDA::cudart)
    set_target_properties(frameblend_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )

    # Link CUDA kernels to frameblend
    target_link_libraries(frameblend PRIVATE frameblend_cuda CUDA::cudart)

    # filmac CUDA kernels
    add_library(filmac_cuda STATIC filmac_cuda.cu)
    target_link_libraries(filmac_cuda PRIVATE CUDA::cudart)
    set_target_properties(filmac_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )

    # Link CUDA kernels to filmac
    target_link_libraries(filmac PRIVATE filmac_cuda CUDA::cudart)

    # ntsc CUDA kernels (composite video simulation on GPU)
    add_library(ntsc_cuda STATIC ntsc_cuda.cu)
    target_link_libraries(ntsc_cuda PRIVATE CUDA::cudart)
    set_target_properties(ntsc_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )

    # Link CUDA kernels to ffmpeg_ntsc
    target_link_libraries(ffmpeg_ntsc PRIVATE ntsc_cuda CUDA::cudart)

    # vhsled CUDA kernels (VHS left-edge detection on GPU)
    add_library(vhsled_cuda STATIC vhsled_cuda.cu)
    target_link_libraries(vhsled_cuda PRIVATE CUDA::cudart)
    set_target_properties(vhsled_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )

    # Link CUDA kernels to ffmpeg_vhsled
    target_link_libraries(ffmpeg_vhsled PRIVATE vhsled_cuda CUDA::cudart)
endif()

# ─── Install ─────────────────────────────────────────────────────────────────
install(TARGETS ffmpeg_ntsc ffmpeg_vhsled frameblend filmac
    RUNTIME DESTINATION bin
)
