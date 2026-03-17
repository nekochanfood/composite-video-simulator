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
        # Generate native SASS for multiple GPU architecture generations.
        # sm_75 = Turing (GTX 1660, RTX 2060-2080) — minimum for CUDA 13.x
        # sm_86 = Ampere (RTX 3060-3090)
        # sm_89 = Ada Lovelace (RTX 4060-4090)
        # Older GPUs (GTX 10xx / Pascal) are not supported by CUDA 13.x
        # and will use the CPU fallback path automatically.
        set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
        message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER_VERSION}")
    else()
        message(WARNING "CUDA compiler not found. Building CPU-only version.")
        set(HAVE_CUDA FALSE)
    endif()
else()
    set(HAVE_CUDA FALSE)
    message(STATUS "CUDA disabled by user option.")
endif()

# ─── Find FFmpeg ─────────────────────────────────────────────────────────────
add_library(ffmpeg_deps INTERFACE)

if(WIN32)
    # ── Windows: find FFmpeg via FFMPEG_ROOT or environment ──────────────────
    # Set FFMPEG_ROOT to your FFmpeg install directory (must contain
    # include/ and lib/ subdirectories). Works with vcpkg, MSYS2, or
    # manually extracted shared/dev builds from gyan.dev / BtbN.
    #
    # Example:  cmake -DFFMPEG_ROOT=C:/ffmpeg ..
    #
    if(NOT FFMPEG_ROOT)
        # Try environment variable
        if(DEFINED ENV{FFMPEG_ROOT})
            set(FFMPEG_ROOT "$ENV{FFMPEG_ROOT}")
        endif()
    endif()

    if(FFMPEG_ROOT)
        message(STATUS "Using FFMPEG_ROOT = ${FFMPEG_ROOT}")
        list(APPEND CMAKE_PREFIX_PATH "${FFMPEG_ROOT}")
    endif()

    # Find individual libraries
    foreach(_lib avcodec avformat avutil swscale swresample)
        find_path(FF_${_lib}_INCLUDE_DIR "lib${_lib}/version.h"
            PATHS "${FFMPEG_ROOT}/include"
            PATH_SUFFIXES ffmpeg
        )
        find_library(FF_${_lib}_LIBRARY
            NAMES ${_lib} ${_lib}.lib
            PATHS "${FFMPEG_ROOT}/lib" "${FFMPEG_ROOT}/bin"
        )
        if(FF_${_lib}_INCLUDE_DIR AND FF_${_lib}_LIBRARY)
            message(STATUS "  Found ${_lib}: ${FF_${_lib}_LIBRARY}")
            target_include_directories(ffmpeg_deps INTERFACE "${FF_${_lib}_INCLUDE_DIR}")
            target_link_libraries(ffmpeg_deps INTERFACE "${FF_${_lib}_LIBRARY}")
        else()
            message(FATAL_ERROR "Could not find FFmpeg library: ${_lib}\n"
                "  Set -DFFMPEG_ROOT=<path> to your FFmpeg installation directory.")
        endif()
    endforeach()

else()
    # ── Linux: find FFmpeg via pkg-config ────────────────────────────────────
    # Use custom FFmpeg build at /usr/local with NVENC support (shared libraries).
    # This build includes h264_nvenc, hevc_nvenc, av1_nvenc encoders.
    find_package(PkgConfig REQUIRED)

    set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig")

    pkg_check_modules(AVCODEC    REQUIRED IMPORTED_TARGET libavcodec)
    pkg_check_modules(AVFORMAT   REQUIRED IMPORTED_TARGET libavformat)
    pkg_check_modules(AVUTIL     REQUIRED IMPORTED_TARGET libavutil)
    pkg_check_modules(SWSCALE    REQUIRED IMPORTED_TARGET libswscale)
    pkg_check_modules(SWRESAMPLE REQUIRED IMPORTED_TARGET libswresample)

    # Set RPATH so the binary finds /usr/local/lib at runtime.
    set(CMAKE_INSTALL_RPATH "/usr/local/lib")
    set(CMAKE_BUILD_RPATH "/usr/local/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    target_link_libraries(ffmpeg_deps INTERFACE
        PkgConfig::AVCODEC
        PkgConfig::AVFORMAT
        PkgConfig::AVUTIL
        PkgConfig::SWSCALE
        PkgConfig::SWRESAMPLE
    )
endif()

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

# ─── spout_ntsc (Windows-only real-time Spout2 proxy) ────────────────────────
if(WIN32)
    include(FetchContent)

    # Prevent Spout2's root CMakeLists.txt from setting global /MT
    set(SPOUT_BUILD_CMT OFF CACHE BOOL "" FORCE)
    # We need the DirectX helper (SpoutDX_static)
    set(SPOUT_BUILD_SPOUTDX ON CACHE BOOL "" FORCE)
    # We don't need the C wrapper library
    set(SPOUT_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        spout2
        GIT_REPOSITORY https://github.com/leadedge/Spout2.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(spout2)

    # spout_ntsc executable
    add_executable(spout_ntsc spout_ntsc.cpp)

    # SpoutDX_static includes: SpoutDX.h, Spout core, d3d11, dxgi, etc.
    # Use the source dirs for includes since installed namespaces aren't available
    target_include_directories(spout_ntsc PRIVATE
        "${spout2_SOURCE_DIR}/SPOUTSDK/SpoutDirectX"
        "${spout2_SOURCE_DIR}/SPOUTSDK/SpoutGL"
    )
    target_link_libraries(spout_ntsc PRIVATE
        ffmpeg_deps
        SpoutDX_static
    )

    # ── Optional Qt6 preview GUI ─────────────────────────────────────────────
    option(ENABLE_QT_GUI "Enable Qt6 preview window for spout_ntsc (-gui)" OFF)
    if(ENABLE_QT_GUI)
        find_package(Qt6 COMPONENTS Widgets REQUIRED)
        target_link_libraries(spout_ntsc PRIVATE Qt6::Widgets)
        target_compile_definitions(spout_ntsc PRIVATE HAVE_QT=1)
        # Enable Qt MOC for Q_OBJECT macros in .cpp file
        set_target_properties(spout_ntsc PROPERTIES AUTOMOC ON)
        message(STATUS "spout_ntsc: Qt6 GUI enabled")
    endif()

    # CUDA linkage is handled below in the CUDA section
endif()

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

    # Link CUDA kernels to spout_ntsc (Windows only)
    if(TARGET spout_ntsc)
        target_link_libraries(spout_ntsc PRIVATE ntsc_cuda CUDA::cudart)
    endif()

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
if(TARGET spout_ntsc)
    install(TARGETS spout_ntsc RUNTIME DESTINATION bin)
endif()
