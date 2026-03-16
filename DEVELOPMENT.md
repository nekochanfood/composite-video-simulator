# Composite Video Simulator - Development & Build Guide

Analog composite video signal simulator. Reproduces VHS and NTSC/PAL artifacts to generate retro-style video output.

## Requirements

### OS

- **Linux** (recommended, primary development platform)
- macOS (may work)
- Windows (may work via MSYS2 / WSL)

### Required Tools

| Tool | Version | Notes |
|------|---------|-------|
| C++17 compiler | GCC 7+ / Clang 5+ | `g++` or `clang++` |
| CMake | 3.18 or later | Recommended build system |
| pkg-config | - | Required for FFmpeg library detection |
| FFmpeg dev libraries | libavcodec >= 55.39, libavformat >= 55.19, libswscale >= 4.0, libavutil >= 52.48, libswresample >= 2.0 | Headers and shared libraries |

### Optional

| Tool | Notes |
|------|-------|
| NVIDIA CUDA Toolkit | GPU acceleration (enabled by default; builds CPU-only if unavailable) |
| Autotools (autoconf, automake) | Legacy build system (CMake recommended) |

## Installing Dependencies

### Ubuntu / Debian

```bash
# Required packages
sudo apt update
sudo apt install build-essential cmake pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev \
  libswscale-dev libswresample-dev

# CUDA (optional: GPU acceleration)
# Install NVIDIA drivers and CUDA Toolkit separately
```

### Fedora / RHEL

```bash
sudo dnf install gcc-c++ cmake pkgconfig \
  ffmpeg-devel
```

### macOS (Homebrew)

```bash
brew install cmake pkg-config ffmpeg
```

## Building

### CMake (Recommended)

```bash
# Create build directory
mkdir build && cd build

# Configure (CUDA auto-detected)
cmake ..

# To disable CUDA
# cmake .. -DENABLE_CUDA=OFF

# Build
cmake --build . -j$(nproc)
```

On success, the following executables are produced in the `build/` directory:

- `ffmpeg_ntsc`
- `ffmpeg_vhsled`
- `frameblend`
- `filmac`

### Install (Optional)

```bash
sudo cmake --install .
# Installs to /usr/local/bin by default
```

### Autotools (Legacy)

```bash
./autogen.sh
./configure
make -j$(nproc)
sudo make install
```

## CUDA Notes

- CMake auto-detects CUDA. If the CUDA Toolkit is not found, a CPU-only build is produced automatically.
- The default GPU architecture is `sm_89` (Ada Lovelace / RTX 4070 Super). For other GPUs, change `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` or pass it as a cmake argument:
  ```bash
  cmake .. -DCMAKE_CUDA_ARCHITECTURES=75  # Turing (RTX 2000 series)
  ```
- When CUDA is enabled, all four programs (`ffmpeg_ntsc`, `ffmpeg_vhsled`, `frameblend`, `filmac`) use GPU kernels.

## Usage

### ffmpeg_ntsc -- NTSC Composite Video Simulator (Main Tool)

Encodes and decodes video as an NTSC/PAL composite video signal, reproducing analog video color bleeding and noise. Output is 720x480 (NTSC 29.97fps) or 720x576 (PAL 25fps) interlaced video.

```bash
ffmpeg_ntsc [options]
```

#### Basic Options

| Option | Description |
|--------|-------------|
| `-i <file>` | Input file (can be specified multiple times) |
| `-o <file>` | Output file |
| `-tvstd <pal\|ntsc>` | TV standard (default: ntsc) |
| `-d <n>` | Video delay buffer (n frames) |
| `-ss <t>` | Start time in seconds |
| `-se <t>` | End time in seconds |
| `-t <t>` | Duration to process in seconds |
| `-422` | Output in 4:2:2 colorspace |
| `-420` | Output in 4:2:0 colorspace (default) |
| `-nocomp` | Transcode only, skip emulation |
| `-v <n>` | Select the n-th video stream |
| `-vn` | Disable video |
| `-a <n>` | Select the n-th audio stream |
| `-an` | Disable audio |
| `-vi` | Render interlaced at frame rate |
| `-vp` | Render progressive at field rate (bob filter) |
| `-width <x>` | Set output width |

#### VHS Emulation

| Option | Description |
|--------|-------------|
| `-vhs` | Enable VHS artifact emulation |
| `-vhs-speed <ep\|lp\|sp>` | Tape speed (default: sp) |
| `-vhs-hifi <0\|1>` | VHS Hi-Fi audio (default: on) |
| `-vhs-svideo <0\|1>` | Render as S-Video output from VHS |
| `-vhs-chroma-vblend <0\|1>` | Vertically blend chroma scanlines |
| `-vhs-head-switching <0\|1>` | Head switching emulation |
| `-vhs-head-switching-point <x>` | Head switching point (0..1) |
| `-vhs-head-switching-phase <x>` | Head switching displacement (-1..1) |
| `-vhs-head-switching-noise-level <x>` | Head switching noise variation |
| `-vhs-linear-high-boost <x>` | High-frequency boost for linear audio tracks |
| `-vhs-linear-video-crosstalk <x>` | Video crosstalk in audio, loudness in dBFS (0=100%) |

#### Noise & Signal Quality

| Option | Description |
|--------|-------------|
| `-noise <0..100>` | Noise amplitude |
| `-chroma-noise <0..100>` | Chroma noise amplitude |
| `-chroma-phase-noise <x>` | Chroma phase noise (0..100) |
| `-chroma-dropout <x>` | Chroma scanline dropouts (0..10000) |
| `-audio-hiss <-120..0>` | Audio hiss in decibels (0=100%) |
| `-subcarrier-amp <0..100>` | Subcarrier amplitude (% of luma) |

#### Composite Signal

| Option | Description |
|--------|-------------|
| `-comp-pre <s>` | Composite preemphasis scale |
| `-comp-cut <f>` | Composite preemphasis frequency |
| `-comp-catv` | CATV preset #1 |
| `-comp-catv2` | CATV preset #2 |
| `-comp-catv3` | CATV preset #3 |
| `-comp-catv4` | CATV preset #4 |
| `-comp-phase <n>` | Subcarrier phase per scanline (0, 90, 180, or 270) |
| `-preemphasis <0\|1>` | Enable preemphasis emulation |
| `-deemphasis <0\|1>` | Enable deemphasis emulation |
| `-yc-recomb <n>` | Recombine Y/C n times |
| `-in-composite-lowpass <n>` | Chroma lowpass on composite input |
| `-out-composite-lowpass <n>` | Chroma lowpass on composite output |
| `-out-composite-lowpass-lite <n>` | Chroma lowpass on composite output (lite) |

#### Debug

| Option | Description |
|--------|-------------|
| `-nocolor-subcarrier` | Emulate color subcarrier but do not decode back |
| `-nocolor-subcarrier-after-yc-sep` | Emulate Y/C separation but do not decode back |
| `-bkey-feedback <n>` | Black key feedback (black level <= n) |

#### Examples

```bash
# Basic NTSC VHS simulation
./ffmpeg_ntsc -i input.mp4 -o output.mp4 -vhs

# VHS EP mode (lowest quality)
./ffmpeg_ntsc -i input.mp4 -o output.mp4 -vhs -vhs-speed ep

# PAL standard output
./ffmpeg_ntsc -i input.mp4 -o output.mp4 -tvstd pal -vhs

# Heavy noise settings
./ffmpeg_ntsc -i input.mp4 -o output.mp4 -vhs -noise 20 -chroma-noise 30

# CATV-style preemphasis
./ffmpeg_ntsc -i input.mp4 -o output.mp4 -comp-catv

# Process only the first 30 seconds
./ffmpeg_ntsc -i input.mp4 -o output.mp4 -vhs -t 30
```

---

### ffmpeg_vhsled -- VHS Black Border Detection & Correction

Detects and corrects the black border on the left edge of VHS tape output (head drum artifact).

```bash
ffmpeg_vhsled [options]
```

| Option | Description |
|--------|-------------|
| `-i <file>` | Input file |
| `-o <file>` | Output file |
| `-or <fps>` | Output frame rate (e.g. `29.97`, `30000/1001`) |
| `-width <x>` | Output width |
| `-height <x>` | Output height |
| `-fa <x>` | Interpolate alternate frames |
| `-gamma <x>` | Gamma correction (number, `ntsc`, or `vga`) |
| `-underscan <x>` | Underscan amount (0..99) |
| `-422` | 4:2:2 colorspace |
| `-420` | 4:2:0 colorspace |

```bash
# Correct VHS left-edge black border
./ffmpeg_vhsled -i vhs_capture.mp4 -o fixed.mp4
```

---

### frameblend -- Multi-Frame Weighted Blending

Blends multiple frames with weighted averaging for frame rate conversion and motion blur effects.

```bash
frameblend [options]
```

| Option | Description |
|--------|-------------|
| `-i <file>` | Input file (can be specified multiple times) |
| `-o <file>` | Output file |
| `-or <fps>` | Output frame rate |
| `-width <x>` | Output width |
| `-height <x>` | Output height |
| `-sqnr` | Squelch frame interpolation when frame rates match (1% margin) |
| `-ffa` | Full frame alternate interpolation |
| `-fa <x>` | Interpolate alternate frames (1..8) |
| `-gamma <x>` | Gamma correction (number, `ntsc`, or `vga`) |
| `-underscan <x>` | Underscan amount (0..99) |
| `-422` | 4:2:2 colorspace |
| `-420` | 4:2:0 colorspace |

```bash
# Convert 60fps to 24fps with frame blending
./frameblend -i input_60fps.mp4 -o output_24fps.mp4 -or 24

# Blend with gamma correction
./frameblend -i input.mp4 -o output.mp4 -or 30 -gamma ntsc
```

---

### filmac -- Film Auto-Contrast / Levels

Automatically adjusts contrast and levels for film scan footage.

```bash
filmac [options]
```

| Option | Description |
|--------|-------------|
| `-i <file>` | Input file |
| `-o <file>` | Output file |
| `-or <fps>` | Output frame rate |
| `-width <x>` | Output width |
| `-height <x>` | Output height |
| `-fa <x>` | Interpolate alternate frames |
| `-gamma <x>` | Gamma correction (number, `ntsc`, or `vga`) |
| `-underscan <x>` | Underscan amount (0..99) |
| `-422` | 4:2:2 colorspace |
| `-420` | 4:2:0 colorspace |

```bash
# Auto-level correction for film scan footage
./filmac -i film_scan.mp4 -o corrected.mp4
```

## Project Structure

```
composite-video-simulator/
├── CMakeLists.txt           # CMake build definition (recommended)
├── configure.ac             # Autotools configuration (legacy)
├── Makefile.am              # Autotools Makefile (legacy)
├── autogen.sh               # Autotools bootstrap script
│
├── ffmpeg_ntsc.cpp          # NTSC composite video simulator (main)
├── ffmpeg_vhsled.cpp        # VHS black border detection/correction
├── frameblend.cpp           # Multi-frame blending
├── filmac.cpp               # Film auto-contrast correction
│
├── common.h                 # Shared header (FFmpeg includes, macros, utilities)
├── color_convert.h          # RGB/YIQ/YUV color space conversion
├── lowpass_filter.h         # Lowpass/highpass filters
├── input_file.h             # Input file management (FFmpeg demuxer)
├── output_context.h         # Output context management (FFmpeg muxer)
│
├── ntsc_cuda.cu / .h        # CUDA kernels for ffmpeg_ntsc
├── vhsled_cuda.cu / .h      # CUDA kernels for ffmpeg_vhsled
├── frameblend_cuda.cu / .h  # CUDA kernels for frameblend
├── filmac_cuda.cu / .h      # CUDA kernels for filmac
│
├── README                   # Project overview
├── COPYING                  # GPLv3 license
└── .gitignore               # Git ignore rules
```

## Notes

- Processing is **slow**. The design prioritizes accuracy over speed. It is recommended to trim your video to the needed segments before processing rather than running entire files through the simulator.
- Output is automatically scaled to 720x480 (NTSC 29.97fps interlaced) or 720x576 (PAL 25fps interlaced).

## License

GNU General Public License v3 (GPLv3)
