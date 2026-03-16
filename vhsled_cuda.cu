// vhsled_cuda.cu — CUDA kernels for VHS left-edge detection and correction
//
// Architecture:
//   3 stages, all launched as GPU kernels:
//   1. kernel_detect_left_edge  — per-scanline: scan from left, find first non-black column
//   2. kernel_smooth_adj        — per-scanline: 9-tap vertical box filter on edge positions
//   3. kernel_shift_scanline    — per-scanline: copy pixels shifted left by smoothed edge amount
//
//   Source and destination BGRA frames are uploaded/downloaded via PCIe.
//   Intermediate adj[] / adj2[] arrays stay on GPU.

#include "vhsled_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// ─── Helper: check CUDA errors ──────────────────────────────────────────────

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

// ─── Persistent GPU state ────────────────────────────────────────────────────

static uint8_t* d_src = nullptr;
static uint8_t* d_dst = nullptr;
static int32_t* d_adj = nullptr;
static int32_t* d_adj2 = nullptr;

static size_t gpu_frame_size = 0;
static int gpu_height = 0;

// ─── Device helper: blackish comparison ─────────────────────────────────────
// Returns true if pixel p is "blackish" relative to reference r.
// Each color channel difference must be < 16.

__device__ bool device_blackish(uint32_t p, uint32_t r) {
    for (unsigned int l = 0; l < 3; l++) {
        int c = (int)(p & 0xFFu) - (int)(r & 0xFFu);
        if (c >= 16) return false;
        p >>= 8u;
        r >>= 8u;
    }
    return true;
}

// ─── Kernel 1: Detect left edge per scanline ────────────────────────────────
// Each thread handles one scanline. Scans from left to find the first column
// where 8+ consecutive non-black pixels appear. Stores result in adj[y] as
// 16.16 fixed-point (x << 16).

__global__ void kernel_detect_left_edge(
    const uint8_t* __restrict__ src,
    int src_linesize,
    int width, int height,
    int32_t* __restrict__ adj)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    const uint32_t* row = (const uint32_t*)(src + src_linesize * y);
    uint32_t ref = row[0];  // reference pixel (leftmost, assumed black)

    unsigned int x = 0;
    unsigned int bc = 0;
    int count = width;

    while (count > 0) {
        if (!device_blackish(row[x], ref)) {
            if (bc >= 8) {
                // Found 8+ consecutive non-black: back up to start of run
                x -= bc;
                break;
            } else {
                bc++;
            }
        } else {
            bc = 0;
        }
        count--;
        x++;
    }

    adj[y] = (int32_t)x << 16;
}

// ─── Kernel 2: Vertical 9-tap box filter smoothing ──────────────────────────
// adj2[y] = average of adj[y-4..y+4] for y in [4, height-5].
// Border rows (y < 4 or y >= height-4) are copied unchanged.

__global__ void kernel_smooth_adj(
    const int32_t* __restrict__ adj,
    int32_t* __restrict__ adj2,
    int height)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    if (y >= 4 && y < height - 4) {
        adj2[y] = (adj[y-4] + adj[y-3] + adj[y-2] +
                   adj[y-1] + adj[y]   + adj[y+1] +
                   adj[y+2] + adj[y+3] + adj[y+4] + 5) / 9;
    } else {
        adj2[y] = adj[y];
    }
}

// ─── Kernel 3: Shift scanline horizontally ──────────────────────────────────
// For each scanline, compute shift x from adj2[y], then copy shifted pixels
// from src to dst. Matches the CPU logic:
//   x = (adj2[y] + 0x8000) >> 16
//   if x < 0: x = 0
//   if x >= width/2: just copy src to dst unshifted (continue in CPU)
//   else: copy (width - x) pixels from src+x to dst

__global__ void kernel_shift_scanline(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int src_linesize, int dst_linesize,
    int width, int height,
    const int32_t* __restrict__ adj2)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    const uint32_t* src_row = (const uint32_t*)(src + src_linesize * y);
    uint32_t* dst_row = (uint32_t*)(dst + dst_linesize * y);

    // First, copy entire source row to destination (memcpy equivalent)
    for (int i = 0; i < width; i++) {
        dst_row[i] = src_row[i];
    }

    int x = (adj2[y] + 0x8000) >> 16;
    if (x < 0) x = 0;
    if (x >= (int)(width / 2u)) return;  // skip shift for this line

    // Overwrite dst with shifted source pixels
    int copy_count = width - x;
    if (copy_count > 0) {
        for (int i = 0; i < copy_count; i++) {
            dst_row[i] = src_row[x + i];
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Host API
// ═════════════════════════════════════════════════════════════════════════════

bool vhsled_cuda_init() {
    // Check for a CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < 1) {
        fprintf(stderr, "vhsled_cuda_init: no CUDA device found\n");
        return false;
    }

    fprintf(stderr, "VHS LED CUDA initialized (lazy allocation on first frame)\n");
    return true;
}

void vhsled_cuda_shutdown() {
    if (d_src) { cudaFree(d_src); d_src = nullptr; }
    if (d_dst) { cudaFree(d_dst); d_dst = nullptr; }
    if (d_adj) { cudaFree(d_adj); d_adj = nullptr; }
    if (d_adj2) { cudaFree(d_adj2); d_adj2 = nullptr; }
    gpu_frame_size = 0;
    gpu_height = 0;
    fprintf(stderr, "VHS LED CUDA shutdown\n");
}

void vhsled_cuda_process(
    const unsigned char* src,
    unsigned char* dst,
    int width, int height,
    int src_linesize, int dst_linesize)
{
    // Lazy allocation / reallocation if frame size changed
    size_t needed_frame = (size_t)src_linesize * height;
    if (needed_frame > gpu_frame_size || height > gpu_height) {
        // Free old buffers
        if (d_src) cudaFree(d_src);
        if (d_dst) cudaFree(d_dst);
        if (d_adj) cudaFree(d_adj);
        if (d_adj2) cudaFree(d_adj2);

        // Use max of src/dst linesize for allocation
        size_t max_linesize = (src_linesize > dst_linesize) ? src_linesize : dst_linesize;
        gpu_frame_size = max_linesize * height;
        gpu_height = height;

        CUDA_CHECK(cudaMalloc(&d_src, gpu_frame_size));
        CUDA_CHECK(cudaMalloc(&d_dst, gpu_frame_size));
        CUDA_CHECK(cudaMalloc(&d_adj, height * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_adj2, height * sizeof(int32_t)));
    }

    // Upload source frame to GPU
    CUDA_CHECK(cudaMemcpy(d_src, src, (size_t)src_linesize * height, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (height + threads - 1) / threads;

    // Stage 1: Detect left edge per scanline
    kernel_detect_left_edge<<<blocks, threads>>>(
        d_src, src_linesize, width, height, d_adj);

    // Stage 2: Smooth adj[] with 9-tap vertical box filter
    kernel_smooth_adj<<<blocks, threads>>>(d_adj, d_adj2, height);

    // Stage 3: Shift each scanline horizontally
    kernel_shift_scanline<<<blocks, threads>>>(
        d_src, d_dst, src_linesize, dst_linesize, width, height, d_adj2);

    // Download result to host
    CUDA_CHECK(cudaMemcpy(dst, d_dst, (size_t)dst_linesize * height, cudaMemcpyDeviceToHost));
}
