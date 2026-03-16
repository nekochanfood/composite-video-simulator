#include "filmac_cuda.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>

// ─── Error-check helpers ────────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        return;                                                     \
    }                                                               \
} while (0)

#define CUDA_CHECK_BOOL(call) do {                                  \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        return false;                                               \
    }                                                               \
} while (0)

// ─── Static CUDA resources ──────────────────────────────────────────────────
static bool s_cuda_ready = false;

static unsigned char*  d_input     = nullptr;  // device input BGRA frame
static unsigned char*  d_output    = nullptr;  // device output BGRA frame
static long*           d_longframe = nullptr;  // device intermediate (3 longs per pixel)
static unsigned long*  d_gamma_dec = nullptr;  // 256 entries
static unsigned long*  d_gamma_enc = nullptr;  // 8193 entries

// Block reduction results (host + device)
// Grid of 128x128 blocks covering the scan region.
// Each block produces one min and one max value.
static long* d_block_min = nullptr;
static long* d_block_max = nullptr;
static long* h_block_min = nullptr;
static long* h_block_max = nullptr;
static int   s_num_blocks = 0;

static size_t s_input_alloc  = 0;
static size_t s_output_alloc = 0;
static size_t s_long_alloc   = 0;

// ─── Init / Shutdown ────────────────────────────────────────────────────────

bool filmac_cuda_init() {
    int device_count = 0;
    CUDA_CHECK_BOOL(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "filmac_cuda: no CUDA devices found\n");
        return false;
    }

    CUDA_CHECK_BOOL(cudaSetDevice(0));

    s_cuda_ready = true;
    fprintf(stderr, "filmac_cuda: GPU acceleration enabled (device 0)\n");
    return true;
}

void filmac_cuda_shutdown() {
    if (!s_cuda_ready) return;

    if (d_input)     { cudaFree(d_input);     d_input     = nullptr; }
    if (d_output)    { cudaFree(d_output);     d_output    = nullptr; }
    if (d_longframe) { cudaFree(d_longframe);  d_longframe = nullptr; }
    if (d_gamma_dec) { cudaFree(d_gamma_dec);  d_gamma_dec = nullptr; }
    if (d_gamma_enc) { cudaFree(d_gamma_enc);  d_gamma_enc = nullptr; }
    if (d_block_min) { cudaFree(d_block_min);  d_block_min = nullptr; }
    if (d_block_max) { cudaFree(d_block_max);  d_block_max = nullptr; }
    if (h_block_min) { cudaFreeHost(h_block_min); h_block_min = nullptr; }
    if (h_block_max) { cudaFreeHost(h_block_max); h_block_max = nullptr; }

    s_input_alloc = s_output_alloc = s_long_alloc = 0;
    s_num_blocks = 0;
    s_cuda_ready = false;
}

// ─── Ensure buffers ─────────────────────────────────────────────────────────
static bool ensure_buffers(int width, int height, int linesize_in, int linesize_out, bool use_gamma) {
    size_t in_bytes  = (size_t)linesize_in  * height;
    size_t out_bytes = (size_t)linesize_out * height;
    size_t long_bytes = (size_t)width * height * 3 * sizeof(long);

    if (in_bytes > s_input_alloc) {
        if (d_input) cudaFree(d_input);
        CUDA_CHECK_BOOL(cudaMalloc(&d_input, in_bytes));
        s_input_alloc = in_bytes;
    }
    if (out_bytes > s_output_alloc) {
        if (d_output) cudaFree(d_output);
        CUDA_CHECK_BOOL(cudaMalloc(&d_output, out_bytes));
        s_output_alloc = out_bytes;
    }
    if (long_bytes > s_long_alloc) {
        if (d_longframe) cudaFree(d_longframe);
        CUDA_CHECK_BOOL(cudaMalloc(&d_longframe, long_bytes));
        s_long_alloc = long_bytes;
    }

    // Gamma LUTs
    if (use_gamma) {
        if (!d_gamma_dec) {
            CUDA_CHECK_BOOL(cudaMalloc(&d_gamma_dec, 256 * sizeof(unsigned long)));
        }
        if (!d_gamma_enc) {
            CUDA_CHECK_BOOL(cudaMalloc(&d_gamma_enc, 8193 * sizeof(unsigned long)));
        }
    }

    // Block reduction buffers
    // The CPU code scans blocks of 128x128 in the region:
    //   x: [15% .. 90%) of width, y: [0% .. 100%) of height
    int minx = (width * 15) / 100;
    int maxx = (width * 90) / 100;
    int miny = 0;
    int maxy = height;
    int blw = 128, blh = 128;

    int blocks_x = (maxx - minx + blw - 1) / blw;
    int blocks_y = (maxy - miny + blh - 1) / blh;
    int num_blocks = blocks_x * blocks_y;

    if (num_blocks > s_num_blocks) {
        if (d_block_min) cudaFree(d_block_min);
        if (d_block_max) cudaFree(d_block_max);
        if (h_block_min) cudaFreeHost(h_block_min);
        if (h_block_max) cudaFreeHost(h_block_max);

        CUDA_CHECK_BOOL(cudaMalloc(&d_block_min, num_blocks * sizeof(long)));
        CUDA_CHECK_BOOL(cudaMalloc(&d_block_max, num_blocks * sizeof(long)));
        CUDA_CHECK_BOOL(cudaMallocHost(&h_block_min, num_blocks * sizeof(long)));
        CUDA_CHECK_BOOL(cudaMallocHost(&h_block_max, num_blocks * sizeof(long)));
        s_num_blocks = num_blocks;
    }

    return true;
}

// ─── Kernel 1: Decode BGRA -> longframe (gamma or linear) ──────────────────

__global__ void kernel_decode_gamma(
    const unsigned char* __restrict__ input,
    long*                __restrict__ longframe,
    const unsigned long* __restrict__ gamma_dec,
    int width, int height, int linesize_in)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const unsigned char* pixel = input + y * linesize_in + x * 4;
    long* out = longframe + (y * width + x) * 3;

    unsigned long bv = pixel[0]; if (bv > 255u) bv = 255u;
    unsigned long gv = pixel[1]; if (gv > 255u) gv = 255u;
    unsigned long rv = pixel[2]; if (rv > 255u) rv = 255u;

    out[0] = (long)(gamma_dec[bv] << 16ul);
    out[1] = (long)(gamma_dec[gv] << 16ul);
    out[2] = (long)(gamma_dec[rv] << 16ul);
}

__global__ void kernel_decode_linear(
    const unsigned char* __restrict__ input,
    long*                __restrict__ longframe,
    int width, int height, int linesize_in)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const unsigned char* pixel = input + y * linesize_in + x * 4;
    long* out = longframe + (y * width + x) * 3;

    out[0] = (long)((unsigned long)pixel[0] << 16ul);
    out[1] = (long)((unsigned long)pixel[1] << 16ul);
    out[2] = (long)((unsigned long)pixel[2] << 16ul);
}

// ─── Kernel 2: Block min/max reduction ──────────────────────────────────────
// Each CUDA block processes one 128x128 tile of the image.
// Uses shared memory reduction to find min(min(B,G,R)) and max(max(B,G,R)).

// We launch 128 threads per block (one per row of the 128x128 tile).
// Each thread scans 128 pixels, then we do a shared-memory reduction.

#define BLOCK_TILE 128

__global__ void kernel_block_minmax(
    const long* __restrict__ longframe,
    long*       __restrict__ block_min_out,
    long*       __restrict__ block_max_out,
    int width, int height,
    int scan_minx, int scan_miny,
    int blocks_x)
{
    // blockIdx.x = block column, blockIdx.y = block row
    int tile_x = scan_minx + blockIdx.x * BLOCK_TILE;
    int tile_y = scan_miny + blockIdx.y * BLOCK_TILE;

    // threadIdx.x = row within the 128-row tile
    int row = tile_y + threadIdx.x;

    // Thread-local min/max
    long local_min_sum = 0;  // sum of per-pixel min for averaging
    long local_max = -0x7FFFFFFFL;
    long local_count = 0;

    if (row < height) {
        for (int sx = 0; sx < BLOCK_TILE; sx++) {
            int px = tile_x + sx;
            if (px >= width || row >= height) continue;

            const long* p = longframe + (row * width + px) * 3;
            long b = p[0], g = p[1], r = p[2];

            long pmin = b < g ? b : g; pmin = pmin < r ? pmin : r;
            long pmax = b > g ? b : g; pmax = pmax > r ? pmax : r;

            local_count++;
            local_min_sum += pmin;
            if (local_max < pmax) local_max = pmax;
        }
    }

    // Shared memory for reduction
    __shared__ long s_min_sum[BLOCK_TILE];
    __shared__ long s_max[BLOCK_TILE];
    __shared__ long s_count[BLOCK_TILE];

    s_min_sum[threadIdx.x] = local_min_sum;
    s_max[threadIdx.x] = local_max;
    s_count[threadIdx.x] = local_count;
    __syncthreads();

    // Tree reduction
    for (int stride = BLOCK_TILE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_min_sum[threadIdx.x] += s_min_sum[threadIdx.x + stride];
            s_count[threadIdx.x] += s_count[threadIdx.x + stride];
            if (s_max[threadIdx.x] < s_max[threadIdx.x + stride])
                s_max[threadIdx.x] = s_max[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int block_id = blockIdx.y * blocks_x + blockIdx.x;
        // Compute average min for this block (matching CPU: grmin = (grmin + grd/2) / grd)
        long grd = s_count[0];
        long grmin = 0;
        if (grd > 0) {
            grmin = (s_min_sum[0] + grd / 2) / grd;
        }
        block_min_out[block_id] = grmin;
        block_max_out[block_id] = s_max[0];
    }
}

// ─── Kernel 3: Contrast stretch ─────────────────────────────────────────────

__global__ void kernel_contrast_stretch(
    long* __restrict__ longframe,
    int width, int height,
    long final_minv, long range,   // range = final_maxv - final_minv
    long scaleto)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    long* p = longframe + (y * width + x) * 3;

    for (int c = 0; c < 3; c++) {
        long long v = ((long long)(p[c] - final_minv) * (long long)scaleto) / (long long)range;
        if (v < -0x7FFFFFFFL) v = -0x7FFFFFFFL;
        if (v >  0x7FFFFFFFL) v =  0x7FFFFFFFL;
        p[c] = (long)v;
    }
}

// ─── Kernel 4: Encode longframe -> BGRA (gamma or linear) ──────────────────

__global__ void kernel_encode_gamma(
    const long*          __restrict__ longframe,
    unsigned char*       __restrict__ output,
    const unsigned long* __restrict__ gamma_enc,
    int width, int height, int linesize_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const long* p = longframe + (y * width + x) * 3;
    unsigned char* out = output + y * linesize_out + x * 4;

    for (int c = 0; c < 3; c++) {
        long v = p[c] >> 16L;
        if (v < 0) v = 0;
        if (v > 8192) v = 8192;
        int enc = (int)gamma_enc[(unsigned long)v];
        if (enc > 255) enc = 255;
        if (enc < 0) enc = 0;
        out[c] = (unsigned char)enc;
    }
    out[3] = 0xFF;
}

__global__ void kernel_encode_linear(
    const long*    __restrict__ longframe,
    unsigned char* __restrict__ output,
    int width, int height, int linesize_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const long* p = longframe + (y * width + x) * 3;
    unsigned char* out = output + y * linesize_out + x * 4;

    for (int c = 0; c < 3; c++) {
        long v = p[c] >> 16L;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        out[c] = (unsigned char)v;
    }
    out[3] = 0xFF;
}

// ─── Host wrapper ───────────────────────────────────────────────────────────

void filmac_cuda_process(
    const unsigned char* input,
    unsigned char* output,
    int width, int height,
    int linesize_in, int linesize_out,
    bool use_gamma,
    const unsigned long* gamma_dec_lut,
    const unsigned long* gamma_enc_lut,
    long* final_minv,
    long* final_maxv,
    bool* final_init)
{
    if (!s_cuda_ready) return;

    if (!ensure_buffers(width, height, linesize_in, linesize_out, use_gamma)) return;

    size_t in_bytes  = (size_t)linesize_in * height;
    size_t out_bytes = (size_t)linesize_out * height;

    // Upload input frame
    CUDA_CHECK(cudaMemcpy(d_input, input, in_bytes, cudaMemcpyHostToDevice));

    // Upload gamma LUTs (once per call — they never change between frames,
    // but this is cheap and keeps the code simple)
    if (use_gamma) {
        CUDA_CHECK(cudaMemcpy(d_gamma_dec, gamma_dec_lut, 256 * sizeof(unsigned long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma_enc, gamma_enc_lut, 8193 * sizeof(unsigned long), cudaMemcpyHostToDevice));
    }

    // ── Stage 1: Decode to longframe ────────────────────────────────────────
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        if (use_gamma) {
            kernel_decode_gamma<<<grid, block>>>(d_input, d_longframe, d_gamma_dec,
                                                  width, height, linesize_in);
        } else {
            kernel_decode_linear<<<grid, block>>>(d_input, d_longframe,
                                                   width, height, linesize_in);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Stage 2: Block min/max reduction ────────────────────────────────────
    long scaleto = use_gamma ? (0x10000L * 8192L) : (0x10000L * 256L);
    long minv = (scaleto * 6L) / 10L;
    long maxv = (scaleto * 4L) / 10L;

    {
        int scan_minx = (width * 15) / 100;
        int scan_maxx = (width * 90) / 100;
        int scan_miny = 0;
        int scan_maxy = height;

        int blocks_x = (scan_maxx - scan_minx + BLOCK_TILE - 1) / BLOCK_TILE;
        int blocks_y = (scan_maxy - scan_miny + BLOCK_TILE - 1) / BLOCK_TILE;
        int num_blocks = blocks_x * blocks_y;

        // Launch: one CUDA block per 128x128 tile, 128 threads per block (one per row)
        dim3 grid_red(blocks_x, blocks_y);
        dim3 block_red(BLOCK_TILE);

        kernel_block_minmax<<<grid_red, block_red>>>(
            d_longframe, d_block_min, d_block_max,
            width, height,
            scan_minx, scan_miny, blocks_x);
        CUDA_CHECK(cudaGetLastError());

        // Download block results
        CUDA_CHECK(cudaMemcpy(h_block_min, d_block_min, num_blocks * sizeof(long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_block_max, d_block_max, num_blocks * sizeof(long), cudaMemcpyDeviceToHost));

        // CPU reduction over blocks (matches the CPU code exactly)
        for (int i = 0; i < num_blocks; i++) {
            if (minv > h_block_min[i])
                minv = h_block_min[i];
            if (maxv < h_block_max[i])
                maxv = h_block_max[i];
        }
    }

    if (minv == maxv) maxv++;

    // ── Stage 3: Temporal EMA (CPU) ─────────────────────────────────────────
    if (!*final_init) {
        *final_init = true;
        *final_minv = minv;
        *final_maxv = maxv;
    } else {
        if (*final_maxv < maxv)
            *final_maxv = ((*final_maxv * 1L) + maxv) / 2L;
        else
            *final_maxv = ((*final_maxv * 4L) + maxv) / 5L;

        if (*final_minv > minv)
            *final_minv = ((*final_minv * 1L) + minv) / 2L;
        else
            *final_minv = ((*final_minv * 4L) + minv) / 5L;
    }

    long range = *final_maxv - *final_minv;
    if (range == 0) range = 1;

    // ── Stage 4: Contrast stretch (GPU) ─────────────────────────────────────
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        kernel_contrast_stretch<<<grid, block>>>(
            d_longframe, width, height,
            *final_minv, range, scaleto);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Stage 5: Encode longframe -> BGRA (GPU) ────────────────────────────
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        if (use_gamma) {
            kernel_encode_gamma<<<grid, block>>>(
                d_longframe, d_output, d_gamma_enc,
                width, height, linesize_out);
        } else {
            kernel_encode_linear<<<grid, block>>>(
                d_longframe, d_output,
                width, height, linesize_out);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Download result
    CUDA_CHECK(cudaMemcpy(output, d_output, out_bytes, cudaMemcpyDeviceToHost));
}
