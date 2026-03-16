#include "frameblend_cuda.h"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// ─── Device constants ────────────────────────────────────────────────────────
// Max frames we support blending in one call (the CPU code typically uses 1-3).
#define MAX_BLEND_FRAMES 8

// ─── Static CUDA resources ──────────────────────────────────────────────────
static bool         s_cuda_ready = false;

// Device-side LUTs (only allocated when gamma path is used)
static unsigned long* d_gamma_dec = nullptr;   // 256 entries
static unsigned long* d_gamma_enc = nullptr;   // 8193 entries

// Pinned host staging buffer for frame pointers -> device pointers
static unsigned char* d_frame_data[MAX_BLEND_FRAMES] = {};
static unsigned int*  d_weights = nullptr;
static unsigned char* d_output  = nullptr;

// Current allocation sizes (to avoid re-alloc every frame)
static size_t s_frame_alloc_size = 0;   // per-frame byte size
static size_t s_output_alloc_size = 0;

// ─── Error-check helper ─────────────────────────────────────────────────────
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

// ─── Init / Shutdown ────────────────────────────────────────────────────────

bool frameblend_cuda_init() {
    int device_count = 0;
    CUDA_CHECK_BOOL(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "frameblend_cuda: no CUDA devices found\n");
        return false;
    }

    CUDA_CHECK_BOOL(cudaSetDevice(0));

    // Pre-allocate weight buffer (MAX_BLEND_FRAMES entries)
    CUDA_CHECK_BOOL(cudaMalloc(&d_weights, MAX_BLEND_FRAMES * sizeof(unsigned int)));

    // LUTs allocated lazily on first gamma call
    s_cuda_ready = true;
    fprintf(stderr, "frameblend_cuda: GPU acceleration enabled (device 0)\n");
    return true;
}

void frameblend_cuda_shutdown() {
    if (!s_cuda_ready) return;

    for (int i = 0; i < MAX_BLEND_FRAMES; i++) {
        if (d_frame_data[i]) { cudaFree(d_frame_data[i]); d_frame_data[i] = nullptr; }
    }
    if (d_weights)   { cudaFree(d_weights);   d_weights   = nullptr; }
    if (d_output)    { cudaFree(d_output);     d_output    = nullptr; }
    if (d_gamma_dec) { cudaFree(d_gamma_dec);  d_gamma_dec = nullptr; }
    if (d_gamma_enc) { cudaFree(d_gamma_enc);  d_gamma_enc = nullptr; }

    s_frame_alloc_size = 0;
    s_output_alloc_size = 0;
    s_cuda_ready = false;
}

// ─── Internal helpers ───────────────────────────────────────────────────────

// Ensure device frame buffers are large enough
static bool ensure_frame_buffers(int num_frames, size_t frame_bytes, size_t output_bytes) {
    // Resize per-frame device buffers if needed
    if (frame_bytes > s_frame_alloc_size) {
        for (int i = 0; i < MAX_BLEND_FRAMES; i++) {
            if (d_frame_data[i]) cudaFree(d_frame_data[i]);
            d_frame_data[i] = nullptr;
        }
        for (int i = 0; i < MAX_BLEND_FRAMES; i++) {
            CUDA_CHECK_BOOL(cudaMalloc(&d_frame_data[i], frame_bytes));
        }
        s_frame_alloc_size = frame_bytes;
    }

    // Resize output buffer if needed
    if (output_bytes > s_output_alloc_size) {
        if (d_output) cudaFree(d_output);
        CUDA_CHECK_BOOL(cudaMalloc(&d_output, output_bytes));
        s_output_alloc_size = output_bytes;
    }

    return true;
}

// ─── CUDA Kernels ───────────────────────────────────────────────────────────

// Gamma-corrected blend kernel.
// Each thread processes one pixel (x, y).
// frame_ptrs[i] points to device memory for frame i.
// All frames have the same linesize_in, output has linesize_out.
__global__ void kernel_blend_gamma(
    const unsigned char* const* __restrict__ frame_ptrs,
    const unsigned int*         __restrict__ weights,
    int num_frames,
    unsigned char*              __restrict__ output,
    int width, int height,
    int linesize_in, int linesize_out,
    const unsigned long* __restrict__ gamma_dec,   // 256 entries
    const unsigned long* __restrict__ gamma_enc)   // 8193 entries
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    unsigned long long r = 0, g = 0, b = 0;

    for (int wi = 0; wi < num_frames; wi++) {
        const unsigned char* inpixel = frame_ptrs[wi] + y * linesize_in + x * 4;
        unsigned long long w = weights[wi];

        unsigned long bv = inpixel[0]; if (bv > 255u) bv = 255u;
        unsigned long gv = inpixel[1]; if (gv > 255u) gv = 255u;
        unsigned long rv = inpixel[2]; if (rv > 255u) rv = 255u;

        b += gamma_dec[bv] * w;
        g += gamma_dec[gv] * w;
        r += gamma_dec[rv] * w;
    }

    // gamma_enc maps 0..8192 -> 0..255
    unsigned long b16 = (unsigned long)(b >> 16ull); if (b16 > 8192u) b16 = 8192u;
    unsigned long g16 = (unsigned long)(g >> 16ull); if (g16 > 8192u) g16 = 8192u;
    unsigned long r16 = (unsigned long)(r >> 16ull); if (r16 > 8192u) r16 = 8192u;

    int ob = (int)gamma_enc[b16]; if (ob > 255) ob = 255; if (ob < 0) ob = 0;
    int og = (int)gamma_enc[g16]; if (og > 255) og = 255; if (og < 0) og = 0;
    int or_ = (int)gamma_enc[r16]; if (or_ > 255) or_ = 255; if (or_ < 0) or_ = 0;

    unsigned char* outpixel = output + y * linesize_out + x * 4;
    outpixel[0] = (unsigned char)ob;
    outpixel[1] = (unsigned char)og;
    outpixel[2] = (unsigned char)or_;
    outpixel[3] = 0xFF;
}

// Linear blend kernel (no gamma correction).
__global__ void kernel_blend_linear(
    const unsigned char* const* __restrict__ frame_ptrs,
    const unsigned int*         __restrict__ weights,
    int num_frames,
    unsigned char*              __restrict__ output,
    int width, int height,
    int linesize_in, int linesize_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    unsigned long r = 0, g = 0, b = 0;

    for (int wi = 0; wi < num_frames; wi++) {
        const unsigned char* inpixel = frame_ptrs[wi] + y * linesize_in + x * 4;
        unsigned long w = weights[wi];

        b += inpixel[0] * w;
        g += inpixel[1] * w;
        r += inpixel[2] * w;
    }

    int ob = (int)(b >> 16ul); if (ob > 255) ob = 255; if (ob < 0) ob = 0;
    int og = (int)(g >> 16ul); if (og > 255) og = 255; if (og < 0) og = 0;
    int or_ = (int)(r >> 16ul); if (or_ > 255) or_ = 255; if (or_ < 0) or_ = 0;

    unsigned char* outpixel = output + y * linesize_out + x * 4;
    outpixel[0] = (unsigned char)ob;
    outpixel[1] = (unsigned char)og;
    outpixel[2] = (unsigned char)or_;
    outpixel[3] = 0xFF;
}

// ─── Host wrapper: gamma path ───────────────────────────────────────────────

void frameblend_cuda_gamma(
    const unsigned char* const* frame_ptrs,
    const unsigned int* weights,
    int num_frames,
    unsigned char* output,
    int width, int height,
    int linesize_in, int linesize_out,
    const unsigned long* gamma_dec_lut,
    const unsigned long* gamma_enc_lut)
{
    if (!s_cuda_ready || num_frames <= 0 || num_frames > MAX_BLEND_FRAMES) return;

    size_t frame_bytes  = (size_t)linesize_in  * height;
    size_t output_bytes = (size_t)linesize_out * height;

    if (!ensure_frame_buffers(num_frames, frame_bytes, output_bytes)) return;

    // Upload gamma LUTs (allocate on first use)
    if (!d_gamma_dec) {
        CUDA_CHECK(cudaMalloc(&d_gamma_dec, 256 * sizeof(unsigned long)));
    }
    if (!d_gamma_enc) {
        CUDA_CHECK(cudaMalloc(&d_gamma_enc, 8193 * sizeof(unsigned long)));
    }
    CUDA_CHECK(cudaMemcpy(d_gamma_dec, gamma_dec_lut, 256 * sizeof(unsigned long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma_enc, gamma_enc_lut, 8193 * sizeof(unsigned long), cudaMemcpyHostToDevice));

    // Upload input frames
    for (int i = 0; i < num_frames; i++) {
        CUDA_CHECK(cudaMemcpy(d_frame_data[i], frame_ptrs[i], frame_bytes, cudaMemcpyHostToDevice));
    }

    // Upload weights
    CUDA_CHECK(cudaMemcpy(d_weights, weights, num_frames * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Build device pointer array and upload it
    unsigned char* h_frame_ptrs[MAX_BLEND_FRAMES];
    for (int i = 0; i < num_frames; i++) {
        h_frame_ptrs[i] = d_frame_data[i];
    }
    unsigned char** d_frame_ptrs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_frame_ptrs, num_frames * sizeof(unsigned char*)));
    CUDA_CHECK(cudaMemcpy(d_frame_ptrs, h_frame_ptrs, num_frames * sizeof(unsigned char*), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    kernel_blend_gamma<<<grid, block>>>(
        (const unsigned char* const*)d_frame_ptrs,
        d_weights,
        num_frames,
        d_output,
        width, height,
        linesize_in, linesize_out,
        d_gamma_dec,
        d_gamma_enc);

    CUDA_CHECK(cudaGetLastError());

    // Download result
    CUDA_CHECK(cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_frame_ptrs);
}

// ─── Host wrapper: linear path ──────────────────────────────────────────────

void frameblend_cuda_linear(
    const unsigned char* const* frame_ptrs,
    const unsigned int* weights,
    int num_frames,
    unsigned char* output,
    int width, int height,
    int linesize_in, int linesize_out)
{
    if (!s_cuda_ready || num_frames <= 0 || num_frames > MAX_BLEND_FRAMES) return;

    size_t frame_bytes  = (size_t)linesize_in  * height;
    size_t output_bytes = (size_t)linesize_out * height;

    if (!ensure_frame_buffers(num_frames, frame_bytes, output_bytes)) return;

    // Upload input frames
    for (int i = 0; i < num_frames; i++) {
        CUDA_CHECK(cudaMemcpy(d_frame_data[i], frame_ptrs[i], frame_bytes, cudaMemcpyHostToDevice));
    }

    // Upload weights
    CUDA_CHECK(cudaMemcpy(d_weights, weights, num_frames * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Build device pointer array
    unsigned char* h_frame_ptrs[MAX_BLEND_FRAMES];
    for (int i = 0; i < num_frames; i++) {
        h_frame_ptrs[i] = d_frame_data[i];
    }
    unsigned char** d_frame_ptrs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_frame_ptrs, num_frames * sizeof(unsigned char*)));
    CUDA_CHECK(cudaMemcpy(d_frame_ptrs, h_frame_ptrs, num_frames * sizeof(unsigned char*), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    kernel_blend_linear<<<grid, block>>>(
        (const unsigned char* const*)d_frame_ptrs,
        d_weights,
        num_frames,
        d_output,
        width, height,
        linesize_in, linesize_out);

    CUDA_CHECK(cudaGetLastError());

    // Download result
    CUDA_CHECK(cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_frame_ptrs);
}
