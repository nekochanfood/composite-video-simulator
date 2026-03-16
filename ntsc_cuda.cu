// ntsc_cuda.cu — CUDA kernels for NTSC composite video simulation
// GPU replacement for composite_layer() in ffmpeg_ntsc.cpp
//
// Performance optimizations:
//   1. CUDA stream for async execution (overlap with CPU encode)
//   2. Pinned host staging buffers for async memory transfers
//   3. Merged per-scanline IIR kernels to reduce launch overhead
//      - kernel_scanline_pre_composite: chroma LP in + subcarrier encode + preemphasis + luma noise
//      - kernel_scanline_post_composite: chroma from luma + chroma noise + phase noise
//      - kernel_scanline_vhs: VHS luma LP + chroma LP + sharpen (merged)
//      - kernel_scanline_vhs_reencode: VHS re-encode + re-decode (merged)
//   4. Async H2D/D2H via cudaMemcpy2DAsync on the compute stream
//
// Kernel launch order for one composite_layer() call:
//   1. async H2D upload
//   2. kernel_rgb_to_yiq                  — per-pixel (2D grid)
//   3. kernel_scanline_pre_composite      — per-scanline (merged IIR stages)
//   4. kernel_vhs_head_switching          — single thread (if VHS)
//   5. kernel_scanline_post_composite     — per-scanline (merged decode+noise)
//   6. kernel_scanline_vhs               — per-scanline (if VHS, merged LP+sharpen)
//   7. kernel_vhs_chroma_vert_blend      — single thread (if VHS+NTSC)
//   8. kernel_scanline_vhs_reencode      — per-scanline (if VHS+!svideo, merged)
//   9. kernel_chroma_dropout             — per-scanline (if enabled)
//  10. kernel_chroma_lowpass_out         — per-scanline (if enabled)
//  11. kernel_yiq_to_rgb                 — per-pixel (2D grid)
//  12. async D2H download

#include "ntsc_cuda.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── Device constants ────────────────────────────────────────────────────────

// Subcarrier lookup tables (same as CPU: Umult[4], Vmult[4])
__constant__ int8_t d_Umult[4] = { 1, 0, -1, 0 };
__constant__ int8_t d_Vmult[4] = { 0, 1, 0, -1 };

// ─── Persistent GPU state ────────────────────────────────────────────────────

// Double-buffered state: one set per pipeline slot
struct BufferSet {
    int* d_fY = nullptr;
    int* d_fI = nullptr;
    int* d_fQ = nullptr;
    uint8_t* d_src_bgra = nullptr;
    uint8_t* d_dst_bgra = nullptr;
    cudaStream_t stream = nullptr;
    
    // Info for async callback to copy result
    uint8_t* host_dst = nullptr;
    int host_stride = 0;
};

static BufferSet buffers[NTSC_CUDA_NUM_BUFFERS];
static curandState* d_rand_states = nullptr;  // shared across buffers (one per scanline)

static size_t gpu_frame_bgra_size = 0;  // width * height * 4
static int gpu_max_scanlines = 0;       // height / 2

// ─── Helper: check CUDA errors ──────────────────────────────────────────────

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

// ─── Device inline: IIR lowpass filter (matches LowpassFilter class) ────────

struct DeviceLowpass {
    double alpha;
    double prev;

    __device__ void init(double rate, double cutoff, double initial = 0.0) {
        double timeInterval = 1.0 / rate;
        double tau = 1.0 / (cutoff * 2.0 * M_PI);
        alpha = timeInterval / (tau + timeInterval);
        prev = initial;
    }

    __device__ double lowpass(double sample) {
        double stage1 = sample * alpha;
        double stage2 = prev - (prev * alpha);
        prev = stage1 + stage2;
        return prev;
    }

    __device__ double highpass(double sample) {
        double stage1 = sample * alpha;
        double stage2 = prev - (prev * alpha);
        prev = stage1 + stage2;
        return sample - prev;
    }
};

// ─── Device inline: color conversion (matches color_convert.h) ──────────────

__device__ void device_RGB_to_YIQ(int &Y, int &I, int &Q, int r, int g, int b) {
    double dY = (0.30 * r) + (0.59 * g) + (0.11 * b);
    Y = (int)(256.0 * dY);
    I = (int)(256.0 * ((-0.27 * (b - dY)) + (0.74 * (r - dY))));
    Q = (int)(256.0 * ((0.41 * (b - dY)) + (0.48 * (r - dY))));
}

__device__ void device_YIQ_to_RGB(int &r, int &g, int &b, int Y, int I, int Q) {
    r = (int)(((1.000 * Y) + (0.956 * I) + (0.621 * Q)) / 256.0);
    g = (int)(((1.000 * Y) + (-0.272 * I) + (-0.647 * Q)) / 256.0);
    b = (int)(((1.000 * Y) + (-1.106 * I) + (1.703 * Q)) / 256.0);
    if (r < 0) r = 0; else if (r > 255) r = 255;
    if (g < 0) g = 0; else if (g > 255) g = 255;
    if (b < 0) b = 0; else if (b > 255) b = 255;
}

// ─── Device inline: compute subcarrier phase index for a scanline ───────────

__device__ unsigned int device_subcarrier_xi(
    int video_scanline_phase_shift, int video_scanline_phase_shift_offset,
    unsigned long long fieldno, int y)
{
    unsigned int xi;
    if (video_scanline_phase_shift == 90)
        xi = (fieldno + video_scanline_phase_shift_offset + (y >> 1)) & 3;
    else if (video_scanline_phase_shift == 180)
        xi = (((fieldno + y) & 2) + video_scanline_phase_shift_offset) & 3;
    else if (video_scanline_phase_shift == 270)
        xi = (fieldno + video_scanline_phase_shift_offset - (y >> 1)) & 3;
    else
        xi = video_scanline_phase_shift_offset & 3;
    return xi;
}

// ─── Kernel 1: RGB to YIQ ───────────────────────────────────────────────────
// One thread per pixel in the active field lines.

__global__ void kernel_rgb_to_yiq(
    const uint8_t* __restrict__ src_bgra, int src_stride,
    int* __restrict__ fY, int* __restrict__ fI, int* __restrict__ fQ,
    int width, int height, unsigned int field, unsigned char opposite)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int scanline_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int y = field + scanline_idx * 2;

    if (x >= width || y >= height) return;

    int src_y = y + opposite;
    if (src_y >= height) src_y = height - 1;

    const uint32_t* src_row = (const uint32_t*)(src_bgra + src_stride * src_y);
    uint32_t pixel = src_row[x];

    int r = (pixel >> 16) & 0xFF;
    int g = (pixel >> 8) & 0xFF;
    int b = (pixel >> 0) & 0xFF;

    int idx = y * width + x;
    device_RGB_to_YIQ(fY[idx], fI[idx], fQ[idx], r, g, b);
}

// ─── Kernel 2+3+4+5 MERGED: Pre-composite scanline processing ──────────────
// One thread per scanline.  Performs (in order, if enabled):
//   - Chroma lowpass (input)        [3 cascaded IIR on I, 3 on Q]
//   - Chroma into luma (subcarrier encode)
//   - Composite preemphasis         [1 IIR highpass]
//   - Luma noise                    [running accumulator + cuRAND]
//
// Merging these avoids 4 separate kernel launches and their associated overhead.
// All operate on the same scanline data so merging has good data locality.

__global__ void kernel_scanline_pre_composite(
    int* __restrict__ fY, int* __restrict__ fI, int* __restrict__ fQ,
    int width, int height, unsigned int field, unsigned long long fieldno,
    // subcarrier
    int subcarrier_amplitude,
    int video_scanline_phase_shift, int video_scanline_phase_shift_offset,
    // noise
    int video_noise,
    // preemphasis
    double composite_preemphasis, double composite_preemphasis_cut,
    // flags
    bool do_chroma_lowpass, bool do_preemphasis, bool do_noise,
    // cuRAND states
    curandState* __restrict__ rand_states)
{
    int scanline_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = field + scanline_idx * 2;
    if (y >= height) return;

    int* Y = fY + (y * width);
    int* I = fI + (y * width);
    int* Q = fQ + (y * width);

    const double rate = (315000000.0 * 4) / 88;  // ~14.318 MHz * 4

    // ── Stage A: Chroma lowpass (input) ──
    if (do_chroma_lowpass) {
        for (int p = 0; p < 2; p++) {
            int* P = (p == 0 ? I : Q);
            double cutoff = (p == 0) ? 1300000.0 : 600000.0;
            int delay = (p == 0) ? 2 : 4;

            DeviceLowpass lp[3];
            for (int f = 0; f < 3; f++)
                lp[f].init(rate, cutoff, 0.0);

            for (int x = 0; x < width; x++) {
                double s = P[x];
                for (int f = 0; f < 3; f++) s = lp[f].lowpass(s);
                if (x >= delay) P[x - delay] = (int)s;
            }
        }
    }

    // ── Stage B: Chroma into luma (subcarrier encode) ──
    {
        unsigned int xi = device_subcarrier_xi(video_scanline_phase_shift,
                                                video_scanline_phase_shift_offset, fieldno, y);
        for (int x = 0; x < width; x++) {
            unsigned int sxi = xi + x;
            int chroma = I[x] * subcarrier_amplitude * d_Umult[sxi & 3];
            chroma += Q[x] * subcarrier_amplitude * d_Vmult[sxi & 3];
            Y[x] += chroma / 50;
            I[x] = 0;
            Q[x] = 0;
        }
    }

    // ── Stage C: Composite preemphasis ──
    if (do_preemphasis) {
        DeviceLowpass pre;
        pre.init(rate, composite_preemphasis_cut, 16.0);

        for (int x = 0; x < width; x++) {
            double s = Y[x];
            s += pre.highpass(s) * composite_preemphasis;
            Y[x] = (int)s;
        }
    }

    // ── Stage D: Luma noise ──
    if (do_noise) {
        curandState localState = rand_states[scanline_idx];
        int noise_mod = (video_noise * 2) + 1;
        int noise = 0;

        for (int x = 0; x < width; x++) {
            Y[x] += noise;
            noise += ((int)(curand(&localState) % noise_mod)) - video_noise;
            noise /= 2;
        }

        rand_states[scanline_idx] = localState;
    }
}

// ─── Kernel 6: VHS head switching ───────────────────────────────────────────
// Single thread — sequential across scanlines (inter-line dependency).

__global__ void kernel_vhs_head_switching(
    int* __restrict__ fY,
    int width, int height, unsigned int field,
    bool output_ntsc,
    double vhs_head_switching_point,
    double vhs_head_switching_phase,
    double vhs_head_switching_phase_noise,
    curandState* __restrict__ rand_states)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int twidth = width + (width / 10);

    double noise = 0;
    if (vhs_head_switching_phase_noise != 0) {
        curandState st = rand_states[0];
        unsigned int r1 = curand(&st);
        unsigned int r2 = curand(&st);
        unsigned int r3 = curand(&st);
        unsigned int r4 = curand(&st);
        unsigned long long xv = (unsigned long long)r1 * r2 * r3 * r4;
        xv %= 2000000000ULL;
        noise = ((double)xv / 1000000000.0) - 1.0;
        noise *= vhs_head_switching_phase_noise;
        rand_states[0] = st;
    }

    double t;
    if (output_ntsc)
        t = twidth * 262.5;
    else
        t = twidth * 312.5;

    unsigned int p_point = (unsigned int)(fmod(vhs_head_switching_point + noise, 1.0) * t);
    int y_start = ((p_point / twidth) * 2) + field;

    unsigned int p_phase = (unsigned int)(fmod(vhs_head_switching_phase + noise, 1.0) * t);
    unsigned int x_start = p_phase % twidth;

    if (output_ntsc)
        y_start -= (262 - 240) * 2;
    else
        y_start -= (312 - 288) * 2;

    unsigned int tx = x_start;
    int ishif;
    if (x_start >= (twidth / 2))
        ishif = (int)x_start - (int)twidth;
    else
        ishif = (int)x_start;

    int shif = 0;
    unsigned int shy = 0;

    int y = y_start;
    while (y < height) {
        if (y >= 0) {
            int* Y = fY + (y * width);

            if (shif != 0) {
                int tmp[800];
                for (int xx = 0; xx < width; xx++) tmp[xx] = Y[xx];
                for (int xx = width; xx < (int)twidth; xx++) tmp[xx] = 0;

                unsigned int x2 = (tx + twidth + (unsigned int)shif) % twidth;
                for (int xx = (int)tx; xx < width; xx++) {
                    Y[xx] = (x2 < (unsigned int)width) ? tmp[x2] : 0;
                    if (++x2 >= twidth) x2 = 0;
                }
            }
        }

        if (shy == 0)
            shif = ishif;
        else
            shif = (shif * 7) / 8;

        tx = 0;
        y += 2;
        shy++;
    }
}

// ─── Kernel 7+8+9 MERGED: Post-composite scanline processing ───────────────
// One thread per scanline. Performs (in order, if enabled):
//   - Chroma from luma (subcarrier decode)  [box blur + demodulation]
//   - Chroma noise
//   - Chroma phase noise

__global__ void kernel_scanline_post_composite(
    int* __restrict__ fY, int* __restrict__ fI, int* __restrict__ fQ,
    int width, int height, unsigned int field, unsigned long long fieldno,
    // subcarrier decode
    int subcarrier_amplitude_back,
    int video_scanline_phase_shift, int video_scanline_phase_shift_offset,
    bool do_decode,
    // chroma noise
    int video_chroma_noise, bool do_chroma_noise,
    // phase noise
    int video_chroma_phase_noise, bool do_phase_noise,
    // cuRAND
    curandState* __restrict__ rand_states)
{
    int scanline_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = field + scanline_idx * 2;
    if (y >= height) return;

    int* Y = fY + (y * width);
    int* I = fI + (y * width);
    int* Q = fQ + (y * width);

    // ── Stage A: Chroma from luma (subcarrier decode) ──
    if (do_decode) {
        int chroma[800];

        // 4-tap box blur
        int delay[4] = {0, 0, 0, 0};
        int sum = 0;
        delay[2] = Y[0]; sum += delay[2];
        delay[3] = Y[1]; sum += delay[3];

        for (int x = 0; x < width; x++) {
            int c;
            if ((x + 2) < width) c = Y[x + 2]; else c = 0;

            sum -= delay[0];
            for (int j = 0; j < 3; j++) delay[j] = delay[j + 1];
            delay[3] = c;
            sum += delay[3];
            Y[x] = sum / 4;
            chroma[x] = c - Y[x];
        }

        // Demodulate
        unsigned int xi = device_subcarrier_xi(video_scanline_phase_shift,
                                                video_scanline_phase_shift_offset, fieldno, y);

        for (int x = ((4 - xi) & 3); (x + 3) < width; x += 4) {
            chroma[x + 2] = -chroma[x + 2];
            chroma[x + 3] = -chroma[x + 3];
        }

        for (int x = 0; x < width; x++) {
            chroma[x] = ((int)chroma[x] * 50) / subcarrier_amplitude_back;
        }

        // Decode I/Q
        int x;
        for (x = 0; (x + xi + 1) < (unsigned int)width; x += 2) {
            I[x] = -chroma[x + xi + 0];
            Q[x] = -chroma[x + xi + 1];
        }
        for (; x < width; x += 2) {
            I[x] = 0;
            Q[x] = 0;
        }
        // Interpolate odd pixels
        for (x = 0; (x + 2) < width; x += 2) {
            I[x + 1] = (I[x] + I[x + 2]) >> 1;
            Q[x + 1] = (Q[x] + Q[x + 2]) >> 1;
        }
        for (; x < width; x++) {
            I[x] = 0;
            Q[x] = 0;
        }
    }

    // ── Stage B: Chroma noise ──
    if (do_chroma_noise) {
        curandState localState = rand_states[scanline_idx];
        int noiseU = 0, noiseV = 0;

        for (int x = 0; x < width; x++) {
            I[x] += noiseU;
            Q[x] += noiseV;
            noiseU += ((int)(curand(&localState) % ((video_chroma_noise * 2) + 1))) - video_chroma_noise;
            noiseU /= 2;
            noiseV += ((int)(curand(&localState) % ((video_chroma_noise * 2) + 1))) - video_chroma_noise;
            noiseV /= 2;
        }

        rand_states[scanline_idx] = localState;
    }

    // ── Stage C: Chroma phase noise ──
    if (do_phase_noise) {
        curandState localState = rand_states[scanline_idx];

        int pnoise = ((int)(curand(&localState) % ((video_chroma_phase_noise * 2) + 1))) - video_chroma_phase_noise;
        pnoise /= 2;
        double pi = ((double)pnoise * M_PI) / 100.0;
        double sinpi = sin(pi);
        double cospi = cos(pi);

        for (int x = 0; x < width; x++) {
            double u = I[x];
            double v = Q[x];
            I[x] = (int)((u * cospi) - (v * sinpi));
            Q[x] = (int)((u * sinpi) + (v * cospi));
        }

        rand_states[scanline_idx] = localState;
    }
}

// ─── Kernel 10 MERGED: VHS luma LP + chroma LP + sharpen ───────────────────
// One thread per scanline.  Performs:
//   - VHS luma lowpass (3 cascaded IIR + preemphasis)
//   - VHS chroma lowpass (3 cascaded IIR on I and Q)
//   - VHS sharpen (3 cascaded IIR on Y)

__global__ void kernel_scanline_vhs(
    int* __restrict__ fY, int* __restrict__ fI, int* __restrict__ fQ,
    int width, int height, unsigned int field,
    double luma_cut, double chroma_cut, int chroma_delay,
    double vhs_out_sharpen)
{
    int scanline_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = field + scanline_idx * 2;
    if (y >= height) return;

    int* Y = fY + (y * width);
    int* U = fI + (y * width);
    int* V = fQ + (y * width);
    const double rate = (315000000.0 * 4) / 88;

    // ── VHS luma lowpass ──
    {
        DeviceLowpass lp[3];
        DeviceLowpass pre;
        for (int f = 0; f < 3; f++)
            lp[f].init(rate, luma_cut, 16.0);
        pre.init(rate, luma_cut, 16.0);

        for (int x = 0; x < width; x++) {
            double s = Y[x];
            for (int f = 0; f < 3; f++) s = lp[f].lowpass(s);
            s += pre.highpass(s) * 1.6;
            Y[x] = (int)s;
        }
    }

    // ── VHS chroma lowpass ──
    {
        DeviceLowpass lpU[3], lpV[3];
        for (int f = 0; f < 3; f++) {
            lpU[f].init(rate, chroma_cut, 0.0);
            lpV[f].init(rate, chroma_cut, 0.0);
        }

        for (int x = 0; x < width; x++) {
            double s;

            s = U[x];
            for (int f = 0; f < 3; f++) s = lpU[f].lowpass(s);
            if (x >= chroma_delay) U[x - chroma_delay] = (int)s;

            s = V[x];
            for (int f = 0; f < 3; f++) s = lpV[f].lowpass(s);
            if (x >= chroma_delay) V[x - chroma_delay] = (int)s;
        }
    }

    // ── VHS sharpen ──
    {
        DeviceLowpass lp[3];
        for (int f = 0; f < 3; f++)
            lp[f].init(rate, luma_cut * 4, 0.0);

        for (int x = 0; x < width; x++) {
            double s = Y[x];
            double ts = s;
            for (int f = 0; f < 3; f++) ts = lp[f].lowpass(ts);
            Y[x] = (int)(s + ((s - ts) * vhs_out_sharpen * 2));
        }
    }
}

// VHS chroma vertical blend — parallelized by column (x)
// Each thread handles one column, iterating through scanlines sequentially.
// This is much faster than a single thread because columns are independent.
__global__ void kernel_vhs_chroma_vert_blend(
    int* __restrict__ fI, int* __restrict__ fQ,
    int width, int height, unsigned int field)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int prev_U = 0;
    int prev_V = 0;

    for (int y = (int)(field + 2); y < height; y += 2) {
        int idx = y * width + x;
        int cU = fI[idx];
        int cV = fQ[idx];
        fI[idx] = (prev_U + cU + 1) >> 1;
        fQ[idx] = (prev_V + cV + 1) >> 1;
        prev_U = cU;
        prev_V = cV;
    }
}

// ─── VHS re-encode + re-decode MERGED ──────────────────────────────────────
// One thread per scanline. Performs subcarrier encode then decode.

__global__ void kernel_scanline_vhs_reencode(
    int* __restrict__ fY, int* __restrict__ fI, int* __restrict__ fQ,
    int width, int height, unsigned int field, unsigned long long fieldno,
    int subcarrier_amplitude, int subcarrier_amplitude_back,
    int video_scanline_phase_shift, int video_scanline_phase_shift_offset)
{
    int scanline_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = field + scanline_idx * 2;
    if (y >= height) return;

    int* Y = fY + (y * width);
    int* I = fI + (y * width);
    int* Q = fQ + (y * width);

    unsigned int xi = device_subcarrier_xi(video_scanline_phase_shift,
                                            video_scanline_phase_shift_offset, fieldno, y);

    // ── Subcarrier encode ──
    for (int x = 0; x < width; x++) {
        unsigned int sxi = xi + x;
        int chroma = I[x] * subcarrier_amplitude * d_Umult[sxi & 3];
        chroma += Q[x] * subcarrier_amplitude * d_Vmult[sxi & 3];
        Y[x] += chroma / 50;
        I[x] = 0;
        Q[x] = 0;
    }

    // ── Subcarrier decode ──
    int chroma[800];

    // 4-tap box blur
    int delay[4] = {0, 0, 0, 0};
    int sum = 0;
    delay[2] = Y[0]; sum += delay[2];
    delay[3] = Y[1]; sum += delay[3];

    for (int x = 0; x < width; x++) {
        int c;
        if ((x + 2) < width) c = Y[x + 2]; else c = 0;

        sum -= delay[0];
        for (int j = 0; j < 3; j++) delay[j] = delay[j + 1];
        delay[3] = c;
        sum += delay[3];
        Y[x] = sum / 4;
        chroma[x] = c - Y[x];
    }

    // Demodulate
    for (int x = ((4 - xi) & 3); (x + 3) < width; x += 4) {
        chroma[x + 2] = -chroma[x + 2];
        chroma[x + 3] = -chroma[x + 3];
    }

    for (int x = 0; x < width; x++) {
        chroma[x] = ((int)chroma[x] * 50) / subcarrier_amplitude_back;
    }

    // Decode I/Q
    int x;
    for (x = 0; (x + xi + 1) < (unsigned int)width; x += 2) {
        I[x] = -chroma[x + xi + 0];
        Q[x] = -chroma[x + xi + 1];
    }
    for (; x < width; x += 2) {
        I[x] = 0;
        Q[x] = 0;
    }
    for (x = 0; (x + 2) < width; x += 2) {
        I[x + 1] = (I[x] + I[x + 2]) >> 1;
        Q[x + 1] = (Q[x] + Q[x + 2]) >> 1;
    }
    for (; x < width; x++) {
        I[x] = 0;
        Q[x] = 0;
    }
}

// ─── Kernel 11: Chroma dropout ──────────────────────────────────────────────
// One thread per scanline.

__global__ void kernel_chroma_dropout(
    int* __restrict__ fI, int* __restrict__ fQ,
    int width, int height, unsigned int field,
    int video_chroma_loss,
    curandState* __restrict__ rand_states)
{
    int scanline_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = field + scanline_idx * 2;
    if (y >= height) return;

    curandState localState = rand_states[scanline_idx];
    if ((curand(&localState) % 100000) < (unsigned int)video_chroma_loss) {
        int* U = fI + (y * width);
        int* V = fQ + (y * width);
        for (int x = 0; x < width; x++) { U[x] = 0; V[x] = 0; }
    }
    rand_states[scanline_idx] = localState;
}

// ─── Kernel 12: Output chroma lowpass ───────────────────────────────────────
// One thread per scanline. 3 cascaded IIR.

__global__ void kernel_chroma_lowpass_out(
    int* __restrict__ fI, int* __restrict__ fQ,
    int width, int height, unsigned int field,
    int mode)  // 0 = full bandwidth limits, 1 = lite (TV-style)
{
    int scanline_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = field + scanline_idx * 2;
    if (y >= height) return;

    const double rate = (315000000.0 * 4) / 88;

    for (int p = 0; p < 2; p++) {
        int* P = (p == 0 ? fI : fQ) + (width * y);
        double cutoff;
        int delay;

        if (mode == 1) {
            cutoff = 2600000.0;
            delay = 1;
        } else {
            cutoff = (p == 0) ? 1300000.0 : 600000.0;
            delay = (p == 0) ? 2 : 4;
        }

        DeviceLowpass lp[3];
        for (int f = 0; f < 3; f++)
            lp[f].init(rate, cutoff, 0.0);

        for (int x = 0; x < width; x++) {
            double s = P[x];
            for (int f = 0; f < 3; f++) s = lp[f].lowpass(s);
            if (x >= delay) P[x - delay] = (int)s;
        }
    }
}

// ─── Kernel 13: YIQ to RGB ──────────────────────────────────────────────────
// One thread per pixel. Writes to d_dst_bgra.

__global__ void kernel_yiq_to_rgb(
    const int* __restrict__ fY, const int* __restrict__ fI, const int* __restrict__ fQ,
    uint8_t* __restrict__ dst_bgra, int dst_stride,
    int width, int height, unsigned int field)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int scanline_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int y = field + scanline_idx * 2;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int r, g, b;
    device_YIQ_to_RGB(r, g, b, fY[idx], fI[idx], fQ[idx]);

    uint32_t* dst_row = (uint32_t*)(dst_bgra + dst_stride * y);
    dst_row[x] = ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}

// ─── Kernel: Initialize cuRAND states ───────────────────────────────────────

__global__ void kernel_init_rand(curandState* states, int count, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// ═════════════════════════════════════════════════════════════════════════════
// Host API
// ═════════════════════════════════════════════════════════════════════════════

bool ntsc_cuda_init(int width, int height) {
    gpu_max_scanlines = (height + 1) / 2;  // max scanlines per field
    gpu_frame_bgra_size = (size_t)width * height * 4;

    size_t yiq_size = (size_t)width * height * sizeof(int);

    // Allocate double-buffered resources
    for (int i = 0; i < NTSC_CUDA_NUM_BUFFERS; i++) {
        BufferSet& buf = buffers[i];
        
        // Create stream for this buffer
        CUDA_CHECK(cudaStreamCreate(&buf.stream));
        
        // Device buffers
        CUDA_CHECK(cudaMalloc(&buf.d_fY, yiq_size));
        CUDA_CHECK(cudaMalloc(&buf.d_fI, yiq_size));
        CUDA_CHECK(cudaMalloc(&buf.d_fQ, yiq_size));
        CUDA_CHECK(cudaMalloc(&buf.d_src_bgra, gpu_frame_bgra_size));
        CUDA_CHECK(cudaMalloc(&buf.d_dst_bgra, gpu_frame_bgra_size));
        
        // Zero YIQ buffers
        CUDA_CHECK(cudaMemset(buf.d_fY, 0, yiq_size));
        CUDA_CHECK(cudaMemset(buf.d_fI, 0, yiq_size));
        CUDA_CHECK(cudaMemset(buf.d_fQ, 0, yiq_size));
        
        if (buf.d_fY == nullptr || buf.d_fI == nullptr || buf.d_fQ == nullptr ||
            buf.d_src_bgra == nullptr || buf.d_dst_bgra == nullptr) {
            fprintf(stderr, "ntsc_cuda_init: buffer %d allocation failed\n", i);
            return false;
        }
    }

    // Shared cuRAND states (one set for all buffers)
    CUDA_CHECK(cudaMalloc(&d_rand_states, gpu_max_scanlines * sizeof(curandState)));

    // Initialize cuRAND states
    int threads = 256;
    int blocks = (gpu_max_scanlines + threads - 1) / threads;
    kernel_init_rand<<<blocks, threads>>>(d_rand_states, gpu_max_scanlines, 42ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (d_rand_states == nullptr) {
        fprintf(stderr, "ntsc_cuda_init: rand states allocation failed\n");
        return false;
    }

    fprintf(stderr, "NTSC CUDA initialized: %dx%d, %d buffers, double-buffering enabled\n",
            width, height, NTSC_CUDA_NUM_BUFFERS);
    return true;
}

void ntsc_cuda_shutdown() {
    // Ensure any pending work completes on all streams
    for (int i = 0; i < NTSC_CUDA_NUM_BUFFERS; i++) {
        BufferSet& buf = buffers[i];
        if (buf.stream) {
            cudaStreamSynchronize(buf.stream);
            cudaStreamDestroy(buf.stream);
            buf.stream = nullptr;
        }
        if (buf.d_fY) { cudaFree(buf.d_fY); buf.d_fY = nullptr; }
        if (buf.d_fI) { cudaFree(buf.d_fI); buf.d_fI = nullptr; }
        if (buf.d_fQ) { cudaFree(buf.d_fQ); buf.d_fQ = nullptr; }
        if (buf.d_src_bgra) { cudaFree(buf.d_src_bgra); buf.d_src_bgra = nullptr; }
        if (buf.d_dst_bgra) { cudaFree(buf.d_dst_bgra); buf.d_dst_bgra = nullptr; }
    }
    if (d_rand_states) { cudaFree(d_rand_states); d_rand_states = nullptr; }
    fprintf(stderr, "NTSC CUDA shutdown\n");
}

// ─── Internal: process frame on a specific buffer ──────────────────────────
// This is the core GPU processing, shared by both legacy and async APIs.

static void process_frame_on_buffer(
    int buf_idx,
    const uint8_t* src_bgra, int src_stride,
    uint8_t* dst_bgra, int dst_stride,
    const NtscCudaParams& p,
    bool async_download)
{
    BufferSet& buf = buffers[buf_idx];
    cudaStream_t stream = buf.stream;
    
    const int width = p.width;
    const int height = p.height;
    const unsigned int field = p.field;
    const unsigned long long fieldno = p.fieldno;

    // Number of active scanlines in this field
    int num_scanlines = 0;
    for (int y = (int)field; y < height; y += 2) num_scanlines++;

    // ── Upload source frame (synchronous to stream) ──
    CUDA_CHECK(cudaMemcpy2DAsync(buf.d_src_bgra, width * 4,
                                  src_bgra, src_stride,
                                  width * 4, height,
                                  cudaMemcpyHostToDevice, stream));

    // Common launch configs
    const int scanline_threads = 256;
    const int scanline_blocks = (num_scanlines + scanline_threads - 1) / scanline_threads;

    // ── Kernel 1: RGB → YIQ ──
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (num_scanlines + block.y - 1) / block.y);
        kernel_rgb_to_yiq<<<grid, block, 0, stream>>>(
            buf.d_src_bgra, width * 4,
            buf.d_fY, buf.d_fI, buf.d_fQ,
            width, height, field, p.opposite);
    }

    // ── Kernel 2+3+4+5 MERGED: pre-composite scanline processing ──
    {
        bool do_preemphasis = (p.composite_preemphasis != 0 && p.composite_preemphasis_cut > 0);
        bool do_noise = (p.video_noise != 0);

        kernel_scanline_pre_composite<<<scanline_blocks, scanline_threads, 0, stream>>>(
            buf.d_fY, buf.d_fI, buf.d_fQ,
            width, height, field, fieldno,
            p.subcarrier_amplitude,
            p.video_scanline_phase_shift, p.video_scanline_phase_shift_offset,
            p.video_noise,
            p.composite_preemphasis, p.composite_preemphasis_cut,
            p.composite_in_chroma_lowpass, do_preemphasis, do_noise,
            d_rand_states);
    }

    // ── Kernel 6: VHS head switching ──
    if (p.vhs_head_switching) {
        kernel_vhs_head_switching<<<1, 1, 0, stream>>>(
            buf.d_fY, width, height, field,
            p.output_ntsc,
            p.vhs_head_switching_point,
            p.vhs_head_switching_phase,
            p.vhs_head_switching_phase_noise,
            d_rand_states);
    }

    // ── Kernel 7+8+9 MERGED: post-composite scanline processing ──
    {
        bool do_decode = !p.nocolor_subcarrier;
        bool do_chroma_noise = (p.video_chroma_noise != 0);
        bool do_phase_noise = (p.video_chroma_phase_noise != 0);

        kernel_scanline_post_composite<<<scanline_blocks, scanline_threads, 0, stream>>>(
            buf.d_fY, buf.d_fI, buf.d_fQ,
            width, height, field, fieldno,
            p.subcarrier_amplitude_back,
            p.video_scanline_phase_shift, p.video_scanline_phase_shift_offset,
            do_decode,
            p.video_chroma_noise, do_chroma_noise,
            p.video_chroma_phase_noise, do_phase_noise,
            d_rand_states);
    }

    // ── Kernel 10: VHS processing ──
    if (p.emulating_vhs) {
        double luma_cut, chroma_cut;
        int chroma_delay;

        switch (p.output_vhs_tape_speed) {
            case 0: luma_cut = 2400000; chroma_cut = 320000; chroma_delay = 9; break;
            case 1: luma_cut = 1900000; chroma_cut = 300000; chroma_delay = 12; break;
            case 2: luma_cut = 1400000; chroma_cut = 280000; chroma_delay = 14; break;
            default: luma_cut = 2400000; chroma_cut = 320000; chroma_delay = 9; break;
        }

        // VHS luma LP + chroma LP + sharpen (merged into single kernel)
        kernel_scanline_vhs<<<scanline_blocks, scanline_threads, 0, stream>>>(
            buf.d_fY, buf.d_fI, buf.d_fQ, width, height, field,
            luma_cut, chroma_cut, chroma_delay, p.vhs_out_sharpen);

        // VHS chroma vertical blend (NTSC only) — parallelized by column
        if (p.vhs_chroma_vert_blend && p.output_ntsc) {
            int col_threads = 256;
            int col_blocks = (width + col_threads - 1) / col_threads;
            kernel_vhs_chroma_vert_blend<<<col_blocks, col_threads, 0, stream>>>(
                buf.d_fI, buf.d_fQ, width, height, field);
        }

        // VHS re-encode/decode if not S-Video out (merged)
        if (!p.vhs_svideo_out) {
            kernel_scanline_vhs_reencode<<<scanline_blocks, scanline_threads, 0, stream>>>(
                buf.d_fY, buf.d_fI, buf.d_fQ,
                width, height, field, fieldno,
                p.subcarrier_amplitude, p.subcarrier_amplitude_back,
                p.video_scanline_phase_shift, p.video_scanline_phase_shift_offset);
        }
    }

    // ── Kernel 11: Chroma dropout ──
    if (p.video_chroma_loss != 0) {
        kernel_chroma_dropout<<<scanline_blocks, scanline_threads, 0, stream>>>(
            buf.d_fI, buf.d_fQ, width, height, field,
            p.video_chroma_loss, d_rand_states);
    }

    // ── Kernel 12: Output chroma lowpass ──
    if (p.composite_out_chroma_lowpass) {
        int mode = p.composite_out_chroma_lowpass_lite ? 1 : 0;
        kernel_chroma_lowpass_out<<<scanline_blocks, scanline_threads, 0, stream>>>(
            buf.d_fI, buf.d_fQ, width, height, field, mode);
    }

    // ── Kernel 13: YIQ → RGB ──
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (num_scanlines + block.y - 1) / block.y);
        kernel_yiq_to_rgb<<<grid, block, 0, stream>>>(
            buf.d_fY, buf.d_fI, buf.d_fQ,
            buf.d_dst_bgra, width * 4,
            width, height, field);
    }

    // ── Download result BGRA ──
    if (async_download) {
        // Async download: store host destination for later retrieval
        buf.host_dst = dst_bgra;
        buf.host_stride = dst_stride;
        CUDA_CHECK(cudaMemcpy2DAsync(dst_bgra, dst_stride,
                                      buf.d_dst_bgra, width * 4,
                                      width * 4, height,
                                      cudaMemcpyDeviceToHost, stream));
    } else {
        // Synchronous download
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy2D(dst_bgra, dst_stride,
                                buf.d_dst_bgra, width * 4,
                                width * 4, height,
                                cudaMemcpyDeviceToHost));
    }
}

// ─── Legacy synchronous API ─────────────────────────────────────────────────

void ntsc_cuda_composite_layer(
    const uint8_t* src_bgra, int src_stride,
    uint8_t* dst_bgra, int dst_stride,
    const NtscCudaParams& p)
{
    // Use buffer 0, synchronous mode
    process_frame_on_buffer(0, src_bgra, src_stride, dst_bgra, dst_stride, p, false);
}

void ntsc_cuda_sync() {
    // Legacy API uses buffer 0
    CUDA_CHECK(cudaStreamSynchronize(buffers[0].stream));
}

// ─── Double-buffered async API ──────────────────────────────────────────────

void ntsc_cuda_submit_async(
    int buf_idx,
    const uint8_t* src_bgra, int src_stride,
    uint8_t* dst_bgra, int dst_stride,
    const NtscCudaParams& params)
{
    if (buf_idx < 0 || buf_idx >= NTSC_CUDA_NUM_BUFFERS) {
        fprintf(stderr, "ntsc_cuda_submit_async: invalid buf_idx %d\n", buf_idx);
        return;
    }
    // Submit with async download
    process_frame_on_buffer(buf_idx, src_bgra, src_stride, dst_bgra, dst_stride, params, true);
}

void ntsc_cuda_wait_async(int buf_idx) {
    if (buf_idx < 0 || buf_idx >= NTSC_CUDA_NUM_BUFFERS) {
        fprintf(stderr, "ntsc_cuda_wait_async: invalid buf_idx %d\n", buf_idx);
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(buffers[buf_idx].stream));
}

bool ntsc_cuda_query_async(int buf_idx) {
    if (buf_idx < 0 || buf_idx >= NTSC_CUDA_NUM_BUFFERS) {
        return false;
    }
    cudaError_t result = cudaStreamQuery(buffers[buf_idx].stream);
    return (result == cudaSuccess);
}
