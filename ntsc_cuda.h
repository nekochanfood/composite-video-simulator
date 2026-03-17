// ntsc_cuda.h — CUDA GPU acceleration for NTSC composite video simulation
// Replaces composite_layer() processing stages with GPU kernels.
// All intermediate YIQ buffers stay on GPU between kernel launches.
//
// Performance optimizations over naive port:
//   - CUDA stream for async kernel execution
//   - Pinned host memory for async H2D/D2H transfers
//   - Merged per-scanline IIR kernels to reduce launch overhead
//   - Double-buffered frame pipeline to overlap GPU compute with CPU encode
#ifndef NTSC_CUDA_H
#define NTSC_CUDA_H

#include <cstdint>
#include <cstddef>

// Number of pipeline buffers for double-buffering
#define NTSC_CUDA_NUM_BUFFERS 2

// Parameters that control the composite simulation — passed per-frame
struct NtscCudaParams {
    int width;
    int height;
    unsigned int field;            // 0 or 1
    unsigned long long fieldno;

    // subcarrier
    int subcarrier_amplitude;
    int subcarrier_amplitude_back;
    int video_scanline_phase_shift;      // 0, 90, 180, 270
    int video_scanline_phase_shift_offset;

    // noise
    int video_noise;
    int video_chroma_noise;
    int video_chroma_phase_noise;
    int video_chroma_loss;

    // composite preemphasis
    double composite_preemphasis;
    double composite_preemphasis_cut;

    // flags
    bool composite_in_chroma_lowpass;
    bool composite_out_chroma_lowpass;
    bool composite_out_chroma_lowpass_lite;
    bool nocolor_subcarrier;
    bool nocolor_subcarrier_after_yc_sep;
    bool enable_composite_emulation;

    // VHS
    bool emulating_vhs;
    int output_vhs_tape_speed;       // 0=SP, 1=LP, 2=EP
    bool vhs_chroma_vert_blend;
    bool vhs_svideo_out;
    double vhs_out_sharpen;
    bool output_ntsc;

    // VHS head switching
    bool vhs_head_switching;
    double vhs_head_switching_point;
    double vhs_head_switching_phase;
    double vhs_head_switching_phase_noise;

    // interlaced source info
    unsigned char opposite;        // field offset for interlaced source

    // progressive mode: when true, kernels process all scanlines (y = scanline_idx).
    // when false, kernels process interlaced field lines (y = field + scanline_idx * 2).
    // spout_ntsc uses progressive=true; ffmpeg_ntsc uses progressive=false.
    bool progressive;
};

// Initialize CUDA resources (device buffers for YIQ, cuRAND states, etc.)
// Call once at startup after output_width/output_height are known.
// priority: 0=low, 1=normal, 2=high (default). Controls CUDA stream priority.
// Returns true on success.
bool ntsc_cuda_init(int width, int height, int priority = 2);

// Free all CUDA resources. Call once at shutdown.
void ntsc_cuda_shutdown();

// ─── Legacy synchronous API (still supported) ───────────────────────────────

// GPU replacement for composite_layer().
// src_bgra: host pointer to source BGRA frame (srcframe->data[0]), stride = src_stride bytes
// dst_bgra: host pointer to destination BGRA frame (dstframe->data[0]), stride = dst_stride bytes
// params: all simulation parameters for this field
//
// Uses async CUDA stream internally. The function returns after launching
// all GPU work but before it completes — call ntsc_cuda_sync() to ensure
// the output buffer is valid before reading it.
void ntsc_cuda_composite_layer(
    const uint8_t* src_bgra, int src_stride,
    uint8_t* dst_bgra, int dst_stride,
    const NtscCudaParams& params);

// Block until all GPU work (including D2H copy) from the most recent
// ntsc_cuda_composite_layer() call has finished. After this returns,
// the dst_bgra buffer passed to the last call is valid to read.
void ntsc_cuda_sync();

// ─── Double-buffered async API ──────────────────────────────────────────────
// Allows overlapping GPU compute for frame N with CPU encoding of frame N-1.
//
// Usage pattern:
//   for each frame:
//     ntsc_cuda_submit_async(buf_idx, src, dst, params);  // start GPU work
//     if (have_previous_frame) {
//         ntsc_cuda_wait_async(prev_buf_idx);              // wait for prev frame
//         // encode prev frame while GPU processes current
//     }
//     buf_idx = (buf_idx + 1) % NTSC_CUDA_NUM_BUFFERS;
//   // wait & encode final frame

// Submit frame for async GPU processing on buffer slot buf_idx (0 or 1).
// Returns immediately after launching GPU work. The result will be written
// to dst_bgra when ntsc_cuda_wait_async(buf_idx) completes.
void ntsc_cuda_submit_async(
    int buf_idx,
    const uint8_t* src_bgra, int src_stride,
    uint8_t* dst_bgra, int dst_stride,
    const NtscCudaParams& params);

// Wait for async GPU work on buffer slot buf_idx to complete.
// After this returns, the dst_bgra passed to submit_async is valid.
void ntsc_cuda_wait_async(int buf_idx);

// Check if async work on buffer slot is complete (non-blocking).
// Returns true if work is done, false if still in progress.
bool ntsc_cuda_query_async(int buf_idx);

#endif // NTSC_CUDA_H
