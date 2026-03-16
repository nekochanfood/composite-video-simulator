#ifndef FRAMEBLEND_CUDA_H
#define FRAMEBLEND_CUDA_H

#include <cstddef>

// Initialize CUDA resources. Call once at startup.
// Returns true on success, false if no usable GPU found.
bool frameblend_cuda_init();

// Free CUDA resources. Call once at shutdown.
void frameblend_cuda_shutdown();

// GPU-accelerated gamma-corrected weighted frame blend.
//
// frame_ptrs:     array of host pointers to input frame BGRA data (linesize_in bytes per row)
// weights:        array of 16-bit fixed-point weights (0x10000 = 1.0)
// num_frames:     number of input frames / weights
// output:         host pointer to output BGRA buffer (linesize_out bytes per row)
// width, height:  frame dimensions in pixels
// linesize_in:    input frame row stride in bytes
// linesize_out:   output frame row stride in bytes
// gamma_dec_lut:  256-entry decode LUT (8-bit -> 13-bit gamma-linear)
// gamma_enc_lut:  8193-entry encode LUT (13-bit -> 8-bit)
void frameblend_cuda_gamma(
    const unsigned char* const* frame_ptrs,
    const unsigned int* weights,
    int num_frames,
    unsigned char* output,
    int width, int height,
    int linesize_in, int linesize_out,
    const unsigned long* gamma_dec_lut,
    const unsigned long* gamma_enc_lut);

// GPU-accelerated linear (no gamma) weighted frame blend.
//
// frame_ptrs:     array of host pointers to input frame BGRA data (linesize_in bytes per row)
// weights:        array of 16-bit fixed-point weights (0x10000 = 1.0)
// num_frames:     number of input frames / weights
// output:         host pointer to output BGRA buffer (linesize_out bytes per row)
// width, height:  frame dimensions in pixels
// linesize_in:    input frame row stride in bytes
// linesize_out:   output frame row stride in bytes
void frameblend_cuda_linear(
    const unsigned char* const* frame_ptrs,
    const unsigned int* weights,
    int num_frames,
    unsigned char* output,
    int width, int height,
    int linesize_in, int linesize_out);

#endif // FRAMEBLEND_CUDA_H
