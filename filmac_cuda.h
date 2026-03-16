#ifndef FILMAC_CUDA_H
#define FILMAC_CUDA_H

// Initialize CUDA resources for filmac. Call once at startup.
// Returns true on success, false if no usable GPU found.
bool filmac_cuda_init();

// Free CUDA resources. Call once at shutdown.
void filmac_cuda_shutdown();

// GPU-accelerated filmac auto-contrast processing.
//
// Performs all 3 GPU stages in one call:
//   1. Decode input BGRA -> internal longframe (gamma or linear)
//   2. Block min/max reduction to find frame brightness range
//   3. Contrast stretch + encode back to BGRA output
//
// The temporal smoothing of min/max (final_minv/final_maxv EMA) is done
// inside this function using the passed-in pointers, which are updated
// in-place for the next frame.
//
// input:          host pointer to BGRA input frame
// output:         host pointer to BGRA output buffer
// width, height:  frame dimensions in pixels
// linesize_in:    input frame row stride in bytes
// linesize_out:   output frame row stride in bytes
// use_gamma:      true if gamma correction is enabled
// gamma_dec_lut:  256-entry decode LUT (only used if use_gamma)
// gamma_enc_lut:  8193-entry encode LUT (only used if use_gamma)
// final_minv:     pointer to temporal EMA min (read + updated in place)
// final_maxv:     pointer to temporal EMA max (read + updated in place)
// final_init:     pointer to init flag (read + updated in place)
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
    bool* final_init);

#endif // FILMAC_CUDA_H
