#ifndef VHSLED_CUDA_H
#define VHSLED_CUDA_H

// Initialize CUDA resources for vhsled. Call once at startup.
// Returns true on success, false if no usable GPU found.
bool vhsled_cuda_init();

// Free CUDA resources. Call once at shutdown.
void vhsled_cuda_shutdown();

// GPU-accelerated VHS left-edge detection and correction.
//
// Performs all 3 stages in one call:
//   1. Per-scanline left-edge detection (find first non-black column)
//   2. Vertical 9-tap box filter smoothing of edge positions
//   3. Per-scanline horizontal shift to crop black borders
//
// src:             host pointer to BGRA source frame (input_avstream_video_frame_rgb)
// dst:             host pointer to BGRA destination frame (output_avstream_video_frame)
// width, height:   frame dimensions in pixels
// src_linesize:    source frame row stride in bytes
// dst_linesize:    destination frame row stride in bytes
void vhsled_cuda_process(
    const unsigned char* src,
    unsigned char* dst,
    int width, int height,
    int src_linesize, int dst_linesize);

#endif // VHSLED_CUDA_H
