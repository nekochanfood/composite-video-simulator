// color_convert.h — RGB/YIQ color space conversion utilities
// Used by ffmpeg_ntsc for NTSC composite signal emulation.
// Will be shared with CUDA kernels in later phases.
#ifndef COLOR_CONVERT_H
#define COLOR_CONVERT_H

static inline void RGB_to_YIQ(int &Y, int &I, int &Q, int r, int g, int b) {
	double dY;

	dY = (0.30 * r) + (0.59 * g) + (0.11 * b);

	Y = (int)(256 * dY);
	I = (int)(256 * ((-0.27 * (b - dY)) + ( 0.74 * (r - dY))));
	Q = (int)(256 * (( 0.41 * (b - dY)) + ( 0.48 * (r - dY))));
}

static inline void YIQ_to_RGB(int &r, int &g, int &b, int Y, int I, int Q) {
	// FIXME
	r = (int)((( 1.000 * Y) + ( 0.956 * I) + ( 0.621 * Q)) / 256);
	g = (int)((( 1.000 * Y) + (-0.272 * I) + (-0.647 * Q)) / 256);
	b = (int)((( 1.000 * Y) + (-1.106 * I) + ( 1.703 * Q)) / 256);
	if (r < 0) r = 0;
	else if (r > 255) r = 255;
	if (g < 0) g = 0;
	else if (g > 255) g = 255;
	if (b < 0) b = 0;
	else if (b > 255) b = 255;
}

#endif // COLOR_CONVERT_H
