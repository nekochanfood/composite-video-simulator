// common.h — Shared includes, macros, and utility functions
// Used by all 4 programs: ffmpeg_ntsc, ffmpeg_vhsled, frameblend, filmac
#ifndef COMMON_H
#define COMMON_H

#define __STDC_CONSTANT_MACROS
#define __STDC_LIMIT_MACROS

#include <sys/types.h>
#include <signal.h>
#include <stdint.h>
#include <assert.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <math.h>

extern "C" {
#include <libavutil/opt.h>
#include <libavutil/avutil.h>
#include <libavutil/pixfmt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/samplefmt.h>
#include <libavutil/pixelutils.h>

#include <libavcodec/avcodec.h>
#include <libavcodec/version.h>

#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavformat/version.h>

#include <libswscale/swscale.h>
#include <libswscale/version.h>

#include <libswresample/swresample.h>
#include <libswresample/version.h>
}

using namespace std;

#include <map>
#include <string>
#include <vector>
#include <stdexcept>

#define RGBTRIPLET(r,g,b) (((uint32_t)(r) << (uint32_t)16) + ((uint32_t)(g) << (uint32_t)8) + ((uint32_t)(b) << (uint32_t)0))

// Signal handler for graceful shutdown
static volatile int DIE = 0;

static void sigma(int x) {
	if (++DIE >= 20) abort();
}

// dBFS utility functions (used by ffmpeg_ntsc audio processing)
static inline double dBFS(double dB) {
	return pow(10.0, dB / 20.0);
}

static inline double attenuate_dBFS(double sample, double dB) {
	return sample * dBFS(dB);
}

static inline double dBFS_measure(double sample) {
	return 20.0 * log10(sample);
}

#endif // COMMON_H
