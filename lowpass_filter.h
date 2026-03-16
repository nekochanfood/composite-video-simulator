// lowpass_filter.h — IIR lowpass/highpass filter classes for audio/signal processing
// Used by ffmpeg_ntsc for audio filtering and composite signal emulation
#ifndef LOWPASS_FILTER_H
#define LOWPASS_FILTER_H

#include <vector>
#include <cassert>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// lowpass filter
// you can make it a highpass filter by applying a lowpass then subtracting from source.
class LowpassFilter {
public:
	LowpassFilter() : timeInterval(0), cutoff(0), alpha(0), prev(0), tau(0) {
	}
	void setFilter(const double rate/*sample rate of audio*/,const double hz/*cutoff*/) {
		timeInterval = 1.0 / rate;
		tau = 1 / (hz * 2 * M_PI);
		cutoff = hz;
		alpha = timeInterval / (tau + timeInterval);
	}
	void resetFilter(const double val=0) {
		prev = val;
	}
	double lowpass(const double sample) {
		const double stage1 = sample * alpha;
		const double stage2 = prev - (prev * alpha); /* NTS: Instead of prev * (1.0 - alpha) */
		return (prev = (stage1 + stage2)); /* prev = stage1+stage2 then return prev */
	}
	double highpass(const double sample) {
		const double stage1 = sample * alpha;
		const double stage2 = prev - (prev * alpha); /* NTS: Instead of prev * (1.0 - alpha) */
		return sample - (prev = (stage1 + stage2)); /* prev = stage1+stage2 then return (sample - prev) */
	}
public:
	double			timeInterval;
	double			cutoff;
	double			alpha; /* timeInterval / (tau + timeInterval) */
	double			prev;
	double			tau;
};

class HiLoPair {
public:
	LowpassFilter		hi,lo;	// highpass, lowpass
public:
	void setFilter(const double rate/*sample rate of audio*/,const double low_hz,const double high_hz) {
		lo.setFilter(rate,low_hz);
		hi.setFilter(rate,high_hz);
	}
	double filter(const double sample) {
		return hi.highpass(lo.lowpass(sample)); /* first lowpass, then highpass */
	}
};

class HiLoPass : public std::vector<HiLoPair> { // all passes, one sample of one channel
public:
	HiLoPass() : std::vector<HiLoPair>() { }
public:
	void setFilter(const double rate/*sample rate of audio*/,const double low_hz,const double high_hz) {
		for (size_t i=0;i < size();i++) (*this)[i].setFilter(rate,low_hz,high_hz);
	}
	double filter(double sample) {
		for (size_t i=0;i < size();i++) sample = (*this)[i].lo.lowpass(sample);
		for (size_t i=0;i < size();i++) sample = (*this)[i].hi.highpass(sample);
		return sample;
	}
	void init(const unsigned int passes) {
		clear();
		resize(passes);
		assert(size() >= passes);
	}
};

class HiLoSample : public std::vector<HiLoPass> { // all passes, all channels of one sample period
public:
	HiLoSample() : std::vector<HiLoPass>() { }
public:
	void init(const unsigned int channels,const unsigned int passes) {
		clear();
		resize(channels);
		assert(size() >= channels);
		for (size_t i=0;i < size();i++) (*this)[i].init(passes);
	}
	void setFilter(const double rate/*sample rate of audio*/,const double low_hz,const double high_hz) {
		for (size_t i=0;i < size();i++) (*this)[i].setFilter(rate,low_hz,high_hz);
	}
};

class HiLoComboPass {
public:
	HiLoComboPass() : passes(0), channels(0), rate(0), low_cutoff(0), high_cutoff(0) {
	}
	~HiLoComboPass() {
		clear();
	}
	void setChannels(const size_t _channels) {
		if (channels != _channels) {
			clear();
			channels = _channels;
		}
	}
	void setCutoff(const double _low_cutoff,const double _high_cutoff) {
		if (low_cutoff != _low_cutoff || high_cutoff != _high_cutoff) {
			clear();
			low_cutoff = _low_cutoff;
			high_cutoff = _high_cutoff;
		}
	}
	void setRate(const double _rate) {
		if (rate != _rate) {
			clear();
			rate = _rate;
		}
	}
	void setPasses(const size_t _passes) {
		if (passes != _passes) {
			clear();
			passes = _passes;
		}
	}
	void clear() {
		audiostate.clear();
	}
	void init() {
		clear();
		if (channels == 0 || passes == 0 || rate == 0 || low_cutoff == 0 || high_cutoff == 0) return;
		audiostate.init(channels,passes);
		audiostate.setFilter(rate,low_cutoff,high_cutoff);
	}
public:
	double		rate;
	size_t		passes;
	size_t		channels;
	double		low_cutoff;
	double		high_cutoff;
	HiLoSample	audiostate;
};

#endif // LOWPASS_FILTER_H
