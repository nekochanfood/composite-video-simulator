// input_file.h — Video input file handling class
// Shared base class for reading and decoding video input files via FFmpeg.
// Used by all 4 programs. ffmpeg_ntsc.cpp extends this with audio support.
//
// Requires these globals to be defined in the including .cpp:
//   AVRational output_field_rate;
//   int output_width, output_height;
//   int underscan;
#ifndef INPUT_FILE_H
#define INPUT_FILE_H

#include "common.h"

extern AVRational output_field_rate;
extern int output_width;
extern int output_height;
extern int underscan;

class InputFile {
public:
	InputFile() {
		input_avfmt = NULL;
		input_avstream_video = NULL;
		input_avstream_video_frame = NULL;
		input_avstream_video_frame_rgb = NULL;
		input_avstream_video_resampler = NULL;
		input_avstream_video_codec_context = NULL;
		next_pts = next_dts = -1LL;
		avpkt = NULL;
		avpkt_valid = false;
		eof_stream = false;
		eof = false;
		got_video = false;
		adj_time = 0;
		t = pt = -1;
		input_avstream_video_resampler_format = AV_PIX_FMT_NONE;
		input_avstream_video_resampler_height = -1;
		input_avstream_video_resampler_width = -1;
		input_avstream_video_resampler_x = 0;
		input_avstream_video_resampler_y = 0;
	}
	virtual ~InputFile() {
		close_input();
	}
public:
	double video_frame_to_output_f(void) {
		if (input_avstream_video_frame != NULL) {
			if (input_avstream_video_frame->pts != AV_NOPTS_VALUE) {
				double n = input_avstream_video_frame->pts;

				n *= (signed long long)input_avstream_video->time_base.num * (signed long long)output_field_rate.num;
				n /= (signed long long)input_avstream_video->time_base.den * (signed long long)output_field_rate.den;

				return n;
			}
		}

		return AV_NOPTS_VALUE;
	}
	double video_frame_rgb_to_output_f(void) {
		if (input_avstream_video_frame_rgb != NULL) {
			if (input_avstream_video_frame_rgb->pts != AV_NOPTS_VALUE) {
				double n = input_avstream_video_frame_rgb->pts;

				n *= (signed long long)input_avstream_video->time_base.num * (signed long long)output_field_rate.num;
				n /= (signed long long)input_avstream_video->time_base.den * (signed long long)output_field_rate.den;

				return n;
			}
		}

		return AV_NOPTS_VALUE;
	}
	void reset_on_dup(void) {
		path.clear();
	}
	virtual bool open_input(void) {
		if (input_avfmt == NULL) {
			if (avformat_open_input(&input_avfmt,path.c_str(),NULL,NULL) < 0) {
				fprintf(stderr,"Failed to open input file\n");
				close_input();
				return false;
			}

			if (avformat_find_stream_info(input_avfmt,NULL) < 0)
				fprintf(stderr,"WARNING: Did not find stream info on input\n");

			/* scan streams for video */
			{
				size_t i;
				AVStream *is;
				int vc=0;
				AVCodecParameters *ispar;

				fprintf(stderr,"Input format: %u streams found\n",input_avfmt->nb_streams);
				for (i=0;i < (size_t)input_avfmt->nb_streams;i++) {
					is = input_avfmt->streams[i];
					if (is == NULL) continue;

					ispar = is->codecpar;
					if (ispar == NULL) continue;

					if (ispar->codec_type == AVMEDIA_TYPE_VIDEO) {
						if (input_avstream_video == NULL && vc == 0) {
							if ((input_avstream_video_codec_context=avcodec_alloc_context3(avcodec_find_decoder(ispar->codec_id))) != NULL) {
								if (avcodec_parameters_to_context(input_avstream_video_codec_context,ispar) < 0)
									fprintf(stderr,"WARNING: parameters to context failed\n");

								if (avcodec_open2(input_avstream_video_codec_context,avcodec_find_decoder(ispar->codec_id),NULL) >= 0) {
									input_avstream_video = is;
									fprintf(stderr,"Found video stream idx=%zu\n",i);
								}
								else {
									fprintf(stderr,"Found video stream but not able to decode\n");
									avcodec_free_context(&input_avstream_video_codec_context);
								}
							}
						}

						vc++;
					}
				}

				if (input_avstream_video == NULL) {
					fprintf(stderr,"Video not found\n");
					close_input();
					return false;
				}
			}
		}

		/* prepare video decoding */
		if (input_avstream_video != NULL) {
			input_avstream_video_frame = av_frame_alloc();
			if (input_avstream_video_frame == NULL) {
				fprintf(stderr,"Failed to alloc video frame\n");
				close_input();
				return false;
			}
		}

		input_avstream_video_resampler_format = AV_PIX_FMT_NONE;
		input_avstream_video_resampler_height = -1;
		input_avstream_video_resampler_width = -1;
		eof_stream = false;
		got_video = false;
		adj_time = 0;
		t = pt = -1;
		eof = false;
		avpkt_init();
		next_pts = next_dts = -1LL;
		return (input_avfmt != NULL);
	}

	uint32_t *copy_rgba(const AVFrame * const src) {
		assert(src != NULL);
		assert(src->data[0] != NULL);
		assert(src->linesize[0] != 0);
		assert(src->height != 0);

		assert(src->linesize[0] >= (src->width * 4));

		uint32_t *r = (uint32_t*)(new uint8_t[src->linesize[0] * src->height]);
		memcpy(r,src->data[0],src->linesize[0] * src->height);

		return r;
	}

	virtual bool next_packet(void) {
		if (eof) return false;
		if (input_avfmt == NULL) return false;

		do {
			if (eof_stream) break;
			avpkt_release();
			avpkt_init();
			if (av_read_frame(input_avfmt,avpkt) < 0) {
				eof_stream = true;
				return false;
			}
			if (avpkt->stream_index >= input_avfmt->nb_streams)
				continue;

			// ugh... this can happen if the source is an AVI file
			if (avpkt->pts == AV_NOPTS_VALUE) avpkt->pts = avpkt->dts;

			/* track time and keep things monotonic for our code */
			if (avpkt->pts != AV_NOPTS_VALUE) {
				t = avpkt->pts * av_q2d(input_avfmt->streams[avpkt->stream_index]->time_base);

				if (pt < 0)
					adj_time = -t;
				else if ((t+1.5) < pt) { // time code jumps backwards (1.5 is safe for DVD timecode resets)
					adj_time += pt - t;
					fprintf(stderr,"Time code jump backwards %.6f->%.6f. adj_time=%.6f\n",pt,t,adj_time);
				}
				else if (t > (pt+5)) { // time code jumps forwards
					adj_time += pt - t;
					fprintf(stderr,"Time code jump forwards %.6f->%.6f. adj_time=%.6f\n",pt,t,adj_time);
				}

				pt = t;
			}

			if (pt < 0)
				continue;

			if (avpkt->pts != AV_NOPTS_VALUE) {
				avpkt->pts += (adj_time * input_avfmt->streams[avpkt->stream_index]->time_base.den) /
					input_avfmt->streams[avpkt->stream_index]->time_base.num;
			}

			if (avpkt->dts != AV_NOPTS_VALUE) {
				avpkt->dts += (adj_time * input_avfmt->streams[avpkt->stream_index]->time_base.den) /
					input_avfmt->streams[avpkt->stream_index]->time_base.num;
			}

			got_video = false;
			if (input_avstream_video != NULL && avpkt->stream_index == input_avstream_video->index) {
				if (got_video) fprintf(stderr,"Video content lost\n");
				handle_frame(/*&*/(*avpkt)); // will set got_video
				break;
			}

			avpkt_release();
		} while (1);

		if (eof_stream) {
			avpkt_release();
			handle_frame(); // will set got_video
			if (!got_video) eof = true;
			else fprintf(stderr,"Got latent frame\n");
		}

		return true;
	}
	void frame_copy_scale(void) {
		if (input_avstream_video_frame_rgb == NULL) {
			fprintf(stderr,"New input frame\n");
			input_avstream_video_frame_rgb = av_frame_alloc();
			if (input_avstream_video_frame_rgb == NULL) {
				fprintf(stderr,"Failed to alloc video frame\n");
				return;
			}

			input_avstream_video_frame_rgb->format = AV_PIX_FMT_BGRA;
			input_avstream_video_frame_rgb->height = output_height;
			input_avstream_video_frame_rgb->width = output_width;
			if (av_frame_get_buffer(input_avstream_video_frame_rgb,64) < 0) {
				fprintf(stderr,"Failed to alloc render frame\n");
				return;
			}
			memset(input_avstream_video_frame_rgb->data[0],0,input_avstream_video_frame_rgb->linesize[0]*input_avstream_video_frame_rgb->height);

			fprintf(stderr,"RGB is %d, %d\n",input_avstream_video_frame_rgb->width,input_avstream_video_frame_rgb->height);
		}

		if (input_avstream_video_resampler != NULL) { // pixel format change or width/height change = free resampler and reinit
			if (input_avstream_video_resampler_format != input_avstream_video_frame->format ||
					input_avstream_video_resampler_width != input_avstream_video_frame->width ||
					input_avstream_video_resampler_height != input_avstream_video_frame->height) {
				sws_freeContext(input_avstream_video_resampler);
				input_avstream_video_resampler = NULL;
			}
		}

		if (input_avstream_video_resampler == NULL) {
			int final_w,final_h;

			final_w = (input_avstream_video_frame_rgb->width * (100 - underscan)) / 100;
			final_h = (input_avstream_video_frame_rgb->height * (100 - underscan)) / 100;

			if (final_w < 1) final_w = 1;
			if (final_h < 1) final_h = 1;

			input_avstream_video_resampler = sws_getContext(
					// source
					input_avstream_video_frame->width,
					input_avstream_video_frame->height,
					(AVPixelFormat)input_avstream_video_frame->format,
					// dest
					final_w,
					final_h,
					(AVPixelFormat)input_avstream_video_frame_rgb->format,
					// opt
					SWS_BILINEAR, NULL, NULL, NULL);

			if (input_avstream_video_resampler != NULL) {
				fprintf(stderr,"sws_getContext new context\n");
				input_avstream_video_resampler_format = (AVPixelFormat)input_avstream_video_frame->format;
				input_avstream_video_resampler_width = input_avstream_video_frame->width;
				input_avstream_video_resampler_height = input_avstream_video_frame->height;
				input_avstream_video_resampler_x = (input_avstream_video_frame_rgb->width - final_w) / 2;
				input_avstream_video_resampler_y = (input_avstream_video_frame_rgb->height - final_h) / 2;
				assert(input_avstream_video_resampler_x >= 0);
				assert(input_avstream_video_resampler_y >= 0);
				fprintf(stderr,"dst %d, %d\n",input_avstream_video_frame_rgb->width,input_avstream_video_frame_rgb->height);
				fprintf(stderr,"ofs %d, %d\n",input_avstream_video_resampler_x,input_avstream_video_resampler_y);
			}
			else {
				fprintf(stderr,"sws_getContext fail\n");
			}
		}

		if (input_avstream_video_resampler != NULL) {
			input_avstream_video_frame_rgb->pts = input_avstream_video_frame->pts;
			input_avstream_video_frame_rgb->flags = (input_avstream_video_frame_rgb->flags & ~(AV_FRAME_FLAG_TOP_FIELD_FIRST | AV_FRAME_FLAG_INTERLACED)) |
				(input_avstream_video_frame->flags & (AV_FRAME_FLAG_TOP_FIELD_FIRST | AV_FRAME_FLAG_INTERLACED));

			unsigned char *dst_planes[8] = {NULL};

			dst_planes[0]  = input_avstream_video_frame_rgb->data[0];
			dst_planes[0] += input_avstream_video_resampler_y * input_avstream_video_frame_rgb->linesize[0];
			dst_planes[0] += input_avstream_video_resampler_x * 4;

			if (sws_scale(input_avstream_video_resampler,
						// source
						input_avstream_video_frame->data,
						input_avstream_video_frame->linesize,
						0,input_avstream_video_frame->height,
						// dest
						dst_planes,
						input_avstream_video_frame_rgb->linesize) <= 0)
				fprintf(stderr,"WARNING: sws_scale failed\n");
		}
	}
	void handle_frame(void) {
		avcodec_send_packet(input_avstream_video_codec_context,NULL);

		if (avcodec_receive_frame(input_avstream_video_codec_context,input_avstream_video_frame) >= 0) {
			got_video = true;
		}
		else {
			got_video = false;
			fprintf(stderr,"No video decoded\n");
		}
	}
	void handle_frame(AVPacket &pkt) {
		avcodec_send_packet(input_avstream_video_codec_context,&pkt);

		if (avcodec_receive_frame(input_avstream_video_codec_context,input_avstream_video_frame) >= 0) {
			got_video = true;
		}
		else {
			got_video = false;
			fprintf(stderr,"No video decoded\n");
		}
	}
	void avpkt_init(void) {
		if (!avpkt_valid) {
			avpkt_valid = true;
			avpkt = av_packet_alloc();
		}
	}
	void avpkt_release(void) {
		if (avpkt_valid) {
			avpkt_valid = false;
			av_packet_free(&avpkt);
		}
		got_video = false;
	}
	virtual void close_input(void) {
		eof = true;
		avpkt_release();
		if (input_avstream_video_codec_context != NULL) {
			avcodec_free_context(&input_avstream_video_codec_context);
			assert(input_avstream_video_codec_context == NULL);
			input_avstream_video = NULL;
		}

		if (input_avstream_video_frame != NULL)
			av_frame_free(&input_avstream_video_frame);
		if (input_avstream_video_frame_rgb != NULL)
			av_frame_free(&input_avstream_video_frame_rgb);

		if (input_avstream_video_resampler != NULL) {
			sws_freeContext(input_avstream_video_resampler);
			input_avstream_video_resampler = NULL;
		}

		avformat_close_input(&input_avfmt);
	}
public:
	std::string             path;
	uint32_t                color;
	bool                    eof;
	bool                    eof_stream;
	bool                    got_video;
public:
	AVFormatContext*        input_avfmt;
	AVStream*               input_avstream_video;	            // do not free
	AVCodecContext*         input_avstream_video_codec_context;
	AVFrame*                input_avstream_video_frame;
	AVFrame*                input_avstream_video_frame_rgb;
	struct SwsContext*      input_avstream_video_resampler;
	AVPixelFormat           input_avstream_video_resampler_format;
	int                     input_avstream_video_resampler_height;
	int                     input_avstream_video_resampler_width;
	int                     input_avstream_video_resampler_y;
	int                     input_avstream_video_resampler_x;
	signed long long        next_pts;
	signed long long        next_dts;
	AVPacket*               avpkt;
	bool                    avpkt_valid;
	double                  adj_time;
	double                  t,pt;
};

#endif // INPUT_FILE_H
