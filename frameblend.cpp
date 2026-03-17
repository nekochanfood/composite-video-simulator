
#include "common.h"
#include "input_file.h"
#include "output_context.h"

#ifdef HAVE_CUDA
#include "frameblend_cuda.h"
static bool cuda_available = false;
#endif

bool            squelch_frameblend_near_match = false;

bool            fullframealt = false;
int             framealt = 1;

double          gamma_correction = -1;

int             underscan = 0;

bool            use_422_colorspace = false;
AVRational	output_field_rate = { 60000, 1001 };	// NTSC 60Hz default
int		output_width = -1;
int		output_height = -1;
int		output_ar_n = 1,output_ar_d = 1;

AVFormatContext*                    output_avfmt = NULL;
AVStream*                           output_avstream_video = NULL;	// do not free
AVCodecContext*	                    output_avstream_video_codec_context = NULL; // do not free
AVFrame*                            output_avstream_video_frame = NULL;         // ARGB
AVFrame*                            output_avstream_video_encode_frame = NULL;  // 4:2:2 or 4:2:0
struct SwsContext*                  output_avstream_video_resampler = NULL;

std::vector<InputFile>      input_files;
std::string                 output_file;

InputFile &current_input_file(void) {
	if (input_files.empty()) {
		std::string what = "input files empty";
		throw std::out_of_range(/*&*/what);
	}

	return *(input_files.rbegin()); /* last one */
}

InputFile &new_input_file(void) {
	if (!input_files.empty()) {
		/* copy the last one, except for some fields */
		{
			InputFile &last = current_input_file();
			input_files.push_back(last);
		}
		{
			InputFile &last = current_input_file();
			last.reset_on_dup();
		}
	}
	else {
		/* make a new one with defaults */
		input_files.push_back(InputFile());
	}

	return current_input_file();
}

void preset_NTSC() {
	output_field_rate.num = 60000;
	output_field_rate.den = 1001;
}

static void help(const char *arg0) {
	fprintf(stderr,"%s [options]\n",arg0);
	fprintf(stderr," -i <input file>               you can specify more than one input file, in order of layering\n");
	fprintf(stderr," -o <output file>\n");
	fprintf(stderr," -or <frame rate>\n");
	fprintf(stderr," -width <x>\n");
	fprintf(stderr," -height <x>\n");
	fprintf(stderr," -sqnr                         Squelch frame interpolation when frame rates match (1%% margin)\n");
	fprintf(stderr," -ffa                          Full frame alternate interpolation\n");
	fprintf(stderr," -fa <x>                       Interpolate alternate frames\n");
	fprintf(stderr," -gamma <x>                    Interpolate with gamma correction (number, ntsc, vga)\n");
	fprintf(stderr," -underscan <x>                Underscan the image during rendering\n");
	fprintf(stderr," -422                          Render to 4:2:2 colorspace\n");
	fprintf(stderr," -420                          Render to 4:2:0 colorspace\n");
}

static int parse_argv(int argc,char **argv) {
	const char *a;
	int i;

	for (i=1;i < argc;) {
		a = argv[i++];

		if (*a == '-') {
			do { a++; } while (*a == '-');

			if (!strcmp(a,"h") || !strcmp(a,"help")) {
				help(argv[0]);
				return 1;
			}
			else if (!strcmp(a,"width")) {
				a = argv[i++];
				if (a == NULL) return 1;
				output_width = (int)strtoul(a,NULL,0);
				if (output_width < 32) return 1;
			}
			else if (!strcmp(a,"height")) {
				a = argv[i++];
				if (a == NULL) return 1;
				output_height = (int)strtoul(a,NULL,0);
				if (output_height < 32) return 1;
			}
			else if (!strcmp(a,"sqnr")) {
				squelch_frameblend_near_match = true;
			}
			else if (!strcmp(a,"ffa")) {
				fullframealt = true;
			}
			else if (!strcmp(a,"fa")) {
				a = argv[i++];
				if (a == NULL) return 1;

				framealt = atoi(a);
				if (framealt < 1) framealt = 1;
				if (framealt > 8) framealt = 8;
			}
			else if (!strcmp(a,"gamma")) {
				a = argv[i++];
				if (a == NULL) return 1;

				if (isdigit(*a))
					gamma_correction = atof(a);
				else if (!strcmp(a,"vga") || !strcmp(a,"ntsc"))
					gamma_correction = 2.2;
			}
			else if (!strcmp(a,"i")) {
				a = argv[i++];
				if (a == NULL) return 1;
				new_input_file().path = a;
			}
			else if (!strcmp(a,"or")) {
				a = argv[i++];
				if (a == NULL) return 1;

				int d = 1;
				double n = strtof(a,(char**)(&a));
				if (*a == ':' || *a == '/' || *a == '\\') {
					a++;
					d = strtoul(a,(char**)(&a),10);
					if (d < 1) d = 1;
				}

				if (n < 0) n = 0;

				/* this code can cause problems below 5fps */
				if ((n/d) < 5) {
					n = 5;
					d = 1;
				}

				if (d > 1) {
					output_field_rate.num = (long)floor(n + 0.5);
					output_field_rate.den = (long)d;
				}
				else {
					output_field_rate.num = (long)floor((n * 10000) + 0.5);
					output_field_rate.den = (long)10000;
				}
			}
			else if (!strcmp(a,"o")) {
				a = argv[i++];
				if (a == NULL) return 1;
				output_file = a;
			}
			else if (!strcmp(a,"underscan")) {
				a = argv[i++];
				if (a == NULL) return 1;
				underscan = atoi(a);
				if (underscan < 0) underscan = 0;
				if (underscan > 99) underscan = 99;
			}
			else if (!strcmp(a,"422")) {
				use_422_colorspace = true;
			}
			else if (!strcmp(a,"420")) {
				use_422_colorspace = false;
			}
			else {
				fprintf(stderr,"Unknown switch '%s'\n",a);
				return 1;
			}
		}
		else {
			fprintf(stderr,"Unhandled arg '%s'\n",a);
			return 1;
		}
	}

	if (output_file.empty()) {
		fprintf(stderr,"No output file specified\n");
		return 1;
	}
	if (input_files.empty()) {
		fprintf(stderr,"No input files specified\n");
		return 1;
	}

	return 0;
}

// This code assumes ARGB and the frame match resolution/
void composite_layer(AVFrame *dstframe,AVFrame *srcframe,InputFile &inputfile) {
	uint32_t *dscan,*sscan;
	unsigned int x,y;
	unsigned int shr;

	if (dstframe == NULL || srcframe == NULL) return;
	if (dstframe->data[0] == NULL || srcframe->data[0] == 0) return;
	if (dstframe->linesize[0] < (dstframe->width*4)) return; // ARGB
	if (srcframe->linesize[0] < (srcframe->width*4)) return; // ARGB
	if (dstframe->width != srcframe->width) return;
	if (dstframe->height != srcframe->height) return;

	for (y=0;y < dstframe->height;y++) {
		sscan = (uint32_t*)(srcframe->data[0] + (srcframe->linesize[0] * y));
		dscan = (uint32_t*)(dstframe->data[0] + (dstframe->linesize[0] * y));
		for (x=0;x < dstframe->width;x++,dscan++,sscan++) {
			*dscan = *sscan;
		}
	}
}

int clamp255(int x) {
	if (x > 255)
		return 255;
	if (x < 0)
		return 0;
	return x;
}

double gamma_dec(double x) {
	return pow(x,gamma_correction);
}

unsigned long gamma_dec16_table[256];
unsigned long gamma_enc16_table[8192 + 1];

bool gamma16_init = false;

void gamma16_do_init(void);

unsigned long gamma_dec16(unsigned long x) {
	if (!gamma16_init) gamma16_do_init();

	if (x > 255u) x = 255u;

	return gamma_dec16_table[x];
}

double gamma_enc(double x) {
	return pow(x,1.0 / gamma_correction);
}

unsigned long gamma_enc16(unsigned long x) {
	if (!gamma16_init) gamma16_do_init();

	if (x > 8192u) x = 8192u;

	return gamma_enc16_table[x];
}

void gamma16_do_init(void) {
	gamma16_init = true;

	for (unsigned int i=0;i < 256;i++)
		gamma_dec16_table[i] = (unsigned long)(gamma_dec(i / 255.0) * 8192);

	for (unsigned int i=0;i <= 8192;i++)
		gamma_enc16_table[i] = (unsigned long)(gamma_enc(i / 8192.0) * 255);
}

int main(int argc,char **argv) {
	preset_NTSC();
	if (parse_argv(argc,argv))
		return 1;

	/* open all input files */
	for (std::vector<InputFile>::iterator i=input_files.begin();i!=input_files.end();i++) {
		if (!(*i).open_input()) {
			fprintf(stderr,"Failed to open %s\n",(*i).path.c_str());
			return 1;
		}
	}

	/* pick output */
	if (output_width < 1 || output_height < 1) {
		for (std::vector<InputFile>::iterator i=input_files.begin();i!=input_files.end();i++) {
			if ((*i).input_avstream_video_codec_context != NULL) {
				output_width = (*i).input_avstream_video_codec_context->width;
				output_height = (*i).input_avstream_video_codec_context->height;
				output_ar_n = (*i).input_avstream_video_codec_context->sample_aspect_ratio.num;
				output_ar_d = (*i).input_avstream_video_codec_context->sample_aspect_ratio.den;
				break;
			}
		}
	}
	fprintf(stderr,"Output frame: %d x %d with %d:%d PAR\n",output_width,output_height,output_ar_n,output_ar_d);

	/* no decision, no frame */
	if (output_width < 16 || output_height < 16) {
		fprintf(stderr,"None or invalid output dimensions\n");
		return 1;
	}

	/* open output file */
	assert(output_avfmt == NULL);
	if (avformat_alloc_output_context2(&output_avfmt,NULL,NULL,output_file.c_str()) < 0) {
		fprintf(stderr,"Failed to open output file\n");
		return 1;
	}

	{
		output_avstream_video = avformat_new_stream(output_avfmt, NULL);
		if (output_avstream_video == NULL) {
			fprintf(stderr,"Unable to create output video stream\n");
			return 1;
		}

		output_avstream_video_codec_context = avcodec_alloc_context3(avcodec_find_encoder(AV_CODEC_ID_H264));
		if (output_avstream_video_codec_context == NULL) {
			fprintf(stderr,"Output stream video no codec context?\n");
			return 1;
		}

		output_avstream_video_codec_context->width = output_width;
		output_avstream_video_codec_context->height = output_height;
		output_avstream_video_codec_context->sample_aspect_ratio = AVRational{output_ar_n,output_ar_d};
		output_avstream_video_codec_context->pix_fmt = use_422_colorspace ? AV_PIX_FMT_YUV422P : AV_PIX_FMT_YUV420P;
		output_avstream_video_codec_context->gop_size = 15;
		output_avstream_video_codec_context->max_b_frames = 0;
		output_avstream_video_codec_context->bit_rate = 25000000;
		output_avstream_video_codec_context->time_base = AVRational{output_field_rate.den, output_field_rate.num};

		output_avstream_video->time_base = output_avstream_video_codec_context->time_base;
		if (output_avfmt->oformat->flags & AVFMT_GLOBALHEADER)
			output_avstream_video_codec_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

		if (avcodec_open2(output_avstream_video_codec_context,avcodec_find_encoder(AV_CODEC_ID_H264),NULL) < 0) {
			fprintf(stderr,"Output stream cannot open codec\n");
			return 1;
		}

		if (avcodec_parameters_from_context(output_avstream_video->codecpar,output_avstream_video_codec_context) < 0)
			fprintf(stderr,"WARNING: parameters from context failed\n");
	}

	if (!(output_avfmt->oformat->flags & AVFMT_NOFILE)) {
		if (avio_open(&output_avfmt->pb, output_file.c_str(), AVIO_FLAG_WRITE) < 0) {
			fprintf(stderr,"Output file cannot open file\n");
			return 1;
		}
	}

	if (avformat_write_header(output_avfmt,NULL) < 0) {
		fprintf(stderr,"Failed to write header\n");
		return 1;
	}

	/* soft break on CTRL+C */
	install_signal_handlers();

#ifdef HAVE_CUDA
	cuda_available = frameblend_cuda_init();
#endif

	/* prepare video encoding */
	output_avstream_video_frame = av_frame_alloc();
	if (output_avstream_video_frame == NULL) {
		fprintf(stderr,"Failed to alloc video frame\n");
		return 1;
	}
	output_avstream_video_frame->format = AV_PIX_FMT_BGRA;
	output_avstream_video_frame->height = output_height;
	output_avstream_video_frame->width = output_width;
	if (av_frame_get_buffer(output_avstream_video_frame,64) < 0) {
		fprintf(stderr,"Failed to alloc render frame\n");
		return 1;
	}

	{
		output_avstream_video_encode_frame = av_frame_alloc();
		if (output_avstream_video_encode_frame == NULL) {
			fprintf(stderr,"Failed to alloc video frame3\n");
			return 1;
		}
		output_avstream_video_encode_frame->colorspace = AVCOL_SPC_SMPTE170M;
		output_avstream_video_encode_frame->color_range = AVCOL_RANGE_MPEG;
		output_avstream_video_encode_frame->format = output_avstream_video_codec_context->pix_fmt;
		output_avstream_video_encode_frame->height = output_height;
		output_avstream_video_encode_frame->width = output_width;
		if (av_frame_get_buffer(output_avstream_video_encode_frame,64) < 0) {
			fprintf(stderr,"Failed to alloc render frame2\n");
			return 1;
		}
	}

	if (output_avstream_video_resampler == NULL) {
		output_avstream_video_resampler = sws_getContext(
			// source
			output_avstream_video_frame->width,
			output_avstream_video_frame->height,
			(AVPixelFormat)output_avstream_video_frame->format,
			// dest
			output_avstream_video_encode_frame->width,
			output_avstream_video_encode_frame->height,
			(AVPixelFormat)output_avstream_video_encode_frame->format,
			// opt
			SWS_BILINEAR, NULL, NULL, NULL);
		if (output_avstream_video_resampler == NULL) {
			fprintf(stderr,"Failed to alloc ARGB -> codec converter\n");
			return 1;
		}
	}

	/* run all inputs and render to output, until done */
	{
		signed long long outbase=0;

		memset(output_avstream_video_frame->data[0],0x00,output_avstream_video_frame->linesize[0] * output_avstream_video_frame->height);

		for (std::vector<InputFile>::iterator ifi=input_files.begin();ifi!=input_files.end();ifi++) {
			signed long long current=0;

			if (DIE) break;

			InputFile &input_file = *ifi;

			while (!input_file.eof && !DIE) {
				if (!input_file.got_video)
					input_file.next_packet();
				else
					break;
			}

			if (input_file.input_avstream_video_frame != NULL && input_file.got_video) {
				input_file.frame_copy_scale();
				input_file.got_video = false;
			}

			std::vector<uint32_t*> frames; /* they're all the same RGBA frame, so just store a pointer */
			std::vector<double> frame_t;

			if (input_file.input_avstream_video_frame_rgb != NULL) {
				frame_t.push_back(input_file.video_frame_rgb_to_output_f());
				frames.push_back(input_file.copy_rgba(input_file.input_avstream_video_frame_rgb));
			}

			while (!DIE) {
				while (!input_file.eof && !DIE && input_file.video_frame_to_output_f() < (current + 30LL)) {
					input_file.next_packet();

					if (input_file.input_avstream_video_frame != NULL && input_file.got_video) {
						input_file.frame_copy_scale();
						input_file.got_video = false;

						if (input_file.input_avstream_video_frame_rgb != NULL) {
							frame_t.push_back(input_file.video_frame_rgb_to_output_f());
							frames.push_back(input_file.copy_rgba(input_file.input_avstream_video_frame_rgb));
						}
					}
				}

				if (input_file.eof &&
					(input_file.video_frame_to_output_f() < -1000/*AV_NOPTS_VALUE*/ ||
					 current > (unsigned long long)ceil(input_file.video_frame_to_output_f())))
					break;

				/* cross-blending weights for this frame period */
				std::vector< pair<size_t,double> > weights;
				assert(frames.size() == frame_t.size());
				size_t cutoff = 0;

				/* scan for first frame to render */
				if (frames.size() > 1) {
					if (framealt > 1) {
						for (size_t i=(size_t)((unsigned long long)current % (unsigned long long)framealt);(i+(size_t)framealt) < frames.size();i += (size_t)framealt) {
							double bt = frame_t[i];
							double et = frame_t[i+framealt];

							if (i != 0) {
								if ((et + 2.0) < current) {
									cutoff = i - (i % framealt);
								}
							}

							if (bt < current)
								bt = current;
							if (bt > (current + (fullframealt ? framealt : 1)))
								bt = (current + (fullframealt ? framealt : 1));

							if (et < current)
								et = current;
							if (et > (current + (fullframealt ? framealt : 1)))
								et = (current + (fullframealt ? framealt : 1));

							assert(bt <= et);

							if (bt < et)
								weights.push_back(pair<size_t,double>(i,(et-bt) / (fullframealt ? framealt : 1)));
						}
					}
					else {
						for (size_t i=0;(i+1ul) < frames.size();i++) {
							double bt = frame_t[i];
							double et = frame_t[i+1];

							if (i != 0) {
								if ((et + 2.0) < current) {
									cutoff = i;
								}
							}

							if (bt < current)
								bt = current;
							if (bt > (current + 1ll))
								bt = (current + 1ll);

							if (et < current)
								et = current;
							if (et > (current + 1ll))
								et = (current + 1ll);

							assert(bt <= et);

							if (bt < et)
								weights.push_back(pair<size_t,double>(i,et-bt));
						}
					}
				}

				if (weights.size() == 0 && frames.size() > cutoff)
					weights.push_back(pair<size_t,double>(cutoff,1.0));

				if (squelch_frameblend_near_match) {
					if (weights.size() == 2 || weights.size() == 3) {
						double sq = 1.0;

						assert(weights[0].first < frame_t.size());
						assert(weights[1].first < frame_t.size());

						double bt = frame_t[weights[0].first];
						double et = frame_t[weights[1].first];

						sq = fabs((et - bt) - 1.0) / 0.01; /* start squelching if less than 1% difference between source & dest rate */
						if (sq < 1.0) {
							sq = pow(sq,2.0);

							if (sq > 0.01) {
								if (weights[0].second > sq) weights[0].second = sq;
								weights[0].second /= sq;
								weights[1].second = 1.0 - weights[0].second;
							}
							else {
								weights[0].second = 1.0;
								weights[1].second = 0.0;
							}

							if (weights.size() > 2)
								weights[2].second = 0.0;
						}
					}
				}

				std::vector<unsigned int> weight16;

				for (size_t i=0;i < weights.size();i++)
					weight16.push_back((unsigned int)floor((weights[i].second * 0x10000)+0.5));

				assert(weights.size() == weight16.size());

#ifdef HAVE_CUDA
				if (cuda_available && weight16.size() <= 8) {
					// Build frame pointer + weight arrays for CUDA
					// (weights[wi].first indexes into frames[], we must
					//  repack so the CUDA wrapper gets contiguous arrays)
					int nf = (int)weight16.size();
					const unsigned char* cuda_frame_ptrs[8];
					unsigned int cuda_weights[8];
					for (int wi = 0; wi < nf; wi++) {
						size_t fi = weights[wi].first;
						assert(fi < frames.size());
						cuda_frame_ptrs[wi] = (const unsigned char*)frames[fi];
						cuda_weights[wi] = weight16[wi];
					}

					int linesize_in  = input_file.input_avstream_video_frame_rgb->linesize[0];
					int linesize_out = output_avstream_video_frame->linesize[0];

					if (gamma_correction > 1) {
						// Ensure gamma LUTs are initialized
						if (!gamma16_init) gamma16_do_init();
						frameblend_cuda_gamma(
							cuda_frame_ptrs, cuda_weights, nf,
							(unsigned char*)output_avstream_video_frame->data[0],
							output_width, output_height,
							linesize_in, linesize_out,
							gamma_dec16_table, gamma_enc16_table);
					}
					else {
						frameblend_cuda_linear(
							cuda_frame_ptrs, cuda_weights, nf,
							(unsigned char*)output_avstream_video_frame->data[0],
							output_width, output_height,
							linesize_in, linesize_out);
					}
				}
				else
#endif
				if (gamma_correction > 1) {
					for (unsigned int y=0;y < output_height;y++) {
						unsigned char *outframe = (unsigned char*)(output_avstream_video_frame->data[0] + (y * (output_avstream_video_frame->linesize[0])));
						for (unsigned int x=0;x < output_width;x++) {
							unsigned long long r = 0,g = 0,b = 0;

							for (size_t wi=0;wi < weight16.size();wi++) {
								size_t fi = weights[wi].first;
								assert(fi < frames.size());
								unsigned char *inframe = ((unsigned char*)frames[fi] + (y * (input_file.input_avstream_video_frame_rgb->linesize[0]))) + (x * 4u);

								b += gamma_dec16(inframe[0]) * (unsigned long long)weight16[wi];
								g += gamma_dec16(inframe[1]) * (unsigned long long)weight16[wi];
								r += gamma_dec16(inframe[2]) * (unsigned long long)weight16[wi];
							}

							outframe[0] = clamp255(gamma_enc16(b >> 16ull));
							outframe[1] = clamp255(gamma_enc16(g >> 16ull));
							outframe[2] = clamp255(gamma_enc16(r >> 16ull));
							outframe[3] = 0xFF;

							outframe += 4;
						}
					}
				}
				else {
					for (unsigned int y=0;y < output_height;y++) {
						unsigned char *outframe = (unsigned char*)(output_avstream_video_frame->data[0] + (y * (output_avstream_video_frame->linesize[0])));
						for (unsigned int x=0;x < output_width;x++) {
							unsigned long r = 0,g = 0,b = 0;

							for (size_t wi=0;wi < weight16.size();wi++) {
								size_t fi = weights[wi].first;
								assert(fi < frames.size());
								unsigned char *inframe = ((unsigned char*)frames[fi] + (y * (input_file.input_avstream_video_frame_rgb->linesize[0]))) + (x * 4u);

								b += inframe[0] * (unsigned long)weight16[wi];
								g += inframe[1] * (unsigned long)weight16[wi];
								r += inframe[2] * (unsigned long)weight16[wi];
							}

							outframe[0] = clamp255(b >> 16ul);
							outframe[1] = clamp255(g >> 16ul);
							outframe[2] = clamp255(r >> 16ul);
							outframe[3] = 0xFF;

							outframe += 4;
						}
					}
				}

				output_avstream_video_frame->pts = current;
				output_avstream_video_frame->flags &= ~(AV_FRAME_FLAG_TOP_FIELD_FIRST | AV_FRAME_FLAG_INTERLACED);

				// convert ARGB to whatever the codec demands, and encode
				output_avstream_video_encode_frame->pts = output_avstream_video_frame->pts;
				output_avstream_video_encode_frame->flags = (output_avstream_video_encode_frame->flags & ~(AV_FRAME_FLAG_TOP_FIELD_FIRST | AV_FRAME_FLAG_INTERLACED)) |
					(output_avstream_video_frame->flags & (AV_FRAME_FLAG_TOP_FIELD_FIRST | AV_FRAME_FLAG_INTERLACED));

				if (sws_scale(output_avstream_video_resampler,
					// source
					output_avstream_video_frame->data,
					output_avstream_video_frame->linesize,
					0,output_avstream_video_frame->height,
					// dest
					output_avstream_video_encode_frame->data,
					output_avstream_video_encode_frame->linesize) <= 0)
					fprintf(stderr,"WARNING: sws_scale failed\n");

				output_frame(output_avstream_video_encode_frame,current);
				current++;

				if (cutoff >= 32) {
					assert(frame_t.size() > cutoff);
					assert(frames.size() > cutoff);

					for (size_t i=0;i < cutoff;i++) {
						if (frames[i] != NULL) {
							delete[] frames[i];
							frames[i] = NULL;
						}
					}

					frame_t.erase(frame_t.begin(),frame_t.begin()+cutoff);
					frames.erase(frames.begin(),frames.begin()+cutoff);
				}
			}

			for (size_t i=0;i < frames.size();i++) {
				if (frames[i] != NULL) {
					delete[] frames[i];
					frames[i] = NULL;
				}
			}
			frames.clear();
			frame_t.clear();

			outbase += current;
		}
	}

	/* flush encoder, write trailer, close output */
	if (output_avstream_video_frame != NULL)
		av_frame_free(&output_avstream_video_frame);
	flush_encoder_and_close();

	/* close all */
	for (std::vector<InputFile>::iterator i=input_files.begin();i!=input_files.end();i++)
		(*i).close_input();

#ifdef HAVE_CUDA
	if (cuda_available)
		frameblend_cuda_shutdown();
#endif

	return 0;
}
