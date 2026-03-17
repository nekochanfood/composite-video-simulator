
#include "common.h"
#include "input_file.h"
#include "output_context.h"

#ifdef HAVE_CUDA
#include "filmac_cuda.h"
static bool cuda_available = false;
#endif

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

InputFile                   input_file;
std::string                 output_file;

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
				input_file.path = a;
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
	if (input_file.path.empty()) {
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
	if (!input_file.open_input()) {
		fprintf(stderr,"Failed to open %s\n",input_file.path.c_str());
		return 1;
	}

	/* pick output */
	if (output_width < 1 || output_height < 1) {
		if (input_file.input_avstream_video_codec_context != NULL) {
			output_width = input_file.input_avstream_video_codec_context->width;
			output_height = input_file.input_avstream_video_codec_context->height;
			output_ar_n = input_file.input_avstream_video_codec_context->sample_aspect_ratio.num;
			output_ar_d = input_file.input_avstream_video_codec_context->sample_aspect_ratio.den;
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
		AVDictionary *opt_dict = NULL;

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
		output_avstream_video_codec_context->time_base = AVRational{output_field_rate.den, output_field_rate.num};

		av_dict_set(&opt_dict,"crf","16",0);
		av_dict_set(&opt_dict,"crf_max","16",0);
		av_dict_set(&opt_dict,"preset","superfast",0);

		output_avstream_video->time_base = output_avstream_video_codec_context->time_base;
		if (output_avfmt->oformat->flags & AVFMT_GLOBALHEADER)
			output_avstream_video_codec_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

		if (avcodec_open2(output_avstream_video_codec_context,avcodec_find_encoder(AV_CODEC_ID_H264),&opt_dict) < 0) {
			fprintf(stderr,"Output stream cannot open codec\n");
			return 1;
		}

		if (avcodec_parameters_from_context(output_avstream_video->codecpar,output_avstream_video_codec_context) < 0)
			fprintf(stderr,"WARNING: parameters from context failed\n");

		av_dict_free(&opt_dict);
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
	cuda_available = filmac_cuda_init();
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
		long final_minv = -1,final_maxv = -1;
		bool final_init = false;

		memset(output_avstream_video_frame->data[0],0x00,output_avstream_video_frame->linesize[0] * output_avstream_video_frame->height);

		{
			signed long long current=0;

			std::vector<uint32_t*> frames; /* they're all the same RGBA frame, so just store a pointer */

			while (!DIE) {
				if (input_file.eof)
					break;

				input_file.next_packet();

				if (input_file.input_avstream_video_frame != NULL && input_file.got_video) {
					input_file.frame_copy_scale();
					input_file.got_video = false;

					if (input_file.input_avstream_video_frame_rgb != NULL) {
						frames.push_back(input_file.copy_rgba(input_file.input_avstream_video_frame_rgb));
						current = input_file.video_frame_rgb_to_output_f();
					}
				}
				else {
					continue;
				}

				long *lframe = NULL;

#ifdef HAVE_CUDA
				if (cuda_available) {
					// Ensure gamma LUTs are initialized before passing to CUDA
					if (gamma_correction > 1 && !gamma16_init) gamma16_do_init();

					filmac_cuda_process(
						(const unsigned char*)frames[0],
						(unsigned char*)output_avstream_video_frame->data[0],
						output_width, output_height,
						input_file.input_avstream_video_frame_rgb->linesize[0],
						output_avstream_video_frame->linesize[0],
						gamma_correction > 1,
						gamma_dec16_table, gamma_enc16_table,
						&final_minv, &final_maxv, &final_init);
				}
				else
#endif
				{
				lframe = new long[output_width*output_height*3];
				if (gamma_correction > 1) {
					for (unsigned int y=0;y < output_height;y++) {
						unsigned char *outframe = (unsigned char*)(output_avstream_video_frame->data[0] + (y * (output_avstream_video_frame->linesize[0])));
						unsigned char *inframe = ((unsigned char*)frames[0] + (y * (input_file.input_avstream_video_frame_rgb->linesize[0])));
						long *longframe = lframe + (y * output_width * 3);

						for (unsigned int x=0;x < output_width;x++) {
							longframe[x*3+0] = gamma_dec16(inframe[x*4+0]) << 16ul;
							longframe[x*3+1] = gamma_dec16(inframe[x*4+1]) << 16ul;
							longframe[x*3+2] = gamma_dec16(inframe[x*4+2]) << 16ul;
						}
					}
				}
				else {
					for (unsigned int y=0;y < output_height;y++) {
						unsigned char *outframe = (unsigned char*)(output_avstream_video_frame->data[0] + (y * (output_avstream_video_frame->linesize[0])));
						unsigned char *inframe = ((unsigned char*)frames[0] + (y * (input_file.input_avstream_video_frame_rgb->linesize[0])));
						long *longframe = lframe + (y * output_width * 3);

						for (unsigned int x=0;x < output_width;x++) {
							longframe[x*3+0] = inframe[x*4+0] << 16ul;
							longframe[x*3+1] = inframe[x*4+1] << 16ul;
							longframe[x*3+2] = inframe[x*4+2] << 16ul;
						}
					}
				}

				long scaleto = gamma_correction > 1 ? (0x10000l * 8192l) : (0x10000l * 256l);
				long minv = (scaleto * 6l) / 10l;
				long maxv = (scaleto * 4l) / 10l;

				{
					unsigned int minx = (output_width*15)/100;
					unsigned int maxx = (output_width*90)/100;
					unsigned int miny = (output_height*0)/100;
					unsigned int maxy = (output_height*100)/100;
					unsigned int blw = 128,blh = 128;

					for (unsigned int y=miny;y < maxy;y += blh) {
						for (unsigned int x=minx;x < maxx;x += blw) {
							long long grmin = 0,grmax = 0,grd = 0;

							for (unsigned int sy=0;sy < blh;sy++) {
								long *longframe = lframe + ((y+sy) * output_width * 3);
								for (unsigned int sx=0;sx < blw;sx++) {
									if ((x+sx) >= output_width || (y+sy) >= output_height) continue;
									/* BGR */
									grd++;
									grmin += std::min(std::min(longframe[(x+sx)*3+0],longframe[(x+sx)*3+1]),longframe[(x+sx)*3+2]);

									grmax = std::max(std::max(longframe[(x+sx)*3+0],longframe[(x+sx)*3+1]),longframe[(x+sx)*3+2]);
									if (maxv < grmax) maxv = grmax;
								}
							}

							grmin += grd/2l;
							grmin /= grd;
							grmax += grd/2l;
							grmax /= grd;

							if (minv > grmin)
								minv = grmin;
						}
					}
				}

				if (minv == maxv) maxv++;

				if (!final_init) {
					final_init = true;
					final_minv = minv;
					final_maxv = maxv;
				}
				else {
					if (final_maxv < maxv)
						final_maxv = ((final_maxv*1l) + maxv) / 2l;
					else
						final_maxv = ((final_maxv*4l) + maxv) / 5l;

					if (final_minv > minv)
						final_minv = ((final_minv*1l) + minv) / 2l;
					else
						final_minv = ((final_minv*4l) + minv) / 5l;
				}

//				fprintf(stderr,"\nmin=%.15f max=%.15f gc=%lu\n",(double)minv/scaleto,(double)maxv/scaleto,scaleto);

				for (unsigned int y=0;y < output_height;y++) {
					long *longframe = lframe + (y * output_width * 3);
					for (unsigned int x=0;x < output_width*3;x++) {
						long long v = (((long long)(longframe[x] - final_minv)) * (long long)scaleto) / ((long long)(final_maxv - final_minv));
						if (v < -0x7FFFFFFFl) v = -0x7FFFFFFFl;
						if (v >  0x7FFFFFFFl) v =  0x7FFFFFFFl;
						longframe[x] = (long)v;
					}
				}

#if 0
				{
					unsigned long tl=0,th=0;

					for (unsigned int y=0;y < output_height;y++) {
						unsigned char *outframe = (unsigned char*)(output_avstream_video_frame->data[0] + (y * (output_avstream_video_frame->linesize[0])));
						unsigned char *inframe = ((unsigned char*)frames[0] + (y * (input_file.input_avstream_video_frame_rgb->linesize[0])));
						long *longframe = lframe + (y * output_width * 3);

						for (unsigned int x=0;x < output_width;x++) {
							bool itl = std::min(std::min(longframe[x*3+0],longframe[x*3+1]),longframe[x*3+2]) < 0;
							bool ith = std::max(std::max(longframe[x*3+0],longframe[x*3+1]),longframe[x*3+2]) > scaleto;
							if (ith) { longframe[x*3+2] = scaleto; longframe[x*3+1] = 0x00; longframe[x*3+0] = 0x00; }
							else if (itl) { longframe[x*3+2] = 0x00; longframe[x*3+1] = scaleto; longframe[x*3+0] = 0x00; }
							tl += itl?1:0;
							th += ith?1:0;
						}
					}

					fprintf(stderr,"toolow=%lu toohigh=%lu\n",tl,th);
				}
#endif

				if (gamma_correction > 1) {
					for (unsigned int y=0;y < output_height;y++) {
						unsigned char *outframe = (unsigned char*)(output_avstream_video_frame->data[0] + (y * (output_avstream_video_frame->linesize[0])));
						unsigned char *inframe = ((unsigned char*)frames[0] + (y * (input_file.input_avstream_video_frame_rgb->linesize[0])));
						long *longframe = lframe + (y * output_width * 3);

						for (unsigned int x=0;x < output_width;x++) {
							outframe[x*4+0] = clamp255(gamma_enc16(std::max(0l,longframe[x*3+0] >> 16l)));
							outframe[x*4+1] = clamp255(gamma_enc16(std::max(0l,longframe[x*3+1] >> 16l)));
							outframe[x*4+2] = clamp255(gamma_enc16(std::max(0l,longframe[x*3+2] >> 16l)));
							outframe[x*4+3] = 0xFF;
						}
					}
				}
				else {
					for (unsigned int y=0;y < output_height;y++) {
						unsigned char *outframe = (unsigned char*)(output_avstream_video_frame->data[0] + (y * (output_avstream_video_frame->linesize[0])));
						unsigned char *inframe = ((unsigned char*)frames[0] + (y * (input_file.input_avstream_video_frame_rgb->linesize[0])));
						long *longframe = lframe + (y * output_width * 3);

						for (unsigned int x=0;x < output_width;x++) {
							outframe[x*4+0] = clamp255(std::max(0l,longframe[x*3+0] >> 16l));
							outframe[x*4+1] = clamp255(std::max(0l,longframe[x*3+1] >> 16l));
							outframe[x*4+2] = clamp255(std::max(0l,longframe[x*3+2] >> 16l));
							outframe[x*4+3] = 0xFF;
						}
					}
				}

				delete[] lframe;
				} /* end CPU fallback */

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

				if (frames.size() != 0) {
					for (size_t i=0;i < frames.size();i++) {
						if (frames[i] != NULL) {
							delete[] frames[i];
							frames[i] = NULL;
						}
					}

					frames.erase(frames.begin(),frames.end());
				}
			}

			for (size_t i=0;i < frames.size();i++) {
				if (frames[i] != NULL) {
					delete[] frames[i];
					frames[i] = NULL;
				}
			}
			frames.clear();
		}
	}

	/* flush encoder, write trailer, close output */
	if (output_avstream_video_frame != NULL)
		av_frame_free(&output_avstream_video_frame);
	flush_encoder_and_close();

	/* close all */
	input_file.close_input();

#ifdef HAVE_CUDA
	if (cuda_available)
		filmac_cuda_shutdown();
#endif

	return 0;
}
