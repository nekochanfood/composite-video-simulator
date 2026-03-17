// output_context.h — Output video encoding and muxing helpers
// Shared by all 4 programs for output_frame, encoder flush, and cleanup.
//
// Requires these globals to be defined in the including .cpp:
//   AVFormatContext* output_avfmt;
//   AVStream* output_avstream_video;
//   AVCodecContext* output_avstream_video_codec_context;
//   struct SwsContext* output_avstream_video_resampler;
//   AVFrame* output_avstream_video_encode_frame;
#ifndef OUTPUT_CONTEXT_H
#define OUTPUT_CONTEXT_H

#include "common.h"

extern AVFormatContext* output_avfmt;
extern AVStream* output_avstream_video;
extern AVCodecContext* output_avstream_video_codec_context;
extern struct SwsContext* output_avstream_video_resampler;
extern AVFrame* output_avstream_video_encode_frame;

// Encode and write one video frame to the output file
static void output_frame(AVFrame *frame, unsigned long long field_number) {
	if ((field_number % (15ULL * 2ULL)) == 0)
		frame->flags |= AV_FRAME_FLAG_KEY;
	else
		frame->flags &= ~AV_FRAME_FLAG_KEY;

	{
		frame->flags &= ~AV_FRAME_FLAG_INTERLACED;
		frame->pts = field_number;
	}

	fprintf(stderr,"\x0D" "Output field %llu ",field_number); fflush(stderr);

	if (avcodec_send_frame(output_avstream_video_codec_context, frame) >= 0) {
		AVPacket *pkt = av_packet_alloc();
		if (pkt != NULL) {
			while (avcodec_receive_packet(output_avstream_video_codec_context, pkt) >= 0) {
				pkt->stream_index = output_avstream_video->index;
				av_packet_rescale_ts(pkt, output_avstream_video_codec_context->time_base, output_avstream_video->time_base);

				if (av_interleaved_write_frame(output_avfmt, pkt) < 0)
					fprintf(stderr,"AV write frame failed video\n");
				av_packet_unref(pkt);
			}
			av_packet_free(&pkt);
		}
	}
}

// Flush remaining frames from the encoder and write trailer
static void flush_encoder_and_close(void) {
	/* flush encoder delay */
	fprintf(stderr,"Flushing delayed frames\n");
	{
		AVPacket *pkt = av_packet_alloc();
		if (pkt != NULL) {
			avcodec_send_frame(output_avstream_video_codec_context, NULL);
			while (avcodec_receive_packet(output_avstream_video_codec_context, pkt) >= 0) {
				pkt->stream_index = output_avstream_video->index;
				av_packet_rescale_ts(pkt, output_avstream_video_codec_context->time_base, output_avstream_video->time_base);

				if (av_interleaved_write_frame(output_avfmt, pkt) < 0)
					fprintf(stderr,"AV write frame failed video\n");
				av_packet_unref(pkt);
			}
			av_packet_free(&pkt);
		}
	}
	fprintf(stderr,"Flushing delayed frames--done\n");

	/* close output */
	if (output_avstream_video_resampler != NULL) {
		sws_freeContext(output_avstream_video_resampler);
		output_avstream_video_resampler = NULL;
	}
	if (output_avstream_video_encode_frame != NULL)
		av_frame_free(&output_avstream_video_encode_frame);

	av_write_trailer(output_avfmt);

	if (output_avstream_video_codec_context != NULL)
		avcodec_free_context(&output_avstream_video_codec_context);
	if (output_avfmt != NULL && !(output_avfmt->oformat->flags & AVFMT_NOFILE))
		avio_closep(&output_avfmt->pb);
	avformat_free_context(output_avfmt);
	output_avfmt = NULL;
}

// Install signal handlers for graceful shutdown
static void install_signal_handlers(void) {
	signal(SIGINT, sigma);
	signal(SIGTERM, sigma);
#ifndef _WIN32
	signal(SIGHUP, sigma);
	signal(SIGQUIT, sigma);
#endif
}

#endif // OUTPUT_CONTEXT_H
