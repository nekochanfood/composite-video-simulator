// spout_ntsc.cpp — Real-time NTSC/VHS composite video simulation proxy via Spout2
//
// Receives video from a Spout2 sender, applies NTSC composite signal emulation
// (same processing as ffmpeg_ntsc), and re-broadcasts via Spout2 for OBS etc.
//
// Usage:
//   spout_ntsc [options]
//     -receiver <name>   Spout2 source name to receive from (default: active sender)
//     -sender <name>     Spout2 sender name to broadcast as (default: "NTSC-Output")
//     -fps <n>           Target frame rate (default: 60)
//     -gui               Show Qt preview window for monitoring
//     -tvstd <pal|ntsc>  TV standard (default: ntsc)
//     -vhs               Enable VHS emulation
//     ... (all standard NTSC parameters from ffmpeg_ntsc)

// Platform headers (before FFmpeg to avoid type conflicts)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmsystem.h>  // timeBeginPeriod/timeEndPeriod for precision timing
#pragma comment(lib, "winmm.lib")

// Spout2 DirectX API (no OpenGL required)
#include "SpoutDX.h"

// FFmpeg headers for AVFrame, sws_scale (pixel format conversion & scaling)
extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/pixfmt.h>
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

#include "lowpass_filter.h"
#include "color_convert.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef HAVE_CUDA
#include "ntsc_cuda.h"
static bool cuda_available = false;

struct GpuPipelineState {
    AVFrame* gpu_output_frame[NTSC_CUDA_NUM_BUFFERS];
    int current_buf_idx;
    bool has_pending_frame;
    int pending_buf_idx;
    signed long long pending_fieldno;

    GpuPipelineState() : current_buf_idx(0), has_pending_frame(false), pending_buf_idx(0), pending_fieldno(0) {
        gpu_output_frame[0] = nullptr;
        gpu_output_frame[1] = nullptr;
    }
};
static GpuPipelineState gpu_pipeline;
#endif

// ─── Qt GUI forward declarations (implemented at bottom if HAVE_QT) ─────────
#ifdef HAVE_QT
#include <QApplication>
#include <QMainWindow>
#include <QLabel>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <mutex>

static std::mutex g_gui_frame_mutex;
static QImage g_gui_frame_image;
static bool g_gui_frame_dirty = false;

class PreviewWindow : public QMainWindow {
    Q_OBJECT
public:
    PreviewWindow(QWidget *parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("spout_ntsc - Preview");
        m_label = new QLabel(this);
        m_label->setAlignment(Qt::AlignCenter);
        m_label->setMinimumSize(360, 240);
        setCentralWidget(m_label);
        resize(720, 480);

        m_timer = new QTimer(this);
        connect(m_timer, &QTimer::timeout, this, &PreviewWindow::updateFrame);
        m_timer->start(33); // ~30fps refresh
    }

private slots:
    void updateFrame() {
        std::lock_guard<std::mutex> lock(g_gui_frame_mutex);
        if (g_gui_frame_dirty && !g_gui_frame_image.isNull()) {
            m_label->setPixmap(QPixmap::fromImage(g_gui_frame_image).scaled(
                m_label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            g_gui_frame_dirty = false;
        }
    }

private:
    QLabel *m_label;
    QTimer *m_timer;
};
#endif // HAVE_QT

// ─── Graceful shutdown ──────────────────────────────────────────────────────
static volatile int DIE = 0;

static BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
            if (++DIE >= 20) abort();
            return TRUE;
        default:
            return FALSE;
    }
}

// ─── Processing parameters (same as ffmpeg_ntsc) ────────────────────────────
static AVRational output_field_rate = { 60000, 1001 };
static int     output_width = 720;
static int     output_height = 480;
static bool    output_ntsc = true;
static bool    output_pal = false;
static int     video_scanline_phase_shift = 180;
static int     video_scanline_phase_shift_offset = 0;

static double  composite_preemphasis = 0;
static double  composite_preemphasis_cut = 1000000;

static double  vhs_out_sharpen = 1.5;

static bool    vhs_head_switching = false;
static double  vhs_head_switching_point = 1.0 - ((4.5+0.01) / 262.5);
static double  vhs_head_switching_phase = ((1.0-0.01) / 262.5);
static double  vhs_head_switching_phase_noise = (((1.0 / 500)) / 262.5);

static bool    composite_in_chroma_lowpass = true;
static bool    composite_out_chroma_lowpass = true;
static bool    composite_out_chroma_lowpass_lite = true;

static int     video_yc_recombine = 0;
static int     video_color_fields = 4;
static int     video_chroma_noise = 0;
static int     video_chroma_phase_noise = 0;
static int     video_chroma_loss = 0;
static int     video_noise = 2;
static int     subcarrier_amplitude = 50;
static int     subcarrier_amplitude_back = 50;

static bool    emulating_vhs = false;
static bool    nocolor_subcarrier = false;
static bool    nocolor_subcarrier_after_yc_sep = false;
static bool    vhs_chroma_vert_blend = true;
static bool    vhs_svideo_out = false;
static bool    enable_composite_emulation = true;

enum {
    VHS_SP=0,
    VHS_LP,
    VHS_EP
};

static int     output_vhs_tape_speed = VHS_SP;

// ─── Spout-specific parameters ──────────────────────────────────────────────
static std::string spout_receiver_name;  // empty = active sender
static std::string spout_sender_name = "NTSC-Output";
static int     target_fps = 60;
static bool    enable_gui = false;
static bool    stretch_mode = false;  // false=center-crop, true=stretch to fill
static int     gpu_priority_level = 2; // 0=low, 1=normal, 2=high (default: high for backward compat)

// ─── Presets ────────────────────────────────────────────────────────────────
static void preset_PAL() {
    output_field_rate.num = 50;
    output_field_rate.den = 1;
    output_height = 576;
    output_width = 720;
    output_pal = true;
    output_ntsc = false;
}

static void preset_NTSC() {
    output_field_rate.num = 60000;
    output_field_rate.den = 1001;
    output_height = 480;
    output_width = 720;
    output_pal = false;
    output_ntsc = true;
}

// ─── Help ───────────────────────────────────────────────────────────────────
static void help(const char *arg0) {
    fprintf(stderr,"%s [options]\n",arg0);
    fprintf(stderr,"\n");
    fprintf(stderr," Spout2 options:\n");
    fprintf(stderr,"   -receiver <name>           Spout2 source to receive from (default: active sender)\n");
    fprintf(stderr,"   -sender <name>             Spout2 sender name (default: NTSC-Output)\n");
    fprintf(stderr,"   -fps <n>                   Target output frame rate (default: 60)\n");
    fprintf(stderr,"   -stretch                   Stretch source to fill target (default: center-crop)\n");
    fprintf(stderr,"   -priority <low|normal|high> GPU/thread priority (default: high)\n");
    fprintf(stderr,"                               low = reduced GPU usage, yields CPU time\n");
    fprintf(stderr,"                               normal = balanced\n");
    fprintf(stderr,"                               high = maximum performance (default)\n");
#ifdef HAVE_QT
    fprintf(stderr,"   -gui                       Show Qt preview window\n");
#endif
    fprintf(stderr,"\n");
    fprintf(stderr," Video standard:\n");
    fprintf(stderr,"   -tvstd <pal|ntsc>          TV standard (default: ntsc)\n");
    fprintf(stderr,"   -width <n>                 Processing width (default: 720)\n");
    fprintf(stderr,"\n");
    fprintf(stderr," NTSC/Composite options:\n");
    fprintf(stderr,"   -vhs                       Enable VHS emulation\n");
    fprintf(stderr,"   -vhs-speed <ep|lp|sp>      VHS tape speed (default: sp)\n");
    fprintf(stderr,"   -noise <0..100>            Noise amplitude\n");
    fprintf(stderr,"   -chroma-noise <0..100>     Chroma noise amplitude\n");
    fprintf(stderr,"   -chroma-phase-noise <x>    Chroma phase noise (0..100)\n");
    fprintf(stderr,"   -subcarrier-amp <0..100>   Subcarrier amplitude\n");
    fprintf(stderr,"   -comp-pre <s>              Composite preemphasis scale\n");
    fprintf(stderr,"   -comp-cut <f>              Composite preemphasis freq\n");
    fprintf(stderr,"   -comp-catv                 CATV preset #1\n");
    fprintf(stderr,"   -comp-catv2                CATV preset #2\n");
    fprintf(stderr,"   -comp-catv3                CATV preset #3\n");
    fprintf(stderr,"   -comp-catv4                CATV preset #4\n");
    fprintf(stderr,"   -nocomp                    Disable composite emulation (passthrough)\n");
    fprintf(stderr,"   -vhs-svideo <0|1>          S-Video output mode\n");
    fprintf(stderr,"   -vhs-chroma-vblend <0|1>   Chroma vertical blend\n");
    fprintf(stderr,"   -vhs-head-switching <0|1>  Head switching emulation\n");
    fprintf(stderr,"   -chroma-dropout <x>        Chroma scanline dropouts (0..10000)\n");
    fprintf(stderr,"   -yc-recomb <n>             Y/C recombine passes\n");
    fprintf(stderr,"   -nocolor-subcarrier        Debug: no color decode after subcarrier\n");
    fprintf(stderr,"   -comp-phase <n>            Subcarrier phase per scanline (0,90,180,270)\n");
    fprintf(stderr,"   -comp-phase-offset <n>     Subcarrier phase offset\n");
    fprintf(stderr,"   -in-composite-lowpass <n>  Chroma lowpass on composite in\n");
    fprintf(stderr,"   -out-composite-lowpass <n> Chroma lowpass on composite out\n");
    fprintf(stderr,"   -out-composite-lowpass-lite <n> Lite chroma lowpass on composite out\n");
    fprintf(stderr,"\n");
    fprintf(stderr," Received video is scaled to %dx%d (NTSC) or 720x576 (PAL) for processing.\n", 720, 480);
    fprintf(stderr," Default: center-crop source to match target aspect ratio. Use -stretch to stretch.\n");
    fprintf(stderr," Press Ctrl+C to stop.\n");
}

// ─── Command line parser ────────────────────────────────────────────────────
static int parse_argv(int argc, char **argv) {
    const char *a;
    int i;

    for (i=1; i < argc;) {
        a = argv[i++];

        if (*a == '-') {
            do { a++; } while (*a == '-');

            if (!strcmp(a,"h") || !strcmp(a,"help")) {
                help(argv[0]);
                return 1;
            }
            else if (!strcmp(a,"receiver")) {
                a = argv[i++];
                if (a == NULL) return 1;
                spout_receiver_name = a;
            }
            else if (!strcmp(a,"sender")) {
                a = argv[i++];
                if (a == NULL) return 1;
                spout_sender_name = a;
            }
            else if (!strcmp(a,"fps")) {
                a = argv[i++];
                if (a == NULL) return 1;
                target_fps = atoi(a);
                if (target_fps < 1 || target_fps > 240) {
                    fprintf(stderr,"Invalid fps %d\n", target_fps);
                    return 1;
                }
            }
            else if (!strcmp(a,"gui")) {
#ifdef HAVE_QT
                enable_gui = true;
#else
                fprintf(stderr,"WARNING: -gui requested but Qt support not compiled in. Ignoring.\n");
#endif
            }
            else if (!strcmp(a,"stretch")) {
                stretch_mode = true;
            }
            else if (!strcmp(a,"priority")) {
                a = argv[i++];
                if (a == NULL) return 1;
                if (!strcmp(a,"low")) {
                    gpu_priority_level = 0;
                } else if (!strcmp(a,"normal")) {
                    gpu_priority_level = 1;
                } else if (!strcmp(a,"high")) {
                    gpu_priority_level = 2;
                } else {
                    fprintf(stderr,"Unknown priority '%s' (use low, normal, or high)\n", a);
                    return 1;
                }
            }
            else if (!strcmp(a,"comp-phase-offset")) {
                video_scanline_phase_shift_offset = atoi(argv[i++]);
            }
            else if (!strcmp(a,"comp-phase")) {
                video_scanline_phase_shift = atoi(argv[i++]);
                if (!(video_scanline_phase_shift == 0 || video_scanline_phase_shift == 90 ||
                    video_scanline_phase_shift == 180 || video_scanline_phase_shift == 270)) {
                    fprintf(stderr,"Invalid phase\n");
                    return 1;
                }
            }
            else if (!strcmp(a,"width")) {
                a = argv[i++];
                if (a == NULL) return 1;
                output_width = (int)strtoul(a,NULL,0);
                if (output_width < 32) return 1;
            }
            else if (!strcmp(a,"tvstd")) {
                a = argv[i++];
                if (!strcmp(a,"pal")) {
                    preset_PAL();
                }
                else if (!strcmp(a,"ntsc")) {
                    preset_NTSC();
                }
                else {
                    fprintf(stderr,"Unknown tv std '%s'\n",a);
                    return 1;
                }
            }
            else if (!strcmp(a,"in-composite-lowpass")) {
                composite_in_chroma_lowpass = atoi(argv[i++]) > 0;
            }
            else if (!strcmp(a,"out-composite-lowpass")) {
                composite_out_chroma_lowpass = atoi(argv[i++]) > 0;
            }
            else if (!strcmp(a,"out-composite-lowpass-lite")) {
                composite_out_chroma_lowpass_lite = atoi(argv[i++]) > 0;
            }
            else if (!strcmp(a,"nocomp")) {
                enable_composite_emulation = false;
            }
            else if (!strcmp(a,"vhs-head-switching-point")) {
                vhs_head_switching_point = atof(argv[i++]);
            }
            else if (!strcmp(a,"vhs-head-switching-phase")) {
                vhs_head_switching_phase = atof(argv[i++]);
            }
            else if (!strcmp(a,"vhs-head-switching-noise-level")) {
                vhs_head_switching_phase_noise = atof(argv[i++]);
            }
            else if (!strcmp(a,"vhs-head-switching")) {
                int x = atoi(argv[i++]);
                vhs_head_switching = (x > 0)?true:false;
            }
            else if (!strcmp(a,"comp-pre")) {
                composite_preemphasis = atof(argv[i++]);
            }
            else if (!strcmp(a,"comp-cut")) {
                composite_preemphasis_cut = atof(argv[i++]);
            }
            else if (!strcmp(a,"comp-catv")) {
                composite_preemphasis = 7;
                composite_preemphasis_cut = 315000000 / 88;
                video_chroma_phase_noise = 2;
            }
            else if (!strcmp(a,"comp-catv2")) {
                composite_preemphasis = 15;
                composite_preemphasis_cut = 315000000 / 88;
                video_chroma_phase_noise = 4;
            }
            else if (!strcmp(a,"comp-catv3")) {
                composite_preemphasis = 25;
                composite_preemphasis_cut = (315000000 * 2) / 88;
                video_chroma_phase_noise = 6;
            }
            else if (!strcmp(a,"comp-catv4")) {
                composite_preemphasis = 40;
                composite_preemphasis_cut = (315000000 * 4) / 88;
                video_chroma_phase_noise = 6;
            }
            else if (!strcmp(a,"chroma-phase-noise")) {
                video_chroma_phase_noise = atoi(argv[i++]);
            }
            else if (!strcmp(a,"yc-recomb")) {
                video_yc_recombine = (int)atof(argv[i++]);
            }
            else if (!strcmp(a,"vhs-svideo")) {
                vhs_svideo_out = (atoi(argv[i++]) > 0);
            }
            else if (!strcmp(a,"vhs-chroma-vblend")) {
                vhs_chroma_vert_blend = (atoi(argv[i++]) > 0);
            }
            else if (!strcmp(a,"chroma-noise")) {
                video_chroma_noise = atoi(argv[i++]);
            }
            else if (!strcmp(a,"noise")) {
                video_noise = atoi(argv[i++]);
            }
            else if (!strcmp(a,"subcarrier-amp")) {
                int x = atoi(argv[i++]);
                subcarrier_amplitude = x;
                subcarrier_amplitude_back = x;
            }
            else if (!strcmp(a,"nocolor-subcarrier")) {
                nocolor_subcarrier = true;
            }
            else if (!strcmp(a,"nocolor-subcarrier-after-yc-sep")) {
                nocolor_subcarrier_after_yc_sep = true;
            }
            else if (!strcmp(a,"chroma-dropout")) {
                video_chroma_loss = atoi(argv[i++]);
            }
            else if (!strcmp(a,"vhs")) {
                emulating_vhs = true;
                vhs_head_switching = true;
                video_chroma_phase_noise = 4;
                video_chroma_noise = 16;
                video_chroma_loss = 4;
                video_noise = 4;
            }
            else if (!strcmp(a,"vhs-speed")) {
                a = argv[i++];
                emulating_vhs = true;
                if (!strcmp(a,"ep")) {
                    output_vhs_tape_speed = VHS_EP;
                    video_chroma_phase_noise = 6;
                    video_chroma_noise = 22;
                    video_chroma_loss = 8;
                    video_noise = 6;
                }
                else if (!strcmp(a,"lp")) {
                    output_vhs_tape_speed = VHS_LP;
                    video_chroma_phase_noise = 5;
                    video_chroma_noise = 19;
                    video_chroma_loss = 6;
                    video_noise = 5;
                }
                else if (!strcmp(a,"sp")) {
                    output_vhs_tape_speed = VHS_SP;
                    video_chroma_phase_noise = 4;
                    video_chroma_noise = 16;
                    video_chroma_loss = 4;
                    video_noise = 4;
                }
                else {
                    fprintf(stderr,"Unknown vhs tape speed '%s'\n",a);
                    return 1;
                }
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

    if (composite_preemphasis != 0)
        subcarrier_amplitude_back += (int)((50 * composite_preemphasis * (315000000.0 / 88)) / (2 * composite_preemphasis_cut));

    return 0;
}

// ─── NTSC composite video processing functions ──────────────────────────────
// These are copied from ffmpeg_ntsc.cpp to be self-contained.
// They operate on int arrays (YIQ planes) derived from BGRA AVFrames.

static void composite_lowpass_tv(AVFrame *dstframe, int *fY, int *fI, int *fQ, unsigned long long fieldno) {
    unsigned int x, y;
    for (unsigned int p=1; p <= 2; p++) {
        for (y=0; y < (unsigned int)dstframe->height; y++) {
            int *P = ((p == 1) ? fI : fQ) + (dstframe->width * y);
            LowpassFilter lp[3];
            double cutoff = 2600000;
            int delay = 1;
            for (unsigned int f=0; f < 3; f++) {
                lp[f].setFilter((315000000.00 * 4) / 88, cutoff);
                lp[f].resetFilter(0);
            }
            for (x=0; x < (unsigned int)dstframe->width; x++) {
                double s = P[x];
                for (unsigned int f=0; f < 3; f++) s = lp[f].lowpass(s);
                if (x >= (unsigned int)delay) P[x-delay] = (int)s;
            }
        }
    }
}

static void composite_lowpass(AVFrame *dstframe, int *fY, int *fI, int *fQ, unsigned long long fieldno) {
    unsigned int x, y;
    for (unsigned int p=1; p <= 2; p++) {
        for (y=0; y < (unsigned int)dstframe->height; y++) {
            int *P = ((p == 1) ? fI : fQ) + (dstframe->width * y);
            LowpassFilter lp[3];
            double cutoff = (p == 1) ? 1300000 : 600000;
            int delay = (p == 1) ? 2 : 4;
            for (unsigned int f=0; f < 3; f++) {
                lp[f].setFilter((315000000.00 * 4) / 88, cutoff);
                lp[f].resetFilter(0);
            }
            for (x=0; x < (unsigned int)dstframe->width; x++) {
                double s = P[x];
                for (unsigned int f=0; f < 3; f++) s = lp[f].lowpass(s);
                if (x >= (unsigned int)delay) P[x-delay] = (int)s;
            }
        }
    }
}

static void chroma_into_luma(AVFrame *dstframe, int *fY, int *fI, int *fQ, unsigned long long fieldno, int sub_amp) {
    unsigned int x, y;
    for (y=0; y < (unsigned int)dstframe->height; y++) {
        static const int8_t Umult[4] = { 1, 0,-1, 0 };
        static const int8_t Vmult[4] = { 0, 1, 0,-1 };
        int *Y = fY + (y * dstframe->width);
        int *I = fI + (y * dstframe->width);
        int *Q = fQ + (y * dstframe->width);
        unsigned int xi;

        if (video_scanline_phase_shift == 90)
            xi = (unsigned int)((fieldno + video_scanline_phase_shift_offset + (y >> 1)) & 3);
        else if (video_scanline_phase_shift == 180)
            xi = (unsigned int)((((fieldno + y) & 2) + video_scanline_phase_shift_offset) & 3);
        else if (video_scanline_phase_shift == 270)
            xi = (unsigned int)((fieldno + video_scanline_phase_shift_offset - (y >> 1)) & 3);
        else
            xi = (unsigned int)(video_scanline_phase_shift_offset & 3);

        for (x=0; x < (unsigned int)dstframe->width; x++) {
            unsigned int sxi = xi+x;
            int chroma;
            chroma  = (int)I[x] * sub_amp * Umult[sxi&3];
            chroma += (int)Q[x] * sub_amp * Vmult[sxi&3];
            Y[x] += (chroma / 50);
            I[x] = 0;
            Q[x] = 0;
        }
    }
}

static void chroma_from_luma(AVFrame *dstframe, int *fY, int *fI, int *fQ, unsigned long long fieldno, int sub_amp) {
    unsigned int x, y;
    for (y=0; y < (unsigned int)dstframe->height; y++) {
        int *Y = fY + (y * dstframe->width);
        int *I = fI + (y * dstframe->width);
        int *Q = fQ + (y * dstframe->width);
        std::vector<int> chroma(dstframe->width);
        int delay[4] = {0,0,0,0};
        int sum = 0;
        int c;

        delay[2] = Y[0]; sum += delay[2];
        delay[3] = Y[1]; sum += delay[3];
        for (x=0; x < (unsigned int)dstframe->width; x++) {
            if ((x+2) < (unsigned int)dstframe->width)
                c = Y[x+2];
            else
                c = 0;
            sum -= delay[0];
            for (unsigned int j=0; j < 3; j++) delay[j] = delay[j+1];
            delay[3] = c;
            sum += delay[3];
            Y[x] = sum / 4;
            chroma[x] = c - Y[x];
        }

        {
            unsigned int xi = 0;
            if (video_scanline_phase_shift == 90)
                xi = (unsigned int)((fieldno + video_scanline_phase_shift_offset + (y >> 1)) & 3);
            else if (video_scanline_phase_shift == 180)
                xi = (unsigned int)((((fieldno + y) & 2) + video_scanline_phase_shift_offset) & 3);
            else if (video_scanline_phase_shift == 270)
                xi = (unsigned int)((fieldno + video_scanline_phase_shift_offset - (y >> 1)) & 3);
            else
                xi = (unsigned int)(video_scanline_phase_shift_offset & 3);

            for (x=((4-xi)&3); (x+3) < (unsigned int)dstframe->width; x += 4) {
                chroma[x+2] = -chroma[x+2];
                chroma[x+3] = -chroma[x+3];
            }
            for (x=0; x < (unsigned int)dstframe->width; x++) {
                chroma[x] = ((int)chroma[x] * 50) / sub_amp;
            }
            for (x=0; (x+xi+1) < (unsigned int)dstframe->width; x += 2) {
                I[x] = -chroma[x+xi+0];
                Q[x] = -chroma[x+xi+1];
            }
            for (; x < (unsigned int)dstframe->width; x += 2) {
                I[x] = 0;
                Q[x] = 0;
            }
            for (x=0; (x+2) < (unsigned int)dstframe->width; x += 2) {
                I[x+1] = (I[x] + I[x+2]) >> 1;
                Q[x+1] = (Q[x] + Q[x+2]) >> 1;
            }
            for (; x < (unsigned int)dstframe->width; x++) {
                I[x] = 0;
                Q[x] = 0;
            }
        }
    }
}

// Main composite processing function — applies NTSC signal simulation to one frame
static void composite_layer(AVFrame *dstframe, AVFrame *srcframe, unsigned long long fieldno) {
    uint32_t *dscan, *sscan;
    unsigned int x, y;
    int r, g, b;

    if (dstframe == NULL || srcframe == NULL) return;
    if (dstframe->data[0] == NULL || srcframe->data[0] == NULL) return;
    if (dstframe->linesize[0] < (dstframe->width*4)) return;
    if (srcframe->linesize[0] < (srcframe->width*4)) return;
    if (dstframe->width != srcframe->width) return;
    if (dstframe->height != srcframe->height) return;

    // Pre-allocated YIQ buffers — avoid per-frame heap allocation
    static int *fY = nullptr, *fI = nullptr, *fQ = nullptr;
    static int yiq_buf_size = 0;
    int needed = dstframe->width * dstframe->height;
    if (needed > yiq_buf_size) {
        delete[] fY; delete[] fI; delete[] fQ;
        fY = new int[needed];
        fI = new int[needed];
        fQ = new int[needed];
        yiq_buf_size = needed;
    }

    memset(fY, 0, needed * sizeof(int));
    memset(fI, 0, needed * sizeof(int));
    memset(fQ, 0, needed * sizeof(int));

    for (y=0; y < (unsigned int)dstframe->height; y++) {
        sscan = (uint32_t*)(srcframe->data[0] + (srcframe->linesize[0] * y));
        for (x=0; x < (unsigned int)dstframe->width; x++) {
            r  = (sscan[x] >> 16UL) & 0xFF;
            g  = (sscan[x] >>  8UL) & 0xFF;
            b  = (sscan[x] >>  0UL) & 0xFF;
            RGB_to_YIQ(fY[(y*dstframe->width)+x], fI[(y*dstframe->width)+x], fQ[(y*dstframe->width)+x], r, g, b);
        }
    }

    if (composite_in_chroma_lowpass)
        composite_lowpass(dstframe, fY, fI, fQ, fieldno);

    chroma_into_luma(dstframe, fY, fI, fQ, fieldno, subcarrier_amplitude);

    /* composite preemphasis */
    if (composite_preemphasis != 0 && composite_preemphasis_cut > 0) {
        for (y=0; y < (unsigned int)dstframe->height; y++) {
            int *Y = fY + (y * dstframe->width);
            LowpassFilter pre;
            double s;
            pre.setFilter((315000000.00 * 4) / 88, composite_preemphasis_cut);
            pre.resetFilter(16);
            for (x=0; x < (unsigned int)dstframe->width; x++) {
                s = Y[x];
                s += pre.highpass(s) * composite_preemphasis;
                Y[x] = (int)s;
            }
        }
    }

    /* video noise */
    if (video_noise != 0) {
        int noise = 0, noise_mod = (video_noise*2)+1;
        for (y=0; y < (unsigned int)dstframe->height; y++) {
            int *Y = fY + (y * dstframe->width);
            for (x=0; x < (unsigned int)dstframe->width; x++) {
                Y[x] += noise;
                noise += ((int)((unsigned int)rand() % noise_mod)) - video_noise;
                noise /= 2;
            }
        }
    }

    if (!nocolor_subcarrier)
        chroma_from_luma(dstframe, fY, fI, fQ, fieldno, subcarrier_amplitude_back);

    /* chroma noise */
    if (video_chroma_noise != 0) {
        int noiseU = 0, noiseV = 0;
        for (y=0; y < (unsigned int)dstframe->height; y++) {
            int *U = fI + (y * dstframe->width);
            int *V = fQ + (y * dstframe->width);
            for (x=0; x < (unsigned int)dstframe->width; x++) {
                U[x] += noiseU;
                V[x] += noiseV;
                noiseU += ((int)((unsigned int)rand() % ((video_chroma_noise*2)+1))) - video_chroma_noise;
                noiseU /= 2;
                noiseV += ((int)((unsigned int)rand() % ((video_chroma_noise*2)+1))) - video_chroma_noise;
                noiseV /= 2;
            }
        }
    }

    if (video_chroma_phase_noise != 0) {
        int noise_val = 0;
        for (y=0; y < (unsigned int)dstframe->height; y++) {
            int *U = fI + (y * dstframe->width);
            int *V = fQ + (y * dstframe->width);
            noise_val += ((int)((unsigned int)rand() % ((video_chroma_phase_noise*2)+1))) - video_chroma_phase_noise;
            noise_val /= 2;
            double pi = ((double)noise_val * M_PI) / 100;
            double sinpi = sin(pi);
            double cospi = cos(pi);
            for (x=0; x < (unsigned int)dstframe->width; x++) {
                double u = U[x];
                double v = V[x];
                double u_ = (u * cospi) - (v * sinpi);
                double v_ = (u * sinpi) + (v * cospi);
                U[x] = (int)u_;
                V[x] = (int)v_;
            }
        }
    }

    // VHS head switching noise (moved after chroma_from_luma so I/Q zeroing is not
    // overwritten by subcarrier decode; runs before VHS emulation so re-encode sees
    // zero chroma in the affected region)
    if (vhs_head_switching) {
        unsigned int twidth = dstframe->width + (dstframe->width / 10);
        unsigned int tx, p, x2, shy=0;
        double noise_val = 0;
        int shif, ishif;
        int iy;
        double t;

        if (vhs_head_switching_phase_noise != 0) {
            unsigned int rx = (unsigned int)rand() * (unsigned int)rand() * (unsigned int)rand() * (unsigned int)rand();
            rx %= 2000000000U;
            noise_val = ((double)rx / 1000000000U) - 1.0;
            noise_val *= vhs_head_switching_phase_noise;
        }

        if (output_ntsc)
            t = twidth * 262.5;
        else
            t = twidth * 312.5;

        p = (unsigned int)(fmod(vhs_head_switching_point + noise_val, 1.0) * t);
        iy = (p / (unsigned int)twidth) * 2;

        p = (unsigned int)(fmod(vhs_head_switching_phase + noise_val, 1.0) * t);
        x = p % (unsigned int)twidth;

        if (output_ntsc)
            iy -= (262 - 240) * 2;
        else
            iy -= (312 - 288) * 2;

        tx = x;
        if (x >= (twidth/2))
            ishif = x - twidth;
        else
            ishif = x;

        shif = 0;
        while (iy < dstframe->height) {
            if (iy >= 0) {
                int *Y = fY + (iy * dstframe->width);

                // Zero I/Q (chroma) for all head-switching-affected scanlines.
                // Real VHS loses color lock in this region, producing monochrome output.
                int *I = fI + (iy * dstframe->width);
                int *Q = fQ + (iy * dstframe->width);
                memset(I, 0, dstframe->width * sizeof(int));
                memset(Q, 0, dstframe->width * sizeof(int));

                if (shif != 0) {
                    std::vector<int> tmp(twidth, 0);
                    memcpy(tmp.data(), Y, dstframe->width * sizeof(int));
                    x2 = (tx + twidth + (unsigned int)shif) % (unsigned int)twidth;
                    for (x=tx; x < (unsigned int)dstframe->width; x++) {
                        Y[x] = tmp[x2];
                        if ((++x2) == twidth) x2 = 0;
                    }
                }
            }
            if (shy == 0)
                shif = ishif;
            else
                shif = (shif * 7) / 8;
            tx = 0;
            iy += 1;
            shy++;
        }
    }

    // VHS emulation
    if (emulating_vhs) {
        double luma_cut, chroma_cut;
        int chroma_delay;
        switch (output_vhs_tape_speed) {
            case VHS_SP:  luma_cut = 2400000; chroma_cut = 320000; chroma_delay = 9; break;
            case VHS_LP:  luma_cut = 1900000; chroma_cut = 300000; chroma_delay = 12; break;
            case VHS_EP:  luma_cut = 1400000; chroma_cut = 280000; chroma_delay = 14; break;
            default: abort();
        }

        // luma lowpass
        for (y=0; y < (unsigned int)dstframe->height; y++) {
            int *Y = fY + (y * dstframe->width);
            LowpassFilter lp[3], pre;
            double s;
            for (unsigned int f=0; f < 3; f++) {
                lp[f].setFilter((315000000.00 * 4) / 88, luma_cut);
                lp[f].resetFilter(16);
            }
            pre.setFilter((315000000.00 * 4) / 88, luma_cut);
            pre.resetFilter(16);
            for (x=0; x < (unsigned int)dstframe->width; x++) {
                s = Y[x];
                for (unsigned int f=0; f < 3; f++) s = lp[f].lowpass(s);
                s += pre.highpass(s) * 1.6;
                Y[x] = (int)s;
            }
        }

        // chroma lowpass
        for (y=0; y < (unsigned int)dstframe->height; y++) {
            int *U = fI + (y * dstframe->width);
            int *V = fQ + (y * dstframe->width);
            LowpassFilter lpU[3], lpV[3];
            double s;
            for (unsigned int f=0; f < 3; f++) {
                lpU[f].setFilter((315000000.00 * 4) / 88, chroma_cut);
                lpU[f].resetFilter(0);
                lpV[f].setFilter((315000000.00 * 4) / 88, chroma_cut);
                lpV[f].resetFilter(0);
            }
            for (x=0; x < (unsigned int)dstframe->width; x++) {
                s = U[x];
                for (unsigned int f=0; f < 3; f++) s = lpU[f].lowpass(s);
                if (x >= (unsigned int)chroma_delay) U[x-chroma_delay] = (int)s;
                s = V[x];
                for (unsigned int f=0; f < 3; f++) s = lpV[f].lowpass(s);
                if (x >= (unsigned int)chroma_delay) V[x-chroma_delay] = (int)s;
            }
        }

        // VHS chroma vertical blend
        if (vhs_chroma_vert_blend && output_ntsc) {
            std::vector<int> delayU(dstframe->width, 0);
            std::vector<int> delayV(dstframe->width, 0);
            for (y=1; y < (unsigned int)dstframe->height; y++) {
                int *U = fI + (y * dstframe->width);
                int *V = fQ + (y * dstframe->width);
                for (x=0; x < (unsigned int)dstframe->width; x++) {
                    int cU = U[x], cV = V[x];
                    U[x] = (delayU[x]+cU+1)>>1;
                    V[x] = (delayV[x]+cV+1)>>1;
                    delayU[x] = cU;
                    delayV[x] = cV;
                }
            }
        }

        // VHS playback sharpening
        for (y=0; y < (unsigned int)dstframe->height; y++) {
            int *Y = fY + (y * dstframe->width);
            LowpassFilter lp[3];
            double s, ts;
            for (unsigned int f=0; f < 3; f++) {
                lp[f].setFilter((315000000.00 * 4) / 88, luma_cut*4);
                lp[f].resetFilter(0);
            }
            for (x=0; x < (unsigned int)dstframe->width; x++) {
                s = ts = Y[x];
                for (unsigned int f=0; f < 3; f++) ts = lp[f].lowpass(ts);
                Y[x] = (int)(s + ((s - ts) * vhs_out_sharpen * 2));
            }
        }

        if (!vhs_svideo_out) {
            chroma_into_luma(dstframe, fY, fI, fQ, fieldno, subcarrier_amplitude);
            chroma_from_luma(dstframe, fY, fI, fQ, fieldno, subcarrier_amplitude);
        }
    }

    // chroma dropout
    if (video_chroma_loss != 0) {
        for (y=0; y < (unsigned int)dstframe->height; y++) {
            int *U = fI + (y * dstframe->width);
            int *V = fQ + (y * dstframe->width);
            if ((((unsigned int)rand())%100000) < (unsigned int)video_chroma_loss) {
                memset(U, 0, dstframe->width*sizeof(int));
                memset(V, 0, dstframe->width*sizeof(int));
            }
        }
    }

    if (composite_out_chroma_lowpass) {
        if (composite_out_chroma_lowpass_lite)
            composite_lowpass_tv(dstframe, fY, fI, fQ, fieldno);
        else
            composite_lowpass(dstframe, fY, fI, fQ, fieldno);
    }

    // Convert back to BGRA
    for (y=0; y < (unsigned int)dstframe->height; y++) {
        dscan = (uint32_t*)(dstframe->data[0] + (dstframe->linesize[0] * y));
        for (x=0; x < (unsigned int)dstframe->width; x++) {
            YIQ_to_RGB(r, g, b, fY[(y*dstframe->width)+x], fI[(y*dstframe->width)+x], fQ[(y*dstframe->width)+x]);
            dscan[x] = (0xFFU << 24) | ((unsigned int)r << 16) | ((unsigned int)g << 8) | (unsigned int)b;
        }
    }

    // YIQ buffers are static — no per-frame deallocation needed
}

// ─── GUI frame update helper ────────────────────────────────────────────────
static void gui_update_frame(AVFrame *frame) {
#ifdef HAVE_QT
    if (!enable_gui) return;
    std::lock_guard<std::mutex> lock(g_gui_frame_mutex);
    // Convert BGRA AVFrame to QImage
    // BGRA in memory (B,G,R,A bytes) = 0xAARRGGBB as uint32 LE = QImage::Format_RGB32
    // Use Format_RGB32 (not Format_ARGB32) because the CUDA path may leave alpha=0
    if (g_gui_frame_image.width() != frame->width || g_gui_frame_image.height() != frame->height) {
        g_gui_frame_image = QImage(frame->width, frame->height, QImage::Format_RGB32);
    }
    for (int y = 0; y < frame->height; y++) {
        memcpy(g_gui_frame_image.scanLine(y),
               frame->data[0] + frame->linesize[0] * y,
               frame->width * 4);
    }
    g_gui_frame_dirty = true;
#else
    (void)frame;
#endif
}

// ─── Processing thread (runs the Spout receive → process → send loop) ───────
static std::atomic<bool> g_processing_running{false};

static void processing_thread_func() {
    spoutDX spoutIn;
    spoutDX spoutOut;

    // Initialize DirectX
    if (!spoutIn.OpenDirectX11()) {
        fprintf(stderr, "ERROR: Failed to open DirectX 11 for receiver\n");
        g_processing_running = false;
        return;
    }
    // Share the DX11 device between sender and receiver
    spoutOut.OpenDirectX11(spoutIn.GetDX11Device());

    // Configure receiver
    if (!spout_receiver_name.empty()) {
        spoutIn.SetReceiverName(spout_receiver_name.c_str());
        fprintf(stderr, "Spout receiver: connecting to '%s'\n", spout_receiver_name.c_str());
    } else {
        fprintf(stderr, "Spout receiver: connecting to active sender\n");
    }

    // Configure sender
    spoutOut.SetSenderName(spout_sender_name.c_str());
    fprintf(stderr, "Spout sender: broadcasting as '%s'\n", spout_sender_name.c_str());

    // Enable Spout2 frame counting so receivers can distinguish new frames.
    // Without this, SetNewFrame() inside SendImage is a no-op and receivers
    // poll the shared texture blindly, causing 60fps sends to appear as 30fps.
    bool wasFrameCountEnabled = spoutOut.frame.IsFrameCountEnabled();
    spoutOut.frame.SetFrameCount(true);
    if (!wasFrameCountEnabled) {
        fprintf(stderr, "Spout frame counting: was DISABLED, now ENABLED (registry updated)\n");
    } else {
        fprintf(stderr, "Spout frame counting: already enabled\n");
    }

    fprintf(stderr, "Processing resolution: %dx%d (%s)\n", output_width, output_height, output_ntsc ? "NTSC" : "PAL");
    fprintf(stderr, "Target FPS: %d\n", target_fps);

#ifdef HAVE_CUDA
    cuda_available = ntsc_cuda_init(output_width, output_height, gpu_priority_level);
    if (cuda_available) {
        fprintf(stderr, "NTSC CUDA acceleration enabled\n");
        for (int i = 0; i < NTSC_CUDA_NUM_BUFFERS; i++) {
            gpu_pipeline.gpu_output_frame[i] = av_frame_alloc();
            if (gpu_pipeline.gpu_output_frame[i] == NULL) {
                fprintf(stderr, "Failed to alloc GPU output frame %d\n", i);
                cuda_available = false;
                break;
            }
            gpu_pipeline.gpu_output_frame[i]->format = AV_PIX_FMT_BGRA;
            gpu_pipeline.gpu_output_frame[i]->height = output_height;
            gpu_pipeline.gpu_output_frame[i]->width = output_width;
            if (av_frame_get_buffer(gpu_pipeline.gpu_output_frame[i], 64) < 0) {
                cuda_available = false;
                break;
            }
            memset(gpu_pipeline.gpu_output_frame[i]->data[0], 0,
                   gpu_pipeline.gpu_output_frame[i]->linesize[0] * gpu_pipeline.gpu_output_frame[i]->height);
        }
        gpu_pipeline.current_buf_idx = 0;
        gpu_pipeline.has_pending_frame = false;
    } else {
        fprintf(stderr, "NTSC CUDA init failed, falling back to CPU\n");
    }
#endif

    // Set Windows thread and process priority based on -priority level
    {
        const char *priority_names[] = {"low", "normal", "high"};
        fprintf(stderr, "Priority level: %s\n", priority_names[gpu_priority_level]);

        if (gpu_priority_level <= 0) {
            // Low: reduce CPU/GPU contention
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
            SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
        } else if (gpu_priority_level >= 2) {
            // High: maximum performance
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
            SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
        } else {
            // Normal: default OS scheduling
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
            SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
        }
    }

    // Allocate processing frames
    AVFrame *srcFrame = av_frame_alloc();
    AVFrame *dstFrame = av_frame_alloc();
    AVFrame *prevFrame = av_frame_alloc();  // previous frame for temporal interlace
    if (!srcFrame || !dstFrame || !prevFrame) {
        fprintf(stderr, "ERROR: Failed to allocate AVFrames\n");
        g_processing_running = false;
        return;
    }
    srcFrame->format = AV_PIX_FMT_BGRA;
    srcFrame->width = output_width;
    srcFrame->height = output_height;
    av_frame_get_buffer(srcFrame, 64);
    memset(srcFrame->data[0], 0, srcFrame->linesize[0] * srcFrame->height);

    dstFrame->format = AV_PIX_FMT_BGRA;
    dstFrame->width = output_width;
    dstFrame->height = output_height;
    av_frame_get_buffer(dstFrame, 64);
    memset(dstFrame->data[0], 0, dstFrame->linesize[0] * dstFrame->height);

    prevFrame->format = AV_PIX_FMT_BGRA;
    prevFrame->width = output_width;
    prevFrame->height = output_height;
    av_frame_get_buffer(prevFrame, 64);
    memset(prevFrame->data[0], 0, prevFrame->linesize[0] * prevFrame->height);
    bool have_prev_frame = false;  // true after first frame received

    // Receive buffer (raw from Spout, may be any resolution)
    unsigned char *recvBuf = nullptr;
    unsigned int recvWidth = 0, recvHeight = 0;
    bool senderIsRGBA = false;  // true if sender uses DXGI_FORMAT_R8G8B8A8_UNORM

    // Scaler for resize (may include crop offsets)
    struct SwsContext *scaler = nullptr;
    int cropOffsetX = 0, cropOffsetY = 0;
    unsigned int cropWidth = 0, cropHeight = 0;  // dimensions to feed into sws_scale

    // Output send buffer (fixed resolution, BGRA)
    unsigned char *sendBuf = new unsigned char[output_width * output_height * 4];
    memset(sendBuf, 0, output_width * output_height * 4);

    unsigned long long fieldno = 0;
    unsigned long long frame_count = 0;
    auto fps_start = std::chrono::steady_clock::now();
    bool was_connected = false;   // track sender connection state
    bool signal_lost = false;     // true when sender disconnected — pause processing
    bool printed_waiting = false; // one-shot "Waiting for sender" message

    // ── Precision frame pacing timer ──
    // We bypass SpoutDX's HoldFps because it uses coarse Sleep and doesn't
    // guarantee the receiver has time to read the shared texture between writes.
    // steady_clock with spin-wait for the last 1.5ms achieves precise timing.
    const double frame_interval_us = 1000000.0 / (double)target_fps;  // microseconds per frame
    auto next_frame_time = std::chrono::steady_clock::now();
    timeBeginPeriod(1);  // request 1ms Windows timer resolution

    // Debug: track actual send intervals for diagnostics
    auto last_send_time = std::chrono::steady_clock::now();
    unsigned long long debug_send_count = 0;
    double debug_interval_sum_us = 0.0;

    // ── Simple 60fps progressive loop ──
    // Every iteration: ReceiveImage → composite_layer (full frame) → SendImage
    // No interlacing or field splitting — each frame is a complete progressive image.

    fprintf(stderr, "Processing loop started (target: %d fps, frame interval: %.1f us). Press Ctrl+C to stop.\n",
            target_fps, frame_interval_us);

    while (!DIE) {
        // ── Fetch new input from Spout (every iteration) ──
        {
            // ReceiveImage handles connection, sender detection, and pixel copy.
            // On first connection or sender change, it returns true with IsUpdated()=true
            // but does NOT copy pixels yet — we must allocate/resize our buffer first.
            // The width/height params are the DESTINATION buffer size (not output params).
            // bRGB=false means BGRA/RGBA (4 bytes/pixel), bInvert=false means no flip.
            bool received = spoutIn.ReceiveImage(recvBuf, recvWidth, recvHeight, false, false);

            // ── Signal loss detection ──
            if (was_connected && !received) {
                if (!signal_lost) {
                    signal_lost = true;
                    fprintf(stderr, "\n[SIGNAL LOST] Sender disconnected — switching to black frames.\n");

                    // Clear srcFrame and prevFrame to black so composite processes black
                    memset(srcFrame->data[0], 0, srcFrame->linesize[0] * srcFrame->height);
                    memset(prevFrame->data[0], 0, prevFrame->linesize[0] * prevFrame->height);
                    have_prev_frame = false;

                    // Also send an immediate black frame to clear the output
                    memset(sendBuf, 0, output_width * output_height * 4);
                    spoutOut.SendImage(sendBuf, output_width, output_height);

                    // Clear the GUI preview to black
                    if (dstFrame) {
                        for (int y = 0; y < dstFrame->height; y++)
                            memset(dstFrame->data[0] + dstFrame->linesize[0] * y, 0, output_width * 4);
                        gui_update_frame(dstFrame);
                    }

                    // ── Full receiver reset ──
                    spoutIn.ReleaseReceiver();

                    if (!spout_receiver_name.empty()) {
                        spoutIn.SetReceiverName(spout_receiver_name.c_str());
                    }

                    delete[] recvBuf;
                    recvBuf = nullptr;
                    recvWidth = 0;
                    recvHeight = 0;
                    senderIsRGBA = false;

                    if (scaler) {
                        sws_freeContext(scaler);
                        scaler = nullptr;
                    }
                    cropOffsetX = 0;
                    cropOffsetY = 0;
                    cropWidth = 0;
                    cropHeight = 0;

                    was_connected = false;
                    printed_waiting = false;
                }
            }

            if (!received) {
                // No connection — sleep to avoid busy-spinning.
                if (!was_connected && !printed_waiting) {
                    fprintf(stderr, "Waiting for Spout2 sender...\n");
                    printed_waiting = true;
                }
                Sleep(100);
                continue;
            }

            // Signal recovered
            if (signal_lost) {
                signal_lost = false;
                fprintf(stderr, "[SIGNAL RESTORED] Sender reconnected — resuming processing.\n");
                frame_count = 0;
                fps_start = std::chrono::steady_clock::now();
                next_frame_time = std::chrono::steady_clock::now();
                last_send_time = std::chrono::steady_clock::now();
            }
            was_connected = true;

            // Check if sender is new or changed
            if (spoutIn.IsUpdated()) {
                unsigned int newW = spoutIn.GetSenderWidth();
                unsigned int newH = spoutIn.GetSenderHeight();
                DWORD senderFormat = spoutIn.GetSenderFormat();
                fprintf(stderr, "Spout source connected/changed: %ux%u (DXGI format %lu)\n", newW, newH, senderFormat);

                senderIsRGBA = (senderFormat == 28);
                spoutIn.SetSwap(senderIsRGBA);
                if (senderIsRGBA) {
                    fprintf(stderr, "  Sender uses RGBA format — enabling R/B swap for BGRA normalization\n");
                }

                recvWidth = newW;
                recvHeight = newH;

                delete[] recvBuf;
                recvBuf = new unsigned char[recvWidth * recvHeight * 4];
                memset(recvBuf, 0, recvWidth * recvHeight * 4);

                if (scaler) {
                    sws_freeContext(scaler);
                    scaler = nullptr;
                }

                if (stretch_mode) {
                    cropOffsetX = 0;
                    cropOffsetY = 0;
                    cropWidth = recvWidth;
                    cropHeight = recvHeight;
                    fprintf(stderr, "  Mode: STRETCH %ux%u -> %dx%d\n", recvWidth, recvHeight, output_width, output_height);
                } else {
                    double srcAspect = (double)recvWidth / (double)recvHeight;
                    double dstAspect = (double)output_width / (double)output_height;

                    if (srcAspect > dstAspect) {
                        cropHeight = recvHeight;
                        cropWidth = (unsigned int)(dstAspect * recvHeight + 0.5);
                        if (cropWidth > recvWidth) cropWidth = recvWidth;
                        cropWidth &= ~1U;
                        cropOffsetX = (int)((recvWidth - cropWidth) / 2) & ~1;
                        cropOffsetY = 0;
                    } else if (srcAspect < dstAspect) {
                        cropWidth = recvWidth;
                        cropHeight = (unsigned int)((double)recvWidth / dstAspect + 0.5);
                        if (cropHeight > recvHeight) cropHeight = recvHeight;
                        cropHeight &= ~1U;
                        cropOffsetX = 0;
                        cropOffsetY = (int)((recvHeight - cropHeight) / 2) & ~1;
                    } else {
                        cropOffsetX = 0;
                        cropOffsetY = 0;
                        cropWidth = recvWidth;
                        cropHeight = recvHeight;
                    }
                    fprintf(stderr, "  Mode: CENTER-CROP %ux%u -> crop %ux%u (offset %d,%d) -> %dx%d\n",
                            recvWidth, recvHeight, cropWidth, cropHeight, cropOffsetX, cropOffsetY,
                            output_width, output_height);
                }

                next_frame_time = std::chrono::steady_clock::now();
                last_send_time = std::chrono::steady_clock::now();
                continue;
            }

            if (!recvBuf || recvWidth == 0 || recvHeight == 0) {
                Sleep(16);
                continue;
            }

            // Create scaler if needed
            if (scaler == nullptr) {
                if (cropWidth == 0 || cropHeight == 0) {
                    Sleep(16);
                    continue;
                }
                scaler = sws_getContext(
                    cropWidth, cropHeight, AV_PIX_FMT_BGRA,
                    output_width, output_height, AV_PIX_FMT_BGRA,
                    SWS_BILINEAR, NULL, NULL, NULL);
                if (scaler == nullptr) {
                    fprintf(stderr, "ERROR: Failed to create scaler %ux%u -> %dx%d\n",
                            cropWidth, cropHeight, output_width, output_height);
                    Sleep(100);
                    continue;
                }
            }

            // Scale received BGRA frame to processing resolution
            {
                const uint8_t *srcPtr = recvBuf + (cropOffsetY * recvWidth * 4) + (cropOffsetX * 4);
                const uint8_t *srcSlice[1] = { srcPtr };
                int srcStride[1] = { (int)(recvWidth * 4) };
                sws_scale(scaler, srcSlice, srcStride, 0, cropHeight,
                          srcFrame->data, srcFrame->linesize);
            }
        } // end receive block

        // ── Process full frame (progressive) ──
        // Both CPU and CUDA paths now process all scanlines in one call.

        if (enable_composite_emulation) {
            AVFrame *comp_src = have_prev_frame ? prevFrame : srcFrame;

#ifdef HAVE_CUDA
            if (cuda_available) {
                NtscCudaParams cp = {};
                cp.width = output_width;
                cp.height = output_height;
                cp.subcarrier_amplitude = subcarrier_amplitude;
                cp.subcarrier_amplitude_back = subcarrier_amplitude_back;
                cp.video_scanline_phase_shift = video_scanline_phase_shift;
                cp.video_scanline_phase_shift_offset = video_scanline_phase_shift_offset;
                cp.video_noise = video_noise;
                cp.video_chroma_noise = video_chroma_noise;
                cp.video_chroma_phase_noise = video_chroma_phase_noise;
                cp.video_chroma_loss = video_chroma_loss;
                cp.composite_preemphasis = composite_preemphasis;
                cp.composite_preemphasis_cut = composite_preemphasis_cut;
                cp.composite_in_chroma_lowpass = composite_in_chroma_lowpass;
                cp.composite_out_chroma_lowpass = composite_out_chroma_lowpass;
                cp.composite_out_chroma_lowpass_lite = composite_out_chroma_lowpass_lite;
                cp.nocolor_subcarrier = nocolor_subcarrier;
                cp.nocolor_subcarrier_after_yc_sep = nocolor_subcarrier_after_yc_sep;
                cp.enable_composite_emulation = enable_composite_emulation;
                cp.emulating_vhs = emulating_vhs;
                cp.output_vhs_tape_speed = output_vhs_tape_speed;
                cp.vhs_chroma_vert_blend = vhs_chroma_vert_blend;
                cp.vhs_svideo_out = vhs_svideo_out;
                cp.vhs_out_sharpen = vhs_out_sharpen;
                cp.output_ntsc = output_ntsc;
                cp.vhs_head_switching = vhs_head_switching;
                cp.vhs_head_switching_point = vhs_head_switching_point;
                cp.vhs_head_switching_phase = vhs_head_switching_phase;
                cp.vhs_head_switching_phase_noise = vhs_head_switching_phase_noise;
                cp.opposite = 0;
                cp.field = 0;
                cp.fieldno = fieldno;
                cp.progressive = true;  // spout_ntsc uses progressive (all scanlines per frame)

                ntsc_cuda_composite_layer(
                    comp_src->data[0], comp_src->linesize[0],
                    dstFrame->data[0], dstFrame->linesize[0],
                    cp);
                ntsc_cuda_sync();
            }
            else
#endif
            {
                composite_layer(dstFrame, comp_src, fieldno);
            }
        } else {
            // Passthrough: copy source to destination
            for (int y = 0; y < output_height; y++) {
                memcpy(dstFrame->data[0] + dstFrame->linesize[0] * y,
                       srcFrame->data[0] + srcFrame->linesize[0] * y,
                       output_width * 4);
            }
        }

        // ── Precision pacing: wait until next_frame_time ──
        {
            auto now = std::chrono::steady_clock::now();
            auto remaining_us = std::chrono::duration_cast<std::chrono::microseconds>(
                next_frame_time - now).count();

            if (remaining_us > 1500) {
                long long sleep_ms = (remaining_us - 1500) / 1000;
                if (sleep_ms > 0) {
                    if (gpu_priority_level <= 0) {
                        Sleep((DWORD)(sleep_ms + 1));
                    } else {
                        Sleep((DWORD)sleep_ms);
                    }
                }
            }

            // Spin-wait for precise timing
            while (std::chrono::steady_clock::now() < next_frame_time) {
                // spin
            }

            // Advance by one frame interval (additive to prevent drift)
            next_frame_time += std::chrono::microseconds((long long)frame_interval_us);

            // If we fell behind by more than 2 intervals, reset
            auto behind = std::chrono::steady_clock::now() - next_frame_time;
            if (behind > std::chrono::microseconds((long long)(frame_interval_us * 2))) {
                next_frame_time = std::chrono::steady_clock::now() +
                    std::chrono::microseconds((long long)frame_interval_us);
            }
        }

        // ── Send frame via Spout ──
        if (dstFrame->linesize[0] == output_width * 4) {
            // Stride matches — send directly from dstFrame, no copy needed
            spoutOut.SendImage(dstFrame->data[0], output_width, output_height);
        } else {
            // Stride mismatch — must copy to contiguous buffer
            for (int y = 0; y < output_height; y++) {
                memcpy(sendBuf + y * output_width * 4,
                       dstFrame->data[0] + dstFrame->linesize[0] * y,
                       output_width * 4);
            }
            spoutOut.SendImage(sendBuf, output_width, output_height);
        }

        // Debug: track actual send intervals
        {
            auto now = std::chrono::steady_clock::now();
            double interval_us = (double)std::chrono::duration_cast<std::chrono::microseconds>(
                now - last_send_time).count();
            last_send_time = now;
            debug_send_count++;
            debug_interval_sum_us += interval_us;
        }

        // Update GUI preview (every other frame to reduce overhead)
        if ((fieldno & 1) == 0) {
            gui_update_frame(dstFrame);
        }

        // FPS counter
        frame_count++;
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start).count();
            if (elapsed >= 2000) {
                double fps = (double)frame_count * 1000.0 / elapsed;
                double avg_interval_ms = (debug_send_count > 0)
                    ? (debug_interval_sum_us / debug_send_count / 1000.0) : 0.0;
                fprintf(stderr, "\x0D" "%.1f fps (%.1fms/frame) | frames %llu | %ux%u -> %s %ux%u -> %dx%d%s   ",
                        fps, avg_interval_ms, fieldno, recvWidth, recvHeight,
                        stretch_mode ? "stretch" : "crop",
                        cropWidth, cropHeight,
                        output_width, output_height,
                        senderIsRGBA ? " [RGBA]" : "");
                fflush(stderr);
                frame_count = 0;
                fps_start = now;
                debug_send_count = 0;
                debug_interval_sum_us = 0.0;
            }
        }

        // ── Advance frame state ──
        // Swap srcFrame and prevFrame pointers — avoids per-frame memcpy.
        // After swap: prevFrame points to this iteration's source data,
        // srcFrame points to the (now stale) buffer that will be overwritten next iteration.
        std::swap(srcFrame, prevFrame);
        have_prev_frame = true;
        fieldno++;  // one frame = one fieldno increment
    }

    fprintf(stderr, "\nShutting down...\n");

    // Restore Windows timer resolution
    timeEndPeriod(1);

    // Cleanup
#ifdef HAVE_CUDA
    if (cuda_available) {
        ntsc_cuda_shutdown();
        for (int i = 0; i < NTSC_CUDA_NUM_BUFFERS; i++) {
            if (gpu_pipeline.gpu_output_frame[i])
                av_frame_free(&gpu_pipeline.gpu_output_frame[i]);
        }
    }
#endif

    if (scaler) sws_freeContext(scaler);
    av_frame_free(&srcFrame);
    av_frame_free(&dstFrame);
    av_frame_free(&prevFrame);
    delete[] recvBuf;
    delete[] sendBuf;

    spoutIn.ReleaseReceiver();
    spoutOut.ReleaseSender();
    spoutIn.CloseDirectX11();

    g_processing_running = false;
}

// ─── Main ───────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    preset_NTSC();
    if (parse_argv(argc, argv))
        return 1;

    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);

    fprintf(stderr, "spout_ntsc — Real-time NTSC composite video simulation proxy\n");
    fprintf(stderr, "Standard: %s | Resolution: %dx%d | Composite: %s | VHS: %s | Mode: %s\n",
            output_ntsc ? "NTSC" : "PAL",
            output_width, output_height,
            enable_composite_emulation ? "ON" : "OFF",
            emulating_vhs ? "ON" : "OFF",
            stretch_mode ? "stretch" : "center-crop");

#ifdef HAVE_QT
    if (enable_gui) {
        // Qt GUI mode: run processing in a thread, Qt event loop on main thread
        QApplication app(argc, argv);

        PreviewWindow window;
        window.show();

        g_processing_running = true;
        std::thread proc_thread(processing_thread_func);

        // When processing finishes or user closes window, quit
        QTimer exitCheck;
        QObject::connect(&exitCheck, &QTimer::timeout, [&]() {
            if (!g_processing_running || DIE) {
                app.quit();
            }
        });
        exitCheck.start(100);

        int ret = app.exec();

        // Signal thread to stop
        DIE = 1;
        if (proc_thread.joinable())
            proc_thread.join();

        return ret;
    }
#endif

    // Non-GUI mode: run processing directly on main thread
    g_processing_running = true;
    processing_thread_func();

    return 0;
}

// Qt MOC include (needed when Q_OBJECT is in a .cpp file)
#ifdef HAVE_QT
#include "spout_ntsc.moc"
#endif
