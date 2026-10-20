#include "encoder.hpp"
#include "bounded_queue.hpp"
#include "pool.hpp"
#include "timing.hpp"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace
{
using Clock = std::chrono::steady_clock;

constexpr double kGopSeconds = 2.0;
// Calibration knob: 0 lets each encoder pick (x264 uses about 1.5 threads per core). The CPU has 4
// threads and capture decodes on one of them: lower this if read_ms rises when --encode is on.
constexpr int kEncodeThreads = 0;

struct CodecSpec
{
    const char *name;
    const char *options;
    bool mpegts; // the muxer cannot say: it answers "unknown" for every codec
};

constexpr CodecSpec kCodecs[] = {
    {"libx264", "preset=ultrafast:tune=zerolatency", true},
    {"libx265", "preset=ultrafast:tune=zerolatency:x265-params=log-level=error", true},
    // b=0 with crf is constant quality; libvpx's default 256 kb/s target would smear 1080p.
    // lag-in-frames=0 is the VP9 analogue of no B-frames: no lookahead, no alt-ref.
    {"libvpx-vp9", "deadline=realtime:cpu-used=8:row-mt=1:lag-in-frames=0:crf=32:b=0", false},
};

const CodecSpec &find_codec(const std::string &name)
{
    for (const CodecSpec &c : kCodecs)
        if (name == c.name)
            return c;
    throw std::invalid_argument("unknown codec: " + name);
}

void check(int ret, const std::string &what)
{
    if (ret >= 0)
        return;
    char msg[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(ret, msg, sizeof msg);
    throw std::runtime_error("encoder: " + what + ": " + msg);
}

struct CodecFree
{
    void operator()(AVCodecContext *c) const { avcodec_free_context(&c); }
};
struct FrameFree
{
    void operator()(AVFrame *f) const { av_frame_free(&f); }
};
struct PacketFree
{
    void operator()(AVPacket *p) const { av_packet_free(&p); }
};
struct SwsFree
{
    void operator()(SwsContext *s) const { sws_freeContext(s); }
};
struct DictFree
{
    void operator()(AVDictionary *d) const { av_dict_free(&d); }
};
struct OutputFree
{
    void operator()(AVFormatContext *f) const
    {
        if (!(f->oformat->flags & AVFMT_NOFILE))
            avio_closep(&f->pb);
        avformat_free_context(f);
    }
};

// Frame ticks in the codec time base (1/fps), strictly increasing. A file uses id - 1: capture never
// skips an id for a file, so an encode drop leaves a gap and the duration stays right. A live source
// uses capture time: its capture drops get no id, so ids would play a 27 fps run of a 30 fps camera
// 10% fast.
// ponytail: a live source that delivers faster than its reported (or fallback 30) fps plays back slow,
// because each frame takes at least one 1/fps tick. Measure the rate from the first capture deltas to fix it.
class PtsClock
{
public:
    PtsClock(bool live, double fps) : live_(live), fps_(fps) {}

    int64_t next(int id, Clock::time_point captured)
    {
        int64_t pts = id - 1;
        if (live_)
        {
            if (last_ < 0)
                t0_ = captured;
            pts = std::llround(std::chrono::duration<double>(captured - t0_).count() * fps_);
        }
        return last_ = std::max(pts, last_ + 1);
    }

private:
    const bool live_;
    const double fps_;
    Clock::time_point t0_{};
    int64_t last_ = -1;
};

struct Job
{
    std::shared_ptr<cv::Mat> bgr;
    int id;
    Clock::time_point captured;
};
} // namespace

// Until the join only the encoder thread touches libav, pts, encoded and ms, and only the caller
// touches dropped; finish() reads them after the join.
struct Encoder::Impl
{
    Impl(const std::string &target, const std::string &codec, const SourceFormat &fmt, int capacity);

    void run();
    void encode(const Job &job);
    void drain();
    void close_output();
    void shutdown();

    std::unique_ptr<AVFormatContext, OutputFree> out;
    std::unique_ptr<AVCodecContext, CodecFree> ctx;
    AVStream *stream = nullptr; // owned by out
    std::unique_ptr<SwsContext, SwsFree> sws;
    std::unique_ptr<AVFrame, FrameFree> yuv;
    std::unique_ptr<AVPacket, PacketFree> pkt;
    bool closed = false;

    PtsClock pts;
    int encoded = 0;
    double ms = 0.0;
    int dropped = 0;

    EncodeStats stats;
    bool finished = false;

    std::exception_ptr error;
    std::atomic<bool> failed{false};

    Pool<cv::Mat> slots;     // before queue: the queue holds slot handles
    BoundedQueue<Job> queue; // as deep as slots, so a job holding a slot always fits
    std::jthread thread;
};

Encoder::Impl::Impl(const std::string &target, const std::string &codec, const SourceFormat &fmt, int capacity)
    : pts(fmt.live, fmt.fps), stats{codec}, slots(std::vector<cv::Mat>(std::max(capacity, 1))), queue(std::max(capacity, 1))
{
    const CodecSpec &spec = find_codec(codec);
    if (fmt.size.width < 2 || fmt.size.height < 2)
        throw std::invalid_argument("source reports no frame size, cannot encode");
    if (!(fmt.fps > 0.0) || capacity < 1)
        throw std::invalid_argument("encoder needs a positive frame rate and at least one slot");
    const AVCodec *c = avcodec_find_encoder_by_name(spec.name);
    if (!c)
        throw std::runtime_error(std::string(spec.name) + " is not in this FFmpeg build");

    // FFmpeg's default 1472-byte datagram splits TS packets; 7 x 188 keeps them whole.
    std::string url = target;
    if (url.starts_with("udp://") && url.find("pkt_size=") == std::string::npos)
        url += (url.find('?') == std::string::npos ? "?" : "&") + std::string("pkt_size=1316");
    AVFormatContext *raw = nullptr;
    check(avformat_alloc_output_context2(&raw, nullptr, target.starts_with("udp://") ? "mpegts" : nullptr, url.c_str()),
          "no container for " + target + " (use .mp4, .mkv, .webm or udp://)");
    out.reset(raw);
    // 0 means "cannot hold", negative means "unknown" (mpegts says so for every codec, VP9 included).
    const int holds = avformat_query_codec(out->oformat, c->id, FF_COMPLIANCE_NORMAL);
    if (holds == 0 || (holds < 0 && !spec.mpegts))
        throw std::runtime_error(std::string("encoder: ") + out->oformat->name + " cannot hold " + spec.name);

    ctx.reset(avcodec_alloc_context3(c));
    ctx->width = fmt.size.width & ~1; // 4:2:0 needs even sizes
    ctx->height = fmt.size.height & ~1;
    ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    ctx->framerate = av_d2q(fmt.fps, 100000); // 29.97 -> 30000/1001
    ctx->time_base = av_inv_q(ctx->framerate);
    ctx->gop_size = std::max<long>(1, std::lround(kGopSeconds * fmt.fps));
    ctx->max_b_frames = 0;
    ctx->thread_count = kEncodeThreads;
    // sws_scale's default matrix; players read the tags instead of guessing from the height.
    ctx->colorspace = AVCOL_SPC_SMPTE170M;
    ctx->color_range = AVCOL_RANGE_MPEG;
    if (out->oformat->flags & AVFMT_GLOBALHEADER)
        ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    AVDictionary *raw_opts = nullptr;
    const int parsed = av_dict_parse_string(&raw_opts, spec.options, "=", ":", 0);
    const int opened = parsed < 0 ? parsed : avcodec_open2(ctx.get(), c, &raw_opts);
    const std::unique_ptr<AVDictionary, DictFree> opts(raw_opts);
    check(opened, std::string("open ") + spec.name);
    if (const AVDictionaryEntry *left = av_dict_get(raw_opts, "", nullptr, AV_DICT_IGNORE_SUFFIX))
        throw std::runtime_error(std::string("encoder: ") + spec.name + " ignored option " + left->key);

    yuv.reset(av_frame_alloc());
    yuv->format = ctx->pix_fmt;
    yuv->width = ctx->width;
    yuv->height = ctx->height;
    check(av_frame_get_buffer(yuv.get(), 0), "frame buffer");
    pkt.reset(av_packet_alloc());

    stream = avformat_new_stream(out.get(), nullptr);
    check(stream ? 0 : AVERROR(ENOMEM), "new stream");
    check(avcodec_parameters_from_context(stream->codecpar, ctx.get()), "codec parameters");
    stream->time_base = ctx->time_base; // a hint: the muxer may change it, so rescale to the stream's after the header
    stream->avg_frame_rate = ctx->framerate;
    if (!(out->oformat->flags & AVFMT_NOFILE))
        check(avio_open(&out->pb, url.c_str(), AVIO_FLAG_WRITE), "open " + target);
    check(avformat_write_header(out.get(), nullptr), "write header to " + target);

    thread = std::jthread([this] { run(); });
}

void Encoder::Impl::run()
{
    try
    {
        cv::TickMeter tm;
        // A default stop_token never fires: only close() ends the loop, so nothing skips the trailer.
        while (auto job = queue.pop(std::stop_token{}))
            ms = timed(tm, ++encoded, "encode", [&] { encode(*job); });
        close_output();
    }
    catch (...)
    {
        error = std::current_exception();
        failed.store(true, std::memory_order_release);
    }
}

void Encoder::Impl::encode(const Job &job)
{
    const cv::Mat &bgr = *job.bgr;
    sws.reset(sws_getCachedContext(sws.release(), bgr.cols, bgr.rows, AV_PIX_FMT_BGR24, yuv->width, yuv->height,
                                   AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr, nullptr, nullptr));
    check(sws ? 0 : AVERROR(EINVAL), "scaler");
    check(av_frame_make_writable(yuv.get()), "frame writable"); // the codec may still hold the last one
    const uint8_t *src[] = {bgr.data};
    const int stride[] = {static_cast<int>(bgr.step)};
    sws_scale(sws.get(), src, stride, 0, bgr.rows, yuv->data, yuv->linesize);
    yuv->pts = pts.next(job.id, job.captured);
    check(avcodec_send_frame(ctx.get(), yuv.get()), "send frame");
    drain();
}

void Encoder::Impl::drain()
{
    int ret;
    while ((ret = avcodec_receive_packet(ctx.get(), pkt.get())) >= 0)
    {
        // Without a duration the last frame of an mp4 has none and the reader drops it.
        pkt->duration = 1;
        av_packet_rescale_ts(pkt.get(), ctx->time_base, stream->time_base);
        pkt->stream_index = stream->index;
        check(av_interleaved_write_frame(out.get(), pkt.get()), "write");
    }
    if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF)
        check(ret, "receive packet");
}

// Runs once: on the encoder thread after the last job, or in ~Encoder if that thread died first.
void Encoder::Impl::close_output()
{
    if (std::exchange(closed, true))
        return;
    check(avcodec_send_frame(ctx.get(), nullptr), "flush");
    drain();
    check(av_write_trailer(out.get()), "write trailer");
}

void Encoder::Impl::shutdown()
{
    queue.close();
    if (thread.joinable())
        thread.join();
}

const std::vector<std::string> &Encoder::codec_names()
{
    static const std::vector<std::string> names = [] {
        std::vector<std::string> v;
        for (const CodecSpec &c : kCodecs)
            v.emplace_back(c.name);
        return v;
    }();
    return names;
}

Encoder::Encoder(const std::string &target, const std::string &codec, const SourceFormat &fmt, int slots)
    : impl_(std::make_unique<Impl>(target, codec, fmt, slots))
{
}

Encoder::~Encoder()
{
    Impl &m = *impl_;
    m.shutdown();
    try
    {
        m.close_output();
    }
    catch (...)
    {
    }
}

void Encoder::submit(const cv::Mat &bgr, int id, Clock::time_point captured)
{
    Impl &m = *impl_;
    if (m.failed.load(std::memory_order_acquire))
        std::rethrow_exception(m.error);
    CV_Assert(bgr.type() == CV_8UC3);
    std::shared_ptr<cv::Mat> slot = m.slots.try_acquire();
    if (!slot)
    {
        ++m.dropped;
        return;
    }
    bgr.copyTo(*slot); // reuses the slot's buffer when the size matches
    if (!m.queue.try_push({std::move(slot), id, captured}))
        ++m.dropped;
}

EncodeStats Encoder::finish()
{
    Impl &m = *impl_;
    if (!std::exchange(m.finished, true))
    {
        m.shutdown();
        m.stats.frames = m.encoded;
        m.stats.dropped = m.dropped;
        m.stats.ms = m.ms;
    }
    if (m.error)
        std::rethrow_exception(m.error);
    return m.stats;
}
