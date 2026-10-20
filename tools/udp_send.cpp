// Streams a video file as MPEG-TS over UDP, like a live source, at most at the file's frame rate.
// At 1080p this CPU's decode and encode cap it near 34 fps (test.mp4 is 59.94), so it prints the
// rate it reached: quote that, not the file's.
// udp_send <file> <url> [seconds]
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

int main(int argc, char **argv)
{
    if (argc < 3)
        return std::fprintf(stderr, "usage: udp_send <file> <url> [seconds]\n"), 2;
    cv::VideoCapture in(argv[1]);
    const double fps = in.get(cv::CAP_PROP_FPS);
    const double seconds = argc > 3 ? std::stod(argv[3]) : 1e9;
    // The trailing "&f=.ts" only lets FFmpeg guess the MPEG-TS muxer from the "extension".
    cv::VideoWriter out(std::string(argv[2]) + "?pkt_size=1316&f=.ts", cv::CAP_FFMPEG,
                        cv::VideoWriter::fourcc('P', 'I', 'M', '1'), fps,
                        {(int)in.get(cv::CAP_PROP_FRAME_WIDTH), (int)in.get(cv::CAP_PROP_FRAME_HEIGHT)});
    if (!in.isOpened() || !out.isOpened())
        return std::fprintf(stderr, "open failed\n"), 1;
    const auto period = std::chrono::duration<double>(1.0 / fps);
    const auto start = std::chrono::steady_clock::now();
    auto next = start;
    cv::Mat f;
    for (int n = 0; n < fps * seconds; n++)
    {
        if (!in.read(f))
        {
            in.set(cv::CAP_PROP_POS_FRAMES, 0);
            in.read(f);
        }
        out.write(f);
        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        std::this_thread::sleep_until(next);
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr, "sent %d frames at %.2f fps (file: %.2f fps)\n", (int)(fps * seconds), fps * seconds / elapsed, fps);
}
