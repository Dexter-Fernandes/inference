# Real-time YOLOv5 inference pipeline

A C++20 object-detection pipeline for video files, cameras, and RTSP or UDP streams. It runs YOLOv5s on TensorRT 8.5.3 on a GeForce 940MX. Each stage runs on its own thread, a fixed pool of frames is the only back-pressure point, and a live source processes its freshest frame at the default `--pool 3`. The pipeline can also encode the annotated video with H.264, H.265, or VP9 and publish each frame's detections to shared memory for other processes.

On a 1080p, 59.94 fps file the pipeline runs at about 26.8 fps with a median capture-to-sink latency of 70 ms. The GPU limits it. Inference takes 37 ms a frame, so about 27 fps is the ceiling on this hardware.

## Results

| Measurement | Result | Limiter |
| --- | --- | --- |
| Serial loop (all stages on one thread) | 22.6 fps | the sum of all stages |
| Threaded pipeline, file input | 26.8 fps, p50 latency 70 ms | inference, 37 ms a frame |
| Live 60 fps feed, drop-newest policy | 26.2 fps, p50 65.5 ms | inference |
| Live 60 fps feed, latest-wins policy | 25.2 fps, p50 54.5 ms | inference |
| Detections to another process, spinning reader | p50 3.2 µs, p95 57 µs | CPU contention in the tail (a guess) |

[Performance](#performance) gives the method, the run counts, and the ranges behind each row.

## Architecture

![Thread graph: capture, preprocess, infer and post + render threads feed a sink on the main thread through queues, with a frame pool, a shared-memory ring and an encoder thread beside them](docs/architecture.svg)

Four stage threads are `std::jthread`s. The sink runs on the caller's thread, because highgui needs the main thread. A bounded queue joins each pair of stages. Each stage owns its resources (the `Capture`, the NPP context and its stream, the TensorRT context), so no stage shares mutable state with another except through a queue.

**The frame pool is the only brake.** `Pool<Frame>` lends out a fixed set of preallocated frames as `shared_ptr`s whose deleter returns the frame. Every queue on a file source can hold the whole pool, so a push never waits. The frame count (`--pool`) therefore sets how many frames sit ahead of the slowest stage, which sets latency. The frames and their device buffers are allocated once, at start.

**File sources block, live sources drop.** A file waits for a free frame before it reads, so it never drops a frame. A camera or a `scheme://` stream reads into a scratch image first and takes a pooled frame only if one is free. Otherwise the capture thread drops the read, so the next one is current.

**A live source processes the freshest frame.** The queue in front of infer holds one frame. When preprocess finishes a newer frame, it replaces the waiting one (`BoundedQueue::push_latest`) and the evicted frame returns to the pool. This needs the third frame that `--pool 3` provides. At `--pool 2` a live source behaves like the older drop-newest policy. Frame ids are renumbered at infer, because replaced frames leave gaps in the capture ids.

**Shutdown and errors.** Returning `false` from the sink stops capture, closes its queue, and drains the stages behind it. An exception in any stage is stored once, a shared `std::stop_source` wakes every blocked wait, and `Pipeline::run` rethrows the first error after all threads join. SIGINT and SIGTERM set a lock-free `std::atomic<bool>` that the sink polls, so an interrupted run still flushes the encoder and writes the JSON report.

## Build and run

You need Docker with the NVIDIA container toolkit and an NVIDIA GPU. Everything builds and runs inside the `inference-trt` image, built from NVIDIA's `tensorrt:23.03` container (TensorRT 8.5.3, CUDA 12.1) plus OpenCV 4.10 built from source and g++-11, so the host needs no other toolchain.

The `assets/` directory is not in git. Put these files in it.

- `yolov5s.onnx`, exported from Ultralytics YOLOv5 at 640x640.
- `coco.names`, one class name per line.
- `test.mp4`, any video. The numbers below use a 1080p file at 59.94 fps.

Build the engine once (about 3 minutes on the 940MX), then build and test the code.

```sh
docker compose run --rm trt trtexec --onnx=assets/yolov5s.onnx --saveEngine=assets/yolov5s.engine
docker compose run --rm trt bash -c "cmake -S . -B build-trt && cmake --build build-trt -j4 && ctest --test-dir build-trt --output-on-failure"
sh tests/tsan.sh
```

Run on a file and print the JSON report.

```sh
docker compose run --rm trt ./build-trt/inference -s assets/test.mp4 --headless --frames 600 --json -
```

Drop `--headless` for a window, after running `xhost +local:` once per login. A windowed run works with `ipc: host` set.

Write the annotated video with `--encode`. The container follows the file extension, and `--codec` accepts `libx264`, `libx265`, and `libvpx-vp9`.

```sh
docker compose run --rm trt ./build-trt/inference -s assets/test.mp4 --headless --frames 600 --encode /work/out.mp4 --codec libx264
```

Read from an RTSP camera. `cam60` and `cam20` are compose services that loop `test.mp4` at 59.94 fps and 20 fps. Start one at a time, because two together starve the CPU.

```sh
docker compose --profile rtsp up -d cam60
docker compose run --rm -e OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp" trt ./build-trt/inference -s rtsp://cam60:8554/cam --headless --frames 300 --json -
docker compose --profile rtsp down
```

Read the detections from a second process. Run the second command in another terminal once the first prints its first `frame 100:` line, because `det_sub` fails if the shared-memory name does not exist yet. It follows the newest frames from the moment it starts, so it must start well before the producer's 600 frames end.

```sh
docker compose run --rm trt ./build-trt/inference -s assets/test.mp4 --headless --frames 600 --publish
docker compose run --rm trt ./build-trt/det_sub --frames 300
```

`tools/udp_send` streams a file as MPEG-TS and prints the frame rate it reached. A reader of an `--encode udp://` H.264 stream needs `?buffer_size=4194304` on its URL. With the default 208 KB socket buffer, a reader logged 8 H.264 decode-error lines at 1080p, and with the larger buffer it logged none in two runs. Keyframe bursts overflowing the buffer is the likely cause, but no socket drop counter was read.

| Option | Default | Meaning |
| --- | --- | --- |
| `-m, --model_path` | `assets/yolov5s.engine` | TensorRT engine |
| `-s, --source` | `0` | camera index, file, or URL |
| `--npp, --no-npp` | `--npp` | preprocess on the GPU with NPP, or on the CPU |
| `--conf`, `--nms` | 0.25, 0.5 | confidence and NMS IoU thresholds |
| `--frames N` | unlimited | stop after N frames |
| `--pool N` | 2 for files, 3 for live | frames in flight, 2 to 16 |
| `--headless` | off | no window |
| `--json PATH` | off | JSON run report (`-` for stdout) |
| `--encode TARGET`, `--codec` | off, `libx264` | write the rendered video to a file or `udp://host:port` |
| `--publish [NAME]` | off, `/inference_dets` | publish detections to shared memory |

## Performance

All numbers come from one machine. It has a GeForce 940MX, 4 CPU cores, and a background app (`orca-ide`) that uses about 0.7 of a core and cannot be stopped. For that reason, comparisons interleave their settings so each side sees the same noise. The input is `assets/test.mp4`, 1080p at 59.94 fps. Runs are headless with 600 frames. Each stage's timer and the latency and throughput samples start at frame 10, so the first nine frames are warmup. Latency is measured from the moment the capture thread finishes reading a frame to the moment the sink receives it, so it excludes read and decode time. Each figure is the median of 5 runs unless the row says otherwise. The commit messages hold most of the raw figures. Where a figure comes from a separate measurement that is in no commit, the text says so.

**Serial against pipelined** (5 alternating runs each, 600 frames). That comparison is in no commit message. `6e79326` reports 22.6 to 26.6 fps at a pool of four, and `7680960` reports 26.8 fps at p50 70 ms for the pool-2 build.

| Build | fps | p50 latency |
| --- | --- | --- |
| Serial | 22.6 (22.1 to 22.8) | not recorded |
| Pipelined, `--pool 2` | 26.8 (26.7 to 26.8) | 70 ms |

Infer takes 37.3 ms a frame, which is 26.8 fps if nothing else costs anything. The pipelined build reaches that bound. A later check on the current `master` (5 runs, not interleaved) gave 26.6 fps (26.5 to 26.7), p50 69.8 ms, and 0 dropped frames.

**Pool size against latency on a file** (`7680960`; 600 frames for pools 2 and 4, 300 for pool 3; the run count is not recorded). Throughput stays at 26.8 fps. Latency is 70 ms at pool 2, 107 ms at pool 3, and 145 ms at pool 4. A file blocks instead of dropping, so each extra frame only waits in a queue and adds about one inference time. The default for a file is therefore 2.

**Live source, latest-wins against drop-newest** (600 frames from `cam60`, 4 good runs per setting, old and new binaries interleaved). `2a642e2` and `3f2220a` report the fps and p50 figures (65.6 and 54.3 ms in the first, from an earlier set of runs). The table below comes from a later 5-run interleaved measurement with extra instrumentation in scratch builds, which is in no commit.

| Build | fps | p50 latency | p99 latency | infer |
| --- | --- | --- | --- | --- |
| Drop-newest, `--pool 2` | 26.2 | 65.5 ms | 80.6 ms | 37.2 ms |
| Latest-wins, `--pool 3` | 25.2 | 54.5 ms | 84.0 ms | 39.1 ms |

Latency falls 17% for a 4% lower frame rate. The frame-to-frame spacing at the sink is no steadier (standard deviation 4.36 ms against 4.23 ms, overlapping ranges), so the gain is lower latency, not smoother spacing. The 4% fps cost fits a guess that replaced frames use GPU time. Infer rose from 37.2 to 39.1 ms and the new build preprocessed 737 frames that were then replaced. That is one correlation, not a traced cause. Across the 15 `cam60` runs of that measurement (three builds and settings, 5 each), 6 returned 0 frames and are excluded. 4 of those 6 came from a third reference setting, the old build at `--pool 3`. The cause was joining the stream mid-keyframe interval with a 1 s read timeout, since fixed by raising it to 5 s.

With the old policy, a 59.94 fps feed lost about 390 frames for every 300 processed. `cam20` runs at 20.0 fps and is unchanged by either policy.

**Encoding** (`e0c60cd`, 600 frames, 5 rounds that alternate the four settings; median and range).

| Encoder | fps | encode ms a frame | frames dropped of 600 |
| --- | --- | --- | --- |
| none | 26.7 (24.9 to 26.9) | | |
| `libx264` | 26.4 (20.1 to 26.6) | 32.9 | 18 (5 to 251) |
| `libx265` | 25.3 (21.9 to 25.8) | 90.2 | 336 (313 to 346) |
| `libvpx-vp9` | 25.9 (20.9 to 26.5) | 60.5 | 215 (167 to 339) |

All of this runs on the CPU, because the 940MX has no usable NVENC. All 4 CPU threads saturate during an encode (`top` showed 91 to 95% user), while infer stays at 37.3 to 37.8 ms. The lost fps comes from CPU contention, which raises read and preprocess time. H.264 keeps up. H.265 and VP9 do not at 1080p on this CPU. In every encoded run, the encoded frames plus the dropped frames equal 600, and `cv::VideoCapture` reads back exactly the encoded frame count at 1920x1080.

**Shared-memory ring** (`3b8fbdd`; single runs, a 600-frame producer on `assets/test.mp4` and a `det_sub` that read 300 frames, in two containers). Latency runs from the writer's `published_ns` stamp, taken just before its stores, to the reader's copy. It includes the writer's copy.

| Reader | p50 | p95 |
| --- | --- | --- |
| Polling every 100 µs | 94 µs | 1.9 ms |
| Spinning (`--poll-us 0`) | 3.2 µs | 57 µs |

With a polling reader, the p50 is the poll period, not the ring. On `cam60`, a spinning reader followed 529 frames with p50 1.8 µs, p95 761 µs, and p99 2.6 ms. That run is not in a commit message. `3b8fbdd` records a 200-frame `cam60` follow with 0 skipped. The millisecond tails are most likely CPU contention, since the cores are busy with inference and the camera containers, but that was not traced. Publishing costs nothing measurable. With `--publish` off and on (5 alternating runs, no subscriber), fps was 26.23 and 26.48, `post` was 0.231 and 0.230 ms, and p50 latency was 69.97 and 69.96 ms. The pipeline is bound by inference, so `post` is the sensitive measure.

## Design decisions

The commit named in each item holds the measurements and the reasoning in its message.

**TensorRT 8.5.3 on a Maxwell GPU** (`d812e45`). The 940MX is compute capability 5.0, and TensorRT 8.5.3 is the newest release that supports it. That release needs a CUDA older than the host's 12.9, so the project builds in NVIDIA's `tensorrt:23.03` image (CUDA 12.1). Moving from `cv::dnn` to TensorRT cut inference from 83 to 40.5 ms a frame. The commit measured 286 of 288 detections matching `cv::dnn` at IoU above 0.9. The later GPU test matches all 288 in both directions over 10 frames.

**Preprocess: CPU resize, GPU the rest** (`d812e45`, `073f230`). `cv::resize` shrinks the frame to fit the 640 box, so the upload is about 0.7 MB instead of the 6.2 MB 1080p frame. The upload lands inside a box pre-filled with the padding value. NPP then swaps channels, converts to float, scales by 1/255, and writes planar NCHW straight into the engine's input buffer, so the blob never returns to the host. The resize stays on the CPU for the smaller upload and for an image identical to the reference path. That is design reasoning, not a measured comparison with an all-GPU resize. The stage costs 3.0 ms in the serial loop (`d812e45`) and 4.7 ms in the pipeline (`7680960`), and the resize is probably most of it, but the stage was never timed in parts. An early version resized on the GPU with `nppiResizeSqrPixel`, chosen because `nppiResize`'s linear mode maps pixel centers differently from `cv::resize` and gives nearest-neighbor output on integer downscales.

**GPU objectness filter** (`1dd7cc7`). A CUDA kernel in `src/filter.cu` drops output rows below the objectness threshold and compacts the rest. At most a few hundred rows (about 150 KB) cross PCIe instead of all 25,200 rows (8.6 MB). Inference time fell from 40.5 to 35.7 ms and `post` from 0.4 to 0.2 ms. Box decoding and NMS stay on the CPU.

**An NPP stream race** (`aed8282`). In NPP 12.1, `nppiCopyConstBorder_8u_C3R_Ctx` ignores `ctx.hStream`. An `nsys` trace showed its kernel on a different stream from every other preprocess call. From frame 10 on, detections differed between runs of the same binary. The border is the same value every frame, so it is now written once with `cudaMemsetAsync`. The fix removed the call, and three unserialized runs now match a `CUDA_LAUNCH_BLOCKING=1` reference byte for byte.

**A false dependency on the copy engine** (`7680960`). The 940MX has one copy engine, so the infer thread's result download queued ahead of the preprocess thread's next upload. Preprocess measured 37 ms, almost all of it waiting. Synchronizing before the download brought it back to 4.7 ms. Throughput did not change, because inference is the bound.

**One back-pressure point** (`6e79326`, `a55f763`, `6ab67ff`, `2a642e2`, `3f2220a`). For a file, every queue can hold the whole pool, so no stage blocks on another and the pool alone limits the frames in flight. For a live source, the queue in front of infer holds one frame and a push there replaces the waiting frame instead of waiting. A camera read can block for a whole frame period. The capture thread therefore reads before it claims a pooled frame, so that no frame sits idle through the read. At `--pool 2` that raised a camera from 20.2 to 27.0 fps and cut dropped frames from 75 to 18. A source with a `scheme://` URL counts as live. Before this, a UDP stream backed up until FFmpeg's buffer overran and the run died at 523 of 900 frames. Stream sources open with 2 s open and 5 s read timeouts, because FFmpeg's default 30 s would stall the pipeline before a reconnect starts. The read timeout is longer than `cam60`'s 4.2 s keyframe interval, because the first frame after a (re)connect waits for a keyframe; at 1 s, about 40% of `cam60` runs returned 0 frames. A live source defaults to `--pool 3`. Capture then holds a fresh frame while one frame is in infer and one waits. A file keeps 2, because a third frame there only adds latency.

**CPU encoding that cannot stall the pipeline** (`e0c60cd`). The `Encoder` owns a thread and two slots of its own. The sink copies a frame into a free slot (about 1 ms at 1080p) or, when both are busy, drops the frame and counts it. The encoder never holds a pooled frame, because a held frame would leave capture, preprocess, and infer one frame to share. The constructor opens the codec and the output and writes the header, so an unknown extension, an unsupported codec and container pair, or a bad path exits 1 before the first frame. `main.cpp` makes the same check for an `--encode` target equal to the `--source` file. A file stamps timestamps from frame ids, so a dropped frame leaves a gap and the output keeps the source's duration.

**A seqlock ring for detections** (`3b8fbdd`). One writer and any number of readers in other processes share a POSIX shared-memory file. The writer never waits for a reader. The ring holds the last 8 frames of up to 128 detections. Publication `n` lives in slot `n % 8`, and its sequence number is `2n+1` while the writer fills the slot and `2n+2` once the slot is stable. A reader can tell "not published yet", "stable", and "overwritten" apart from the sequence alone.

The writer stores the odd sequence, issues a release fence, stores the payload, and then stores the even sequence with release order. A reader loads the sequence with acquire order, copies the payload, issues an acquire fence, and rechecks the sequence. A reader that fell behind gets `Lapped`, never a torn frame. Payload words are `std::atomic<uint64_t>` with relaxed order, so the concurrent copy is not a C++ data race. `det_ring.hpp` and `det_ring.cpp` have no OpenCV or CUDA includes, so a reader needs only those two files and `tests/tsan.sh` builds the ring without the GPU toolchain. Published ids are the sink's ids. After a lap, `det_sub` jumps to the newest complete frame and counts the frames it passed as "skipped". The count can include frames that were still readable, and it never includes capture drops.

**No MKL or BLAS.** TensorRT runs the convolutions and matrix multiplies inside the engine. The CPU work left is the resize, NMS, drawing, and encoding, and none of it is a dense linear-algebra problem.

## Testing

`ctest` runs 65 tests, 6 of them labeled `gpu`. The suite covers letterbox geometry, postprocess and NMS on synthetic rows, the queue and the pool, the encoder (including a readback of each codec), the JSON report, and the ring. The GPU tests compare the NPP preprocess with the CPU blob (maximum absolute difference 5.96e-8), compare TensorRT with `cv::dnn` on the ONNX model over 10 frames (288 of 288 detections matched in both directions), and check that the pipeline returns the same frames and detections as the serial stages. `tests/tsan.sh` builds the queue, pool, and ring tests alone under ThreadSanitizer (23 tests, no reports).

The ring's memory ordering has evidence of three kinds, and ThreadSanitizer is not one of them. GCC 11's ThreadSanitizer does not model `atomic_thread_fence`, and the writer and each reader map the file at different addresses, so it checks only the rest of the test. The evidence is the stress test, the seqlock argument (Boehm, "Can Seqlocks Get Along with Programming Language Memory Models?", 2012), and two independent reviews. The stress test has one writer and two readers, with every payload word derived from the publication number. Removing the reader's sequence recheck makes it fail with 2,844 torn frames in about 266,000 reads. Removing a fence does not make it fail, because x86 already orders those accesses. Nothing here has run on a weakly ordered CPU such as ARM.

## Known limits

- `compose.yaml` sets `ipc: host` on the `trt` service. Of three ways tried to share `/dev/shm` between two containers running as uid 1000, it was the only one that worked. A tmpfs volume gives each container its own empty tmpfs, and a plain volume is owned by root. Two side effects follow. Every `docker compose run trt` shares the host's `/dev/shm`, and a SIGKILL leaves the ring file there. Whether to keep the setting on `trt` or move it to a separate publisher service is undecided.
- `det_sub` reporting 0 skipped does not mean no frames were dropped. Published ids are the sink's, so frames dropped at capture or replaced before infer never receive an id. The producer's `--json` `dropped` field holds that count.
- After a producer restart, `det_sub` keeps waiting on the old mapping and needs a restart.
- `DetWriter` unlinks any same-user object that has its name, because it must replace a stale file. A second publisher on one name unlinks the first one's file.
- `det_sub` links all of `inference_core`. A small `det_ring` library would fix that.
- The encoder uses `kEncodeThreads = 0` (the codec's default), which is not tuned. A live source that delivers frames faster than its reported rate plays back slow, and `finish()` would hang on a stalled FIFO target.
- Everything is measured on one GPU and one 4-core CPU, with one background app running.

## Layout

| Path | Contents |
| --- | --- |
| `src/main.cpp` | command line and wiring |
| `src/pipeline.{hpp,cpp}` | the stage threads and their queues |
| `src/capture.{hpp,cpp}` | files, cameras, and streams, with reconnects |
| `src/preprocess.{hpp,cpp}` | letterbox geometry, CPU path, and the NPP path |
| `src/engine.{hpp,cpp}`, `src/filter.cu` | TensorRT wrapper and the objectness filter |
| `src/postprocess.{hpp,cpp}`, `src/render.{hpp,cpp}` | decode, NMS, and drawing |
| `src/bounded_queue.hpp`, `src/pool.hpp` | the queue and the frame pool |
| `src/encoder.{hpp,cpp}` | libavcodec output |
| `src/det_ring.{hpp,cpp}` | the shared-memory seqlock ring |
| `src/report.{hpp,cpp}`, `src/timing.hpp` | the JSON report and stage timing |
| `tools/det_sub.cpp`, `tools/udp_send.cpp` | the ring reader and the UDP test sender |
| `tests/` | GoogleTest suites and `tsan.sh` |
| `docs/architecture.svg` | the thread diagram above |
