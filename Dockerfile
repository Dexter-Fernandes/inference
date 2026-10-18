FROM nvcr.io/nvidia/tensorrt:23.03-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# CPU-only OpenCV; TensorRT does the inference.
ARG OPENCV_VERSION=4.10.0
RUN git clone --depth 1 -b ${OPENCV_VERSION} https://github.com/opencv/opencv.git /tmp/opencv \
    && cmake -S /tmp/opencv -B /tmp/opencv/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,dnn \
        -DWITH_CUDA=OFF -DWITH_GTK=ON -DWITH_FFMPEG=ON \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
    && cmake --build /tmp/opencv/build -j"$(nproc)" \
    && cmake --install /tmp/opencv/build \
    && ldconfig \
    && rm -rf /tmp/opencv

WORKDIR /work
