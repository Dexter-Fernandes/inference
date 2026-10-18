#pragma once

#include "common.hpp"
#include "cuda_check.hpp"

#include <NvInfer.h>

#include <cstddef>
#include <memory>
#include <string>

struct TrtLogger : nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override;
};

size_t volume(const nvinfer1::Dims &d);

// Owns the engine, one input and one output device buffer, and the stream everything runs on.
struct Trt
{
    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    CudaStream stream;
    DeviceBuffer<float> d_in, d_out;
    DeviceBuffer<float> d_kept; // filtered output rows, same capacity as d_out
    DeviceBuffer<int> d_count;
    nvinfer1::Dims in_dims{}, out_dims{};

    explicit Trt(const std::string &engine_path);
};

// Runs the engine, then drops rows below the objectness threshold on the GPU so only
// the survivors cross PCIe (a few hundred rows instead of the full 8.6 MB output).
void infer(Trt &trt, FrameData &framedata, const Config &cfg);
