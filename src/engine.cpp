#include "engine.hpp"
#include "filter.hpp"

#include <nvtx3/nvtx3.hpp>

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

void TrtLogger::log(Severity severity, const char *msg) noexcept
{
    if (severity <= Severity::kWARNING)
        std::cerr << "[TRT] " << msg << "\n";
}

size_t volume(const nvinfer1::Dims &d)
{
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i)
        v *= d.d[i];
    return v;
}

Trt::Trt(const std::string &engine_path)
{
    std::ifstream f(engine_path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open engine " + engine_path);
    const std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    runtime.reset(nvinfer1::createInferRuntime(logger));
    engine.reset(runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine)
        throw std::runtime_error("failed to deserialize engine " + engine_path);
    context.reset(engine->createExecutionContext());

    if (engine->getNbIOTensors() != 2)
        throw std::runtime_error("expected 1 input and 1 output tensor");
    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char *name = engine->getIOTensorName(i);
        if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
            throw std::runtime_error(std::string("tensor ") + name + " is not float32");
        const bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        const nvinfer1::Dims d = engine->getTensorShape(name);
        DeviceBuffer<float> &buf = is_input ? d_in : d_out;
        (is_input ? in_dims : out_dims) = d;
        buf = make_device<float>(volume(d));
        context->setTensorAddress(name, buf.get());
    }
    d_kept = make_device<float>(volume(out_dims));
    d_count = make_device<int>(1);
}

void infer(Trt &trt, FrameData &framedata, const Config &cfg)
{
    const int num_boxes = trt.out_dims.d[1];
    const int stride = trt.out_dims.d[2];
    {
        nvtx3::scoped_range r{"trt"};
        if (!trt.context->enqueueV3(trt.stream.get()))
            throw std::runtime_error("enqueueV3 failed");
    }
    {
        nvtx3::scoped_range r{"filter"};
        check(filter_rows(trt.d_out.get(), num_boxes, stride, cfg.conf_thr, trt.d_kept.get(), trt.d_count.get(), trt.stream.get()));
    }
    {
        nvtx3::scoped_range r{"d2h"};
        check(cudaMemcpyAsync(&framedata.num_kept, trt.d_count.get(), sizeof(int), cudaMemcpyDeviceToHost, trt.stream.get()));
        check(cudaStreamSynchronize(trt.stream.get()));
        check(cudaMemcpyAsync(framedata.out.data, trt.d_kept.get(), static_cast<size_t>(framedata.num_kept) * stride * sizeof(float),
                              cudaMemcpyDeviceToHost, trt.stream.get()));
        check(cudaStreamSynchronize(trt.stream.get()));
    }
}
