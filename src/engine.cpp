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
        const nvinfer1::Dims d = engine->getTensorShape(name);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            input_name = name;
            in_dims = d;
            continue;
        }
        out_dims = d;
        d_out = make_device<float>(volume(d));
        context->setTensorAddress(name, d_out.get());
    }
    if (input_name.empty())
        throw std::runtime_error("engine has no input tensor");
    d_kept = make_device<float>(volume(out_dims));
    d_count = make_device<int>(1);
}

Frame make_frame(const Trt &trt)
{
    Frame f;
    f.d_in = make_device<float>(volume(trt.in_dims));
    const std::vector<int> out_shape(trt.out_dims.d, trt.out_dims.d + trt.out_dims.nbDims);
    f.out.create(static_cast<int>(out_shape.size()), out_shape.data(), CV_32F);
    return f;
}

void infer(Trt &trt, Frame &frame, const Config &cfg)
{
    const int num_boxes = trt.out_dims.d[1];
    const int stride = trt.out_dims.d[2];
    {
        nvtx3::scoped_range r{"trt"};
        if (!trt.context->setTensorAddress(trt.input_name.c_str(), frame.d_in.get()))
            throw std::runtime_error("setTensorAddress failed");
        if (!trt.context->enqueueV3(trt.stream.get()))
            throw std::runtime_error("enqueueV3 failed");
    }
    {
        nvtx3::scoped_range r{"filter"};
        check(filter_rows(trt.d_out.get(), num_boxes, stride, cfg.conf_thr, trt.d_kept.get(), trt.d_count.get(), trt.stream.get()));
    }
    {
        nvtx3::scoped_range r{"d2h"};
        check(cudaMemcpyAsync(&frame.num_kept, trt.d_count.get(), sizeof(int), cudaMemcpyDeviceToHost, trt.stream.get()));
        check(cudaStreamSynchronize(trt.stream.get()));
        check(cudaMemcpyAsync(frame.out.data, trt.d_kept.get(), static_cast<size_t>(frame.num_kept) * stride * sizeof(float),
                              cudaMemcpyDeviceToHost, trt.stream.get()));
        check(cudaStreamSynchronize(trt.stream.get()));
    }
}
