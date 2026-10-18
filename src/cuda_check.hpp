#pragma once

#include <cuda_runtime.h>
#include <npp.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

inline void check(NppStatus s)
{
    if (s < NPP_SUCCESS)
        throw std::runtime_error("NPP error " + std::to_string(s));
}

inline void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e));
}

struct CudaFree
{
    void operator()(void *p) const noexcept { cudaFree(p); }
};

template <class T>
using DeviceBuffer = std::unique_ptr<T, CudaFree>;

template <class T>
DeviceBuffer<T> make_device(size_t count)
{
    T *p = nullptr;
    check(cudaMalloc(&p, count * sizeof(T)));
    return DeviceBuffer<T>(p);
}

class CudaStream
{
public:
    CudaStream() { check(cudaStreamCreate(&s_)); }
    ~CudaStream() { cudaStreamDestroy(s_); }
    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    cudaStream_t get() const { return s_; }

private:
    cudaStream_t s_ = nullptr;
};
