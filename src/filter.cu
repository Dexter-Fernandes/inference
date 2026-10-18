#include "filter.hpp"

// Compacts rows whose objectness passes thr into kept; count ends up as the number kept.
// Row order is whatever atomicAdd hands out, which NMS doesn't care about.
__global__ void filter_kernel(const float *out, int num_boxes, int stride, float thr, float *kept, int *count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boxes)
        return;
    const float *row = out + static_cast<size_t>(i) * stride;
    if (row[4] < thr)
        return;
    float *dst = kept + static_cast<size_t>(atomicAdd(count, 1)) * stride;
    for (int k = 0; k < stride; ++k)
        dst[k] = row[k];
}

cudaError_t filter_rows(const float *d_out, int num_boxes, int stride, float thr, float *d_kept, int *d_count, cudaStream_t s)
{
    if (const cudaError_t e = cudaMemsetAsync(d_count, 0, sizeof(int), s); e != cudaSuccess)
        return e;
    constexpr int block = 256;
    filter_kernel<<<(num_boxes + block - 1) / block, block, 0, s>>>(d_out, num_boxes, stride, thr, d_kept, d_count);
    return cudaGetLastError();
}
