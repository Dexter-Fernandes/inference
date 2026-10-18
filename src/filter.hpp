#pragma once

#include <cuda_runtime.h>

// Compacts rows with objectness >= thr into d_kept and the row count into d_count.
cudaError_t filter_rows(const float *d_out, int num_boxes, int stride, float thr, float *d_kept, int *d_count, cudaStream_t s);
