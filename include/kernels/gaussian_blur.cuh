#pragma once
#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

// host helpers
void copy_kernel_to_const(const float* h_kernel, int kernel_size);
void host_gaussian_blur(float* d_in, float* d_out, int W, int H, int kernel_size);
