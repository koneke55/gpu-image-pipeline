#pragma once
#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

void copy_sobel_kernels();
void host_sobel(const float* d_in, float* d_mag, float* d_dir, int W, int H);
