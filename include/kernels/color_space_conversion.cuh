#pragma once
#include "../utils/cuda_utils.cuh"
void host_rgb_to_gray(const float* d_in, float* d_out, int W, int H);
