#pragma once
#include "../utils/cuda_utils.cuh"
void host_histogram_equalization(float* d_in, int W, int H, bool clahe=false);
