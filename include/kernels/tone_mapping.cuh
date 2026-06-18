#pragma once
#include "../utils/cuda_utils.cuh"
float host_reinhard_tonemap(float* d_luminance, int N, float exposure);
