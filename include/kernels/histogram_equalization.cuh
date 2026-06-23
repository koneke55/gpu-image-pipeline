#pragma once
#include "../utils/cuda_utils.cuh"
// d_in: device pointer to interleaved float image [W*H*channels], range [0,1]
// channels: number of channels (1 for grayscale, 3 for RGB)
// If clahe==true, tile_w/tile_h control tile size and clip_limit controls histogram clipping.
void host_histogram_equalization(float* d_in, int W, int H, bool clahe=false, int channels=1, int tile_w=64, int tile_h=64, float clip_limit=40.0f);
