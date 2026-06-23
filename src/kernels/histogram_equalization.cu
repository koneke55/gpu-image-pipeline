#include "../../include/kernels/histogram_equalization.cuh"
// Advanced GPU histogram/CLAHE implementation
#include <vector>
#include <algorithm>

// Number of histogram bins
#define BIN 256

// Build per-block shared histogram then atomically add to global histogram.
// Build per-block shared histogram then write it to a per-block global array (no global atomics).
// per_block_hist must be sized blocks * channels * BIN.
__global__ void build_histogram_kernel(const float* data, int N, int channels, int* per_block_hist){
  extern __shared__ unsigned int s_hist[]; // size channels*BIN when used
  int tid = threadIdx.x;
  int total_bins = channels * BIN;
  // initialize shared histogram
  for(int i = tid; i < total_bins; i += blockDim.x) s_hist[i] = 0u;
  __syncthreads();

  // process elements with block-stride loop (handles interleaved channels)
  int idx = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;
  while(idx < N){
    int base = idx * channels;
    // only build histogram for channel 0 here (global per-channel support handled elsewhere)
    int b = int(data[base] * 255.0f);
    b = b < 0 ? 0 : (b > 255 ? 255 : b);
    atomicAdd(&s_hist[b], 1u);
    idx += stride;
  }
  __syncthreads();

  // write shared histogram to per-block global memory (no atomics)
  int out_base = blockIdx.x * total_bins;
  for(int i = tid; i < total_bins; i += blockDim.x){
    per_block_hist[out_base + i] = (int)s_hist[i];
  }
}

// Reduce per-block histograms into a single global histogram. Each thread handles one bin (and channel).
__global__ void reduce_block_histograms(const int* per_block_hist, int* hist_out, int blocks, int channels){
  int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_bins = channels * BIN;
  if(bin_idx >= total_bins) return;
  int sum = 0;
  // per_block_hist is organized as [block0: bins...][block1: bins...]
  for(int b = 0; b < blocks; ++b){
    sum += per_block_hist[b * total_bins + bin_idx];
  }
  hist_out[bin_idx] = sum;
}

// Per-tile histogram kernel: one block per tile, independent per-channel histogram arrays
__global__ void build_tile_histograms(const float* data, int W, int H, int channels, int tile_w, int tile_h, int tiles_x, int tiles_y, int* per_tile_hist){
  int tile_idx = blockIdx.x; // one block per tile
  int tx = tile_idx % tiles_x;
  int ty = tile_idx / tiles_x;
  int x0 = tx * tile_w;
  int y0 = ty * tile_h;
  int x1 = min(x0 + tile_w, W);
  int y1 = min(y0 + tile_h, H);

  extern __shared__ unsigned int s_hist[]; // sized channels*BIN
  int tid = threadIdx.x;
  int total_bins = channels * BIN;
  for(int i = tid; i < total_bins; i += blockDim.x) s_hist[i] = 0u;
  __syncthreads();

  // iterate over pixels in tile with thread stride
  for(int y = y0 + tid; y < y1; y += blockDim.x){
    for(int x = x0; x < x1; ++x){
      int p = (y * W + x) * channels;
      for(int c = 0; c < channels; ++c){
        int b = int(data[p + c] * 255.0f);
        b = b < 0 ? 0 : (b > 255 ? 255 : b);
        atomicAdd(&s_hist[c * BIN + b], 1u);
      }
    }
  }
  __syncthreads();

  // write shared hist to global per_tile_hist
  int base = tile_idx * channels * BIN;
  for(int i = tid; i < total_bins; i += blockDim.x){
    unsigned int v = s_hist[i];
    per_tile_hist[base + i] = (int)v;
  }
}

// Apply mapping for non-CLAHE (single mapping per channel)
__global__ void apply_histogram_mapping(float* data, int N, int channels, const float* mapping){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while(idx < N){
    int base = idx * channels;
    for(int c = 0; c < channels; ++c){
      int b = int(data[base + c] * 255.0f);
      b = b < 0 ? 0 : (b > 255 ? 255 : b);
      data[base + c] = mapping[c * BIN + b];
    }
    idx += stride;
  }
}

// Apply CLAHE mapping with bilinear interpolation between tile mappings
__global__ void apply_clahe_mapping(float* data, int W, int H, int channels, int tile_w, int tile_h, int tiles_x, int tiles_y, const float* per_tile_map){
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = blockDim.x * gridDim.x;
  for(int x = px; x < W; x += stride_x){
    for(int y = 0; y < H; ++y){
      int p = (y * W + x) * channels;
      float fx = float(x) / float(tile_w) - 0.5f;
      float fy = float(y) / float(tile_h) - 0.5f;
      int tx = floorf(fx);
      int ty = floorf(fy);
      float wx = fx - tx;
      float wy = fy - ty;
      // clamp tile indices
      int tx0 = max(0, min(tx, tiles_x - 1));
      int ty0 = max(0, min(ty, tiles_y - 1));
      int tx1 = max(0, min(tx + 1, tiles_x - 1));
      int ty1 = max(0, min(ty + 1, tiles_y - 1));

      for(int c = 0; c < channels; ++c){
        int b = int(data[p + c] * 255.0f);
        b = b < 0 ? 0 : (b > 255 ? 255 : b);
        // fetch four mappings
        int idx00 = ((ty0 * tiles_x + tx0) * channels + c) * BIN + b;
        int idx10 = ((ty0 * tiles_x + tx1) * channels + c) * BIN + b;
        int idx01 = ((ty1 * tiles_x + tx0) * channels + c) * BIN + b;
        int idx11 = ((ty1 * tiles_x + tx1) * channels + c) * BIN + b;
        float v00 = per_tile_map[idx00];
        float v10 = per_tile_map[idx10];
        float v01 = per_tile_map[idx01];
        float v11 = per_tile_map[idx11];
        // bilinear interpolation
        float v0 = v00 * (1.0f - wx) + v10 * wx;
        float v1 = v01 * (1.0f - wx) + v11 * wx;
        float v = v0 * (1.0f - wy) + v1 * wy;
        data[p + c] = v;
      }
    }
  }
}

void host_histogram_equalization(float* d_in, int W, int H, bool clahe, int channels, int tile_w, int tile_h, float clip_limit){
  int N = W * H;

  if(!clahe){
    // Single global histogram per channel
    int* d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hist, channels * BIN * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, channels * BIN * sizeof(int)));

      int threads = 256;
      int blocks = (N + threads - 1) / threads;
      // allocate per-block histograms
      int* d_block_hist = nullptr;
      CUDA_CHECK(cudaMalloc(&d_block_hist, blocks * channels * BIN * sizeof(int)));
      // shared memory size: channels * BIN * sizeof(unsigned int)
      size_t sh_bytes = channels * BIN * sizeof(unsigned int);
      build_histogram_kernel<<<blocks, threads, sh_bytes>>>(d_in, N, channels, d_block_hist);
      CUDA_CHECK(cudaGetLastError());

      // reduce per-block histograms into d_hist
      CUDA_CHECK(cudaMemset(d_hist, 0, channels * BIN * sizeof(int)));
      int rthreads = 256;
      int rblocks = (channels * BIN + rthreads - 1) / rthreads;
      reduce_block_histograms<<<rblocks, rthreads>>>(d_block_hist, d_hist, blocks, channels);
      CUDA_CHECK(cudaGetLastError());

      // copy histogram to host and compute per-channel CDF and mapping
      std::vector<int> h_hist(channels * BIN);
      CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist, channels * BIN * sizeof(int), cudaMemcpyDeviceToHost));
      // free per-block hist buffer
      cudaFree(d_block_hist);
    std::vector<float> h_map(channels * BIN);
    for(int c = 0; c < channels; ++c){
      int acc = 0;
      for(int i = 0; i < BIN; ++i){ acc += h_hist[c * BIN + i]; h_map[c * BIN + i] = float(acc) / float(N); }
    }

    float* d_map = nullptr;
    CUDA_CHECK(cudaMalloc(&d_map, channels * BIN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_map, h_map.data(), channels * BIN * sizeof(float), cudaMemcpyHostToDevice));

    apply_histogram_mapping<<<blocks, threads>>>(d_in, N, channels, d_map);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_hist); cudaFree(d_map);
    // free per-block hist
    // d_block_hist freed earlier after reduction
    return;
  }

  // CLAHE path: per-tile histograms and mappings
  int tiles_x = (W + tile_w - 1) / tile_w;
  int tiles_y = (H + tile_h - 1) / tile_h;
  int tiles = tiles_x * tiles_y;

  // allocate per-tile histograms
  int* d_tile_hist = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tile_hist, tiles * channels * BIN * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_tile_hist, 0, tiles * channels * BIN * sizeof(int)));

  // launch one block per tile, threads per block tuned to min(256, tile_h)
  int threads = min(256, max(32, tile_h));
  size_t shared_bytes = channels * BIN * sizeof(unsigned int);
  build_tile_histograms<<<tiles, threads, shared_bytes>>>(d_in, W, H, channels, tile_w, tile_h, tiles_x, tiles_y, d_tile_hist);
  CUDA_CHECK(cudaGetLastError());

  // copy per-tile hist to host
  std::vector<int> h_tile_hist(tiles * channels * BIN);
  CUDA_CHECK(cudaMemcpy(h_tile_hist.data(), d_tile_hist, tiles * channels * BIN * sizeof(int), cudaMemcpyDeviceToHost));

  // apply clipping and compute per-tile mapping
  std::vector<float> h_tile_map(tiles * channels * BIN);
  for(int t = 0; t < tiles; ++t){
    for(int c = 0; c < channels; ++c){
      int base = (t * channels + c) * BIN;
      int npixels = 0;
      for(int i = 0; i < BIN; ++i) npixels += h_tile_hist[base + i];
      // clip histogram
      float clip = clip_limit;
      int excess = 0;
      for(int i = 0; i < BIN; ++i){
        if(h_tile_hist[base + i] > clip){
          excess += h_tile_hist[base + i] - int(clip);
          h_tile_hist[base + i] = int(clip);
        }
      }
      // redistribute excess uniformly
      int redistribute = excess / BIN;
      for(int i = 0; i < BIN; ++i) h_tile_hist[base + i] += redistribute;

      int acc = 0;
      for(int i = 0; i < BIN; ++i){ acc += h_tile_hist[base + i]; h_tile_map[base + i] = (npixels>0) ? float(acc) / float(npixels) : 0.0f; }
    }
  }

  // copy mappings to device
  float* d_tile_map = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tile_map, tiles * channels * BIN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_tile_map, h_tile_map.data(), tiles * channels * BIN * sizeof(float), cudaMemcpyHostToDevice));

  // apply mapping with interpolation; launch threads across X dimension
  int bx = 256;
  int gx = (W + bx - 1) / bx;
  apply_clahe_mapping<<<gx, bx>>>(d_in, W, H, channels, tile_w, tile_h, tiles_x, tiles_y, d_tile_map);
  CUDA_CHECK(cudaGetLastError());

  cudaFree(d_tile_hist);
  cudaFree(d_tile_map);
}
