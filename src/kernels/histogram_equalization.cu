#include "../../include/kernels/histogram_equalization.cuh"
// Simple GPU histogram builder + host CDF + GPU apply mapping
#include <vector>
#include <algorithm>

// Number of histogram bins
#define BIN 256

// Build per-block shared histogram then atomically add to global histogram.
__global__ void build_histogram_kernel(const float* data, int N, int* hist){
  __shared__ unsigned int s_hist[BIN];
  int tid = threadIdx.x;
  // initialize shared histogram (stride in case blockDim.x < BIN)
  for(int i = tid; i < BIN; i += blockDim.x) s_hist[i] = 0u;
  __syncthreads();

  // process elements with block-stride loop
  int idx = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;
  while(idx < N){
    int b = int(data[idx] * 255.0f);
    b = b < 0 ? 0 : (b > 255 ? 255 : b);
    atomicAdd(&s_hist[b], 1u);
    idx += stride;
  }
  __syncthreads();

  // flush shared histogram to global
  for(int i = tid; i < BIN; i += blockDim.x){
    unsigned int v = s_hist[i];
    if(v) atomicAdd(&hist[i], (int)v);
  }
}

// Apply mapping using a precomputed device-side float lookup table
__global__ void apply_histogram_mapping(float* data, int N, const float* mapping){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while(i < N){
    int b = int(data[i] * 255.0f);
    b = b < 0 ? 0 : (b > 255 ? 255 : b);
    data[i] = mapping[b];
    i += stride;
  }
}

void host_histogram_equalization(float* d_in, int W, int H, bool clahe){
  int N = W * H;
  // allocate histogram on device
  int* d_hist = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hist, BIN * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_hist, 0, BIN * sizeof(int)));

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  build_histogram_kernel<<<blocks,threads>>>(d_in, N, d_hist);
  CUDA_CHECK(cudaGetLastError());

  // copy histogram to host and compute CDF
  std::vector<int> h_hist(BIN);
  CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist, BIN * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> h_cdf(BIN);
  int acc = 0;
  for(int i = 0; i < BIN; ++i){ acc += h_hist[i]; h_cdf[i] = acc; }

  // precompute mapping (float) on host and copy to device
  std::vector<float> h_map(BIN);
  for(int i = 0; i < BIN; ++i){
    h_map[i] = float(h_cdf[i]) / float(N);
  }
  float* d_map = nullptr;
  CUDA_CHECK(cudaMalloc(&d_map, BIN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_map, h_map.data(), BIN * sizeof(float), cudaMemcpyHostToDevice));

  // apply mapping on GPU
  apply_histogram_mapping<<<blocks,threads>>>(d_in, N, d_map);
  CUDA_CHECK(cudaGetLastError());

  cudaFree(d_hist);
  cudaFree(d_map);
}
