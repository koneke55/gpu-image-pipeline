#include "../../include/kernels/histogram_equalization.cuh"
// Simple GPU histogram builder + host CDF + GPU apply mapping
#include <vector>
#include <algorithm>

__global__ void build_histogram_kernel(const float* data, int N, int* hist){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=N) return;
  int b = min(max(int(data[i]*255.0f),0),255);
  atomicAdd(&hist[b],1);
}

__global__ void apply_histogram_mapping(float* data, int N, const int* cdf, int total){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=N) return;
  int b = min(max(int(data[i]*255.0f),0),255);
  int c = cdf[b];
  data[i] = float(c) / float(total);
}

void host_histogram_equalization(float* d_in, int W, int H, bool clahe){
  int N = W*H;
  const int BIN = 256;
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
  CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist, BIN*sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> h_cdf(BIN);
  int acc = 0;
  for(int i=0;i<BIN;++i){ acc += h_hist[i]; h_cdf[i] = acc; }

  // copy CDF back
  int* d_cdf=nullptr; CUDA_CHECK(cudaMalloc(&d_cdf, BIN*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_cdf, h_cdf.data(), BIN*sizeof(int), cudaMemcpyHostToDevice));

  // apply mapping on GPU
  apply_histogram_mapping<<<blocks,threads>>>(d_in, N, d_cdf, N);
  CUDA_CHECK(cudaGetLastError());

  cudaFree(d_hist); cudaFree(d_cdf);
}
