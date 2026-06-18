#include "../../include/kernels/tone_mapping.cuh"
// Simple tone mapping: compute log-average on host then apply GPU kernel
#include <cmath>
#include <vector>

__global__ void apply_tonemap_kernel(float* data, int N, float exposure, float log_avg){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=N) return;
  float L = data[i];
  data[i] = L * exposure / (1.0f + L/log_avg);
}

float host_reinhard_tonemap(float* d_luminance, int N, float exposure){
  // copy to host, compute log-average luminance
  std::vector<float> h(N);
  CUDA_CHECK(cudaMemcpy(h.data(), d_luminance, N*sizeof(float), cudaMemcpyDeviceToHost));
  double sum = 0.0;
  for(int i=0;i<N;++i) sum += log(1e-6 + std::max(0.0f, h[i]));
  float log_avg = expf(sum / N);
  int threads = 256; int blocks = (N + threads - 1)/threads;
  apply_tonemap_kernel<<<blocks,threads>>>(d_luminance, N, exposure, log_avg);
  CUDA_CHECK(cudaGetLastError());
  return log_avg;
}
