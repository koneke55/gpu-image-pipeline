#include "../../include/kernels/color_space_conversion.cuh"
#include <algorithm>

__global__ void rgb_to_gray_kernel(const float* in, float* out, int W, int H){
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x>=W||y>=H) return;
  int idx = (y*W + x)*3;
  float r = in[idx+0], g = in[idx+1], b = in[idx+2];
  out[y*W + x] = 0.299f*r + 0.587f*g + 0.114f*b;
}

void host_rgb_to_gray(const float* d_in, float* d_out, int W, int H){
  dim3 t(16,16); dim3 b((W+t.x-1)/t.x,(H+t.y-1)/t.y);
  rgb_to_gray_kernel<<<b,t>>>(d_in,d_out,W,H);
  CUDA_CHECK(cudaGetLastError());
}
