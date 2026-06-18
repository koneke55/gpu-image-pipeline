#include "../../include/kernels/gaussian_blur.cuh"
#include <cmath>
#include <algorithm>

__constant__ float d_kernel[21]; // support up to radius 10 (21 taps)

__global__ void gaussian_separable_row(const float* in, float* out, int W, int H, int radius){
  extern __shared__ float sdata[];
  int ty = blockIdx.y*blockDim.y + threadIdx.y;
  int tx = blockIdx.x*blockDim.x + threadIdx.x;
  int lane = threadIdx.x;
  int half = radius;
  // load tile with halo
  int tile_w = blockDim.x + 2*half;
  int sidx = threadIdx.y*tile_w + threadIdx.x + half;
  int gx = tx;
  int gy = ty;
  int load_x = gx - half;
  load_x = min(max(load_x,0), W-1);
  if(gy < H){
    sdata[sidx] = in[gy*W + load_x];
    // left halo
    if(threadIdx.x < half){
      int lx = gx - half - threadIdx.x;
      lx = min(max(lx,0),W-1);
      sdata[sidx - half] = in[gy*W + lx];
    }
    // right halo
    if(threadIdx.x >= blockDim.x - half){
      int rx = gx + half;
      rx = min(max(rx,0),W-1);
      sdata[sidx + half] = in[gy*W + rx];
    }
  }
  __syncthreads();
  if(gx < W && gy < H){
    float sum = 0.0f;
    for(int k=-radius;k<=radius;++k){ sum += d_kernel[k+radius]*sdata[sidx + k]; }
    out[gy*W + gx] = sum;
  }
}

__global__ void gaussian_separable_col(const float* in, float* out, int W, int H, int radius){
  extern __shared__ float sdata[];
  int ty = blockIdx.y*blockDim.y + threadIdx.y;
  int tx = blockIdx.x*blockDim.x + threadIdx.x;
  int half = radius;
  int tile_h = blockDim.y + 2*half;
  int sidx = (threadIdx.y + half)*blockDim.x + threadIdx.x;
  int gy = ty;
  int gx = tx;
  int load_y = gy - half;
  load_y = min(max(load_y,0), H-1);
  if(gx < W){
    sdata[sidx] = in[load_y*W + gx];
    if(threadIdx.y < half){ int ly = gy - half - threadIdx.y; ly = min(max(ly,0),H-1); sdata[sidx - half*blockDim.x] = in[ly*W + gx]; }
    if(threadIdx.y >= blockDim.y - half){ int ry = gy + half; ry = min(max(ry,0),H-1); sdata[sidx + half*blockDim.x] = in[ry*W + gx]; }
  }
  __syncthreads();
  if(gx < W && gy < H){ float sum=0.0f; for(int k=-radius;k<=radius;++k) sum += d_kernel[k+radius]*sdata[sidx + k*blockDim.x]; out[gy*W + gx] = sum; }
}

void host_gaussian_blur(const float* d_in, float* d_out, int W, int H, int kernel_size){
  int radius = kernel_size/2;
  // assumption: host has prepared kernel in host memory; copy done by caller via copy_kernel_to_const
  dim3 t(32,8);
  dim3 b((W + t.x -1)/t.x, (H + t.y -1)/t.y);
  size_t shared_row = (t.x + 2*radius)*t.y*sizeof(float);
  gaussian_separable_row<<<b,t,shared_row>>>(d_in, d_out, W, H, radius);
  CUDA_CHECK(cudaGetLastError());
  // column pass uses same buffer
  dim3 t2(32,8);
  dim3 b2((W + t2.x -1)/t2.x, (H + t2.y -1)/t2.y);
  size_t shared_col = (t2.y + 2*radius)*t2.x*sizeof(float);
  gaussian_separable_col<<<b2,t2,shared_col>>>(d_out, d_in, W, H, radius);
}

void copy_kernel_to_const(const float* h_kernel, int kernel_size){
  CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_size*sizeof(float)));
}
