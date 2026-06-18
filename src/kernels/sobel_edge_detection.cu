#include "../../include/kernels/sobel_edge_detection.cuh"
#include <cmath>

__constant__ int d_sobel_x[9];
__constant__ int d_sobel_y[9];

__global__ void sobel_kernel(const float* in, float* mag, float* dir, int W, int H){
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x>=W||y>=H) return;
  float gx=0, gy=0;
  for(int ky=-1;ky<=1;++ky) for(int kx=-1;kx<=1;++kx){
    int ix = min(max(x+kx,0),W-1);
    int iy = min(max(y+ky,0),H-1);
    float v = in[iy*W + ix];
    int idx = (ky+1)*3 + (kx+1);
    gx += d_sobel_x[idx]*v; gy += d_sobel_y[idx]*v;
  }
  mag[y*W + x] = sqrtf(gx*gx + gy*gy);
  dir[y*W + x] = atan2f(gy,gx);
}

void copy_sobel_kernels(){
  int hx[9] = {-1,0,1,-2,0,2,-1,0,1};
  int hy[9] = {-1,-2,-1,0,0,0,1,2,1};
  CUDA_CHECK(cudaMemcpyToSymbol(d_sobel_x,hx,sizeof(hx)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_sobel_y,hy,sizeof(hy)));
}

void host_sobel(const float* d_in, float* d_mag, float* d_dir, int W, int H){
  dim3 t(16,16); dim3 b((W+t.x-1)/t.x,(H+t.y-1)/t.y);
  sobel_kernel<<<b,t>>>(d_in,d_mag,d_dir,W,H);
  CUDA_CHECK(cudaGetLastError());
}
