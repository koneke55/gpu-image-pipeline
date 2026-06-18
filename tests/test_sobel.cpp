#include <catch2/catch.hpp>
#include <vector>
#include "../include/kernels/sobel_edge_detection.cuh"

TEST_CASE("sobel sanity","[sobel]"){
  int W=64,H=64;
  std::vector<float> img(W*H,0.0f);
  for(int y=0;y<H;++y) for(int x=0;x<W;++x) img[y*W+x] = (x>W/2)?1.0f:0.0f;
  float *d_in,*d_mag,*d_dir;
  CUDA_CHECK(cudaMalloc(&d_in,W*H*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mag,W*H*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dir,W*H*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in,img.data(),W*H*sizeof(float),cudaMemcpyHostToDevice));
  copy_sobel_kernels();
  host_sobel(d_in,d_mag,d_dir,W,H);
  std::vector<float> mag(W*H);
  CUDA_CHECK(cudaMemcpy(mag.data(),d_mag,W*H*sizeof(float),cudaMemcpyDeviceToHost));
  REQUIRE(mag[H*(W/2)+W/2] > 0.0f);
  cudaFree(d_in); cudaFree(d_mag); cudaFree(d_dir);
}
