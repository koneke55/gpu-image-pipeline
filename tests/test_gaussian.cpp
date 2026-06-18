#include <catch2/catch.hpp>
#include <vector>
#include "../include/kernels/gaussian_blur.cuh"

TEST_CASE("gaussian sanity", "[gaussian]"){
  int W=128,H=128;
  std::vector<float> img(W*H,1.0f), out(W*H,0.0f);
  float *d_in,*d_out;
  CUDA_CHECK(cudaMalloc(&d_in,W*H*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out,W*H*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in,img.data(),W*H*sizeof(float),cudaMemcpyHostToDevice));
  float kernel[9]; for(int i=0;i<9;++i) kernel[i]=1.0f/9.0f;
  copy_kernel_to_const(kernel,9);
  host_gaussian_blur(d_in,d_out,W,H,3);
  CUDA_CHECK(cudaMemcpy(out.data(),d_out,W*H*sizeof(float),cudaMemcpyDeviceToHost));
  REQUIRE(out[0] == Approx(1.0f).margin(1e-3));
  cudaFree(d_in); cudaFree(d_out);
}
