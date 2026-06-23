#include <cstdio>
#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/kernels/histogram_equalization.cuh"
#include "../../include/utils/cuda_utils.cuh"

int main(int argc, char** argv){
  int W = 1024, H = 768, channels = 1;
  bool clahe = false; int tile_w = 64, tile_h = 64; float clip = 40.0f;
  if(argc > 1) W = atoi(argv[1]);
  if(argc > 2) H = atoi(argv[2]);
  if(argc > 3) channels = atoi(argv[3]);
  if(argc > 4) clahe = atoi(argv[4]) != 0;
  if(argc > 5) tile_w = atoi(argv[5]);
  if(argc > 6) tile_h = atoi(argv[6]);
  if(argc > 7) clip = atof(argv[7]);

  int Npix = W * H;
  size_t total = size_t(Npix) * channels;
  std::vector<float> h_img(total);
  // Fill with horizontal gradient + some noise
  for(int y=0;y<H;++y){
    for(int x=0;x<W;++x){
      float v = float(x) / float(W-1);
      for(int c=0;c<channels;++c) h_img[(y*W + x)*channels + c] = v;
    }
  }

  float* d_img = nullptr;
  CUDA_CHECK(cudaMalloc(&d_img, total * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_img, h_img.data(), total * sizeof(float), cudaMemcpyHostToDevice));

  // Warmup
  host_histogram_equalization(d_img, W, H, clahe, channels, tile_w, tile_h, clip);

  // timed runs
  const int runs = 5;
  cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for(int i=0;i<runs;++i) host_histogram_equalization(d_img, W, H, clahe, channels, tile_w, tile_h, clip);
  CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  printf("Histogram equalization: %d x %d x %d, clahe=%d, avg=%.3f ms\n", W, H, channels, clahe?1:0, ms / runs);

  // copy back one pixel sample
  CUDA_CHECK(cudaMemcpy(h_img.data(), d_img, std::min<size_t>(10, total) * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Sample pixels:\n");
  for(int i=0;i<std::min<size_t>(10, total); ++i) printf("%.3f ", h_img[i]);
  printf("\n");

  cudaFree(d_img);
  return 0;
}
