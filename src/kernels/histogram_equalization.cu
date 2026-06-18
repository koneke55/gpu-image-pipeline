#include "../../include/kernels/histogram_equalization.cuh"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

void host_histogram_equalization(float* d_in, int W, int H, bool clahe){
  int N = W*H;
  const int BIN = 256;
  thrust::device_vector<int> hist(BIN);
  thrust::device_ptr<float> data(d_in);
  // zero hist
  thrust::fill(thrust::device, hist.begin(), hist.end(), 0);
  // build histogram
  thrust::for_each(thrust::device, data, data+N, [=] __device__ (float v){
    int b = min(max(int(v*255.0f),0),255);
    atomicAdd(&hist[b],1);
  });
  // cdf
  thrust::device_vector<int> cdf(BIN);
  thrust::inclusive_scan(thrust::device, hist.begin(), hist.end(), cdf.begin());
  // apply mapping
  thrust::for_each(thrust::device, data, data+N, [=] __device__ (float &v){
    int b = min(max(int(v*255.0f),0),255);
    int c = cdf[b];
    v = float(c) / float(N);
  });
}
