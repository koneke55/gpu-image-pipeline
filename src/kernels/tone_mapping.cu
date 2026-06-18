#include "../../include/kernels/tone_mapping.cuh"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <cmath>

struct LogFunctor{ __host__ __device__ float operator()(const float& v) const { return logf(1e-6f + v); } };

float host_reinhard_tonemap(float* d_luminance, int N, float exposure){
  thrust::device_ptr<float> dev_ptr(d_luminance);
  // compute log-average luminance
  thrust::device_vector<float> tmp(dev_ptr, dev_ptr+N);
  thrust::transform(tmp.begin(), tmp.end(), tmp.begin(), LogFunctor());
  float sum = thrust::reduce(tmp.begin(), tmp.end(), 0.0f, thrust::plus<float>());
  float log_avg = expf(sum / N);
  // apply exposure
  thrust::transform(dev_ptr, dev_ptr+N, dev_ptr, [=] __host__ __device__ (float L){ return L * exposure / (1.0f + L/log_avg); });
  return log_avg;
}
