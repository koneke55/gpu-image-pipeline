#pragma once
#include <cuda_runtime.h>

namespace utils {

struct GPUTimer{
  GPUTimer();
  ~GPUTimer();
  void start();
  float stop();
private:
  cudaEvent_t start_, stop_;
};

struct CPUTimer{
  CPUTimer();
  ~CPUTimer();
};

}
