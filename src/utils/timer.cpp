#include "../../include/utils/timer.hpp"
#include <cuda_runtime.h>

namespace utils {

GPUTimer::GPUTimer(){
  cudaEventCreate(&start_); cudaEventCreate(&stop_);
}
GPUTimer::~GPUTimer(){
  cudaEventDestroy(start_); cudaEventDestroy(stop_);
}
void GPUTimer::start(){ cudaEventRecord(start_); }
float GPUTimer::stop(){ cudaEventRecord(stop_); cudaEventSynchronize(stop_); float ms; cudaEventElapsedTime(&ms,start_,stop_); return ms; }

CPUTimer::CPUTimer(){}
CPUTimer::~CPUTimer(){}

}
