#pragma once
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA error " << cudaGetErrorString(err) << " at "   \
                << __FILE__ << ":" << __LINE__ << std::endl;             \
      std::terminate();                                                    \
    }                                                                      \
  } while (0)

namespace cuda_utils {
  inline void print_device_info(){
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "CUDA Device: " << prop.name << " (SM " << prop.major << prop.minor << ")\n";
    std::cout << "TotalGlobalMem: " << prop.totalGlobalMem << "\n";
  }
}
