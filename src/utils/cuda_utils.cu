#include "../../include/utils/cuda_utils.cuh"
#include <iostream>

namespace cuda_utils {
void print_device_info(){
  int dev; CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::cout << "CUDA Device: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")\n";
  std::cout << "TotalGlobalMem: " << prop.totalGlobalMem << "\n";
}
}
