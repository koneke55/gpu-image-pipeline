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

namespace cuda_utils { void print_device_info(); }
