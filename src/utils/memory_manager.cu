#include "../../include/utils/memory_manager.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace mem {

UnifiedImage::UnifiedImage(int w, int h, int c): width(w), height(h), channels(c){
  size_t size = (size_t)w*h*c*sizeof(float);
  CUDA_CHECK(cudaMallocManaged(&data, size));
}
UnifiedImage::~UnifiedImage(){ if(data) cudaFree(data); }

}
