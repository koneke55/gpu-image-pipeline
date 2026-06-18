#pragma once
#include "cuda_utils.cuh"

namespace mem {
struct UnifiedImage{
  int width, height, channels;
  float* data = nullptr;
  UnifiedImage(int w,int h,int c);
  ~UnifiedImage();
  float* ptr(){ return data; }
};
}
