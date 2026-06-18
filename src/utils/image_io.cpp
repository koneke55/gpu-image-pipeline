#include "../../include/utils/image_io.hpp"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../..//include/utils/stb_image.h"
#include "../../include/utils/stb_image_write.h"
#include <vector>
#include <algorithm>

namespace io {
bool load_image_float(const char* path, std::vector<float>& out, int& W, int& H, int& C){
  int w=0,h=0,c=0;
  float* data = stbi_loadf(path,&w,&h,&c,0);
  if(!data) return false;
  W=w;H=h;C=c;
  out.assign(data, data + (size_t)w*h*c);
  stbi_image_free(data);
  return true;
}

bool save_image_float(const char* path, const std::vector<float>& img, int W, int H, int C){
  std::vector<unsigned char> out(W*H*C);
  for(size_t i=0;i<img.size();++i){ float v = img[i]; v = std::min(1.0f,std::max(0.0f,v)); out[i] = (unsigned char)(v*255.0f); }
  return stbi_write_png(path,W,H,C,out.data(),W*C)!=0;
}
}
