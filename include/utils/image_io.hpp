#pragma once
#include <vector>

namespace io {
bool load_image_float(const char* path, std::vector<float>& out, int& W, int& H, int& C);
bool save_image_float(const char* path, const std::vector<float>& img, int W, int H, int C);
}
