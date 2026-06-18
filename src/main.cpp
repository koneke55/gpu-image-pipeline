#include <iostream>
#include <vector>
#include "../include/utils/cuda_utils.cuh"
#include "../include/utils/image_io.hpp"
#include "../include/kernels/gaussian_blur.cuh"
#include "../include/kernels/sobel_edge_detection.cuh"
#include "../include/kernels/color_space_conversion.cuh"

int main(int argc, char** argv){
  std::cout << "cuda-image-processing CLI (placeholder)\n";
  cuda_utils::print_device_info();
  std::cout << "Run with --help to see options (not yet implemented)." << std::endl;
  return 0;
}
