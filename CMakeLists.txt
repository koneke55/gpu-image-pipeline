cmake_minimum_required(VERSION 3.20)
project(cuda-image-processing LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDAToolkit REQUIRED)

set(CUDA_ARCH_FLAGS "-gencode=arch=compute_89,code=sm_89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS} -lineinfo --extended-lambda")

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE CPP_SOURCES src/*.cpp)
file(GLOB_RECURSE CU_SOURCES src/*.cu)

add_executable(cuda_image_processing ${CPP_SOURCES} ${CU_SOURCES})
target_link_libraries(cuda_image_processing PUBLIC CUDA::cudart)
target_include_directories(cuda_image_processing PUBLIC include)

# Example/test runner for histogram equalization (built by default)
add_executable(hist_eq_test src/tools/hist_eq_test.cu)
target_link_libraries(hist_eq_test PUBLIC CUDA::cudart)
target_include_directories(hist_eq_test PUBLIC include)

option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
  )
  FetchContent_MakeAvailable(Catch2)
  enable_testing()
  add_subdirectory(tests)
endif()

install(TARGETS cuda_image_processing DESTINATION bin)
