add_executable(tests_main test_main.cpp test_gaussian.cpp test_sobel.cpp test_histogram.cpp)
target_link_libraries(tests_main PRIVATE Catch2::Catch2WithMain CUDA::cudart)
target_include_directories(tests_main PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME image_tests COMMAND tests_main)
