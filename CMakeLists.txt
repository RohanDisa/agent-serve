cmake_minimum_required(VERSION 3.16)
project(tinferno LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W3 /permissive- /EHsc /arch:AVX2)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
  add_compile_options(-Wall -Wextra -O3)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx2" HAS_AVX2)
  if(HAS_AVX2)
    add_compile_options(-mavx2 -mfma -mf16c)
  endif()
endif()

set(TINF_LIB_SRC
  src/tensor.cpp
  src/gguf.cpp
  src/quant.cpp
  src/matmul.cpp
  src/attention.cpp
  src/layers.cpp
  src/kvcache.cpp
  src/model.cpp
  src/sampler.cpp
  src/tokenizer.cpp
)

add_library(tinferno_lib STATIC ${TINF_LIB_SRC})
target_include_directories(tinferno_lib PUBLIC src)

find_package(Threads REQUIRED)
target_link_libraries(tinferno_lib PUBLIC Threads::Threads)

add_executable(tinferno src/main.cpp)
target_link_libraries(tinferno PRIVATE tinferno_lib)

add_executable(stream bench/stream.c)
target_compile_definitions(stream PRIVATE STREAM_ARRAY_SIZE=100000000)
if(MSVC)
  target_compile_options(stream PRIVATE /O2 /openmp /arch:AVX2)
  target_link_options(stream PRIVATE /openmp)
else()
  target_compile_options(stream PRIVATE -O3 -march=native -fopenmp)
  target_link_options(stream PRIVATE -fopenmp)
endif()
add_executable(bench_matmul bench/bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE tinferno_lib)

add_executable(bench_e2e bench/bench_e2e.cpp)
target_link_libraries(bench_e2e PRIVATE tinferno_lib)

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE tinferno_lib)

add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE tinferno_lib)

enable_testing()
add_test(NAME ops COMMAND test_ops)
add_test(NAME correctness COMMAND test_correctness)
