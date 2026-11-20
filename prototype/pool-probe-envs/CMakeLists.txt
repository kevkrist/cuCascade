# Throwaway probe. Mirrors the repo's dependency lookup and warning flags.
cmake_minimum_required(VERSION 3.26)
project(pool_probe LANGUAGES CXX CUDA)
message(STATUS "env CUDAARCHS=$ENV{CUDAARCHS}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Forced: the pixi envs preset an arch list (CUDAARCHS) that, under CUDA 12, has no image a
# Blackwell (sm_120) GPU can load.
set(CMAKE_CUDA_ARCHITECTURES 75 86 120)

find_package(CUDAToolkit REQUIRED)
find_package(rmm REQUIRED CONFIG)

add_executable(pool_probe probe.cpp kernels.cu)
target_include_directories(pool_probe PRIVATE ${REPO_INCLUDE})
target_link_libraries(pool_probe PRIVATE rmm::rmm CUDA::cudart_static)
set_source_files_properties(
  probe.cpp
  PROPERTIES
    COMPILE_OPTIONS
    "-Wall;-Wextra;-Wpedantic;-Wcast-align;-Wunused;-Wconversion;-Wsign-conversion;-Wdouble-promotion;-Wformat=2;-Wimplicit-fallthrough;-Werror"
)
