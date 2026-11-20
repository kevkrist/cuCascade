// Throwaway probe kernel: verify a pattern written by the host, then flip it.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

__global__ void verify_and_flip(uint32_t* p, size_t n, uint32_t seed, int* bad)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    if (p[i] != static_cast<uint32_t>(i) * seed) { atomicAdd(bad, 1); }
    p[i] = ~(static_cast<uint32_t>(i) * seed);
  }
}

cudaError_t launch_verify_and_flip(uint32_t* p, size_t n, uint32_t seed, int* bad, cudaStream_t s)
{
  verify_and_flip<<<64, 256, 0, s>>>(p, n, seed, bad);
  return cudaGetLastError();
}
