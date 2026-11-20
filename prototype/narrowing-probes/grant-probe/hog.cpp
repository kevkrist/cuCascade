// helper: hold a context on CUDA device <dev> and allocate <mib> MiB; sleep <sec> then exit.
#include <cuda_runtime.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv)
{
  int dev = std::atoi(argv[1]); long mib = std::atol(argv[2]); int sec = std::atoi(argv[3]);
  cudaError_t e = cudaSetDevice(dev);
  void* p = nullptr;
  if (e == cudaSuccess && mib > 0) { e = cudaMalloc(&p, static_cast<size_t>(mib) << 20); }
  size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
  std::printf("hog dev=%d mib=%ld: %s free=%zuMiB total=%zuMiB\n", dev, mib, cudaGetErrorString(e), fr >> 20, tot >> 20);
  std::fflush(stdout);
  sleep(static_cast<unsigned>(sec));
  return 0;
}
