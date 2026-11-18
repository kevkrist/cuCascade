// Throwaway probe: HostNuma cudaMemPool on this machine.
#include <cuda_runtime.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
static constexpr size_t MiB_ = 1 << 20;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::printf("  FAIL %s -> %s (%d)\n", #x, cudaGetErrorName(e_), (int)e_); std::exit(1);} } while (0)
#define TRY(x) [&]{ cudaError_t e_ = (x); std::printf("  %-60s -> %s\n", #x, cudaGetErrorName(e_)); return e_; }()

__global__ void verify_and_flip(uint32_t* p, size_t n, uint32_t seed, int* bad) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    if (p[i] != (uint32_t)i * seed) atomicAdd(bad, 1);
    p[i] = ~((uint32_t)i * seed);
  }
}

static size_t pool_attr(cudaMemPool_t p, cudaMemPoolAttr a) { uint64_t v = 0; CK(cudaMemPoolGetAttribute(p, a, &v)); return v; }
static void print_usage(cudaMemPool_t p, const char* tag) {
  std::printf("  [%s] reserved=%zu MiB used=%zu MiB\n", tag, pool_attr(p, cudaMemPoolAttrReservedMemCurrent) >> 20,
              pool_attr(p, cudaMemPoolAttrUsedMemCurrent) >> 20);
}

static cudaMemPool_t make_pool(int numa, size_t max_size, bool set_access, int ndev) {
  cudaMemPoolProps props{};
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypeNone;
  props.location.type = cudaMemLocationTypeHostNuma;
  props.location.id = numa;
  props.maxSize = max_size;
  cudaMemPool_t pool{};
  CK(cudaMemPoolCreate(&pool, &props));
  uint64_t thr = UINT64_MAX;
  CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
  if (set_access) {
    std::vector<cudaMemAccessDesc> d(ndev);
    for (int i = 0; i < ndev; ++i) { d[i].location = {cudaMemLocationTypeDevice, i}; d[i].flags = cudaMemAccessFlagsProtReadWrite; }
    CK(cudaMemPoolSetAccess(pool, d.data(), d.size()));
  }
  return pool;
}

static void gpu_roundtrip(uint32_t* p, size_t n, int dev, cudaStream_t s, uint32_t seed) {
  for (size_t i = 0; i < n; ++i) p[i] = (uint32_t)i * seed;  // host write
  CK(cudaSetDevice(dev));
  int* bad; CK(cudaMallocManaged(&bad, sizeof(int))); *bad = 0;
  verify_and_flip<<<256, 256, 0, s>>>(p, n, seed, bad);
  CK(cudaGetLastError()); CK(cudaStreamSynchronize(s));
  size_t host_bad = 0;
  for (size_t i = 0; i < n; ++i) host_bad += p[i] != ~((uint32_t)i * seed);
  std::printf("  dev%d: GPU saw %d bad host-written words; host saw %zu bad GPU-written words\n", dev, *bad, host_bad);
  CK(cudaFree(bad));
}

static void check_numa(void* p, size_t bytes) {
  long pg = sysconf(_SC_PAGESIZE);
  size_t npages = bytes / pg;
  std::vector<void*> pages(npages); std::vector<int> status(npages, -99);
  for (size_t i = 0; i < npages; ++i) pages[i] = (char*)p + i * pg;
  long r = syscall(SYS_move_pages, 0, npages, pages.data(), nullptr, status.data(), 0);
  int hist[8] = {0}; int neg = 0;
  for (int s : status) { if (s >= 0 && s < 8) hist[s]++; else neg++; }
  std::printf("  move_pages rc=%ld pages=%zu node0=%d node1=%d err/other=%d (first status=%d)\n", r, npages, hist[0], hist[1], neg, status[0]);
}

static double bw_gbs(void* host, void* dev, size_t bytes, cudaStream_t s) {
  CK(cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, s)); CK(cudaStreamSynchronize(s));
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 5; ++i) CK(cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, s));
  CK(cudaStreamSynchronize(s));
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return 5.0 * bytes / sec / 1e9;
}

static int no_access_child(int ndev);
int main(int argc, char** argv) {
  if (argc > 1) { int nd=0; cudaGetDeviceCount(&nd); return no_access_child(nd); }
  int rt = 0, drv = 0, ndev = 0;
  CK(cudaRuntimeGetVersion(&rt)); CK(cudaDriverGetVersion(&drv)); CK(cudaGetDeviceCount(&ndev));
  std::printf("runtime=%d driver=%d devices=%d\n", rt, drv, ndev);
  for (int d = 0; d < ndev; ++d) {
    int a142 = -1, a144 = -1, numa = -1, cc = 0;
    cudaDeviceGetAttribute(&a142, (cudaDeviceAttr)142, d);
    cudaDeviceGetAttribute(&a144, (cudaDeviceAttr)144, d); cudaGetLastError();
    cudaDeviceGetAttribute(&numa, cudaDevAttrHostNumaId, d); cudaGetLastError();
    cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, d);
    std::printf("dev%d sm_%d: HostNumaMemoryPoolsSupported(142)=%d HostMemoryPoolsSupported(144)=%d hostNumaId=%d\n", d, cc, a142, a144, numa);
  }

  std::printf("\n== invalid locations ==\n");
  {
    cudaMemPoolProps props{}; props.allocType = cudaMemAllocationTypePinned; cudaMemPool_t p{};
    props.location = {cudaMemLocationTypeHostNumaCurrent, 0};
    if (TRY(cudaMemPoolCreate(&p, &props)) == cudaSuccess) cudaMemPoolDestroy(p);
    cudaGetLastError();
    props.location = {cudaMemLocationTypeHostNuma, 7};
    if (TRY(cudaMemPoolCreate(&p, &props)) == cudaSuccess) cudaMemPoolDestroy(p);
    cudaGetLastError();
#if CUDART_VERSION >= 13000
    props.location = {cudaMemLocationTypeHost, 0};
    if (TRY(cudaMemPoolCreate(&p, &props)) == cudaSuccess) { std::printf("  (Host pool created)\n"); cudaMemPoolDestroy(p); }
    cudaGetLastError();
#endif
  }

  const size_t MiB = 1 << 20, max_size = 256 * MiB, n_bytes = 64 * MiB, n = n_bytes / 4;
  std::printf("\n== pool: HostNuma 0, maxSize=256MiB, threshold=UINT64_MAX, access all devs ==\n");
  CK(cudaSetDevice(0));
  cudaMemPool_t pool = make_pool(0, max_size, true, ndev);
  for (int d = 0; d < ndev; ++d) {
    cudaMemLocation loc{cudaMemLocationTypeDevice, d}; cudaMemAccessFlags f{};
    CK(cudaMemPoolGetAccess(&f, pool, &loc)); std::printf("  GetAccess dev%d = %d\n", d, (int)f);
  }
  { cudaMemLocation loc{cudaMemLocationTypeHost, 0}; cudaMemAccessFlags f{};
    cudaError_t e = cudaMemPoolGetAccess(&f, pool, &loc); cudaGetLastError();
    std::printf("  GetAccess host -> %s flags=%d\n", cudaGetErrorName(e), (int)f); }
  std::vector<cudaStream_t> st(ndev);
  for (int d = 0; d < ndev; ++d) { CK(cudaSetDevice(d)); CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking)); }
  CK(cudaSetDevice(0));

  void* a = nullptr;
  CK(cudaMallocFromPoolAsync(&a, n_bytes, pool, st[0]));
  CK(cudaStreamSynchronize(st[0]));
  std::printf("  alloc on dev0 stream: %p (align 2MiB? %d)\n", a, (int)(((uintptr_t)a % (2 * MiB)) == 0));
  print_usage(pool, "after 64MiB alloc");
  cudaPointerAttributes pa{};
  CK(cudaPointerGetAttributes(&pa, a));
  std::printf("  ptrattr type=%d device=%d devicePointer=%p hostPointer=%p (same VA: %d)\n", (int)pa.type, pa.device,
              pa.devicePointer, pa.hostPointer, (int)(pa.devicePointer == a && pa.hostPointer == a));
  check_numa(a, n_bytes);  // before first host touch? already touched? (not yet touched by host)
  for (int d = 0; d < ndev; ++d) gpu_roundtrip((uint32_t*)a, n, d, st[d], 2654435761u + d);
  check_numa(a, n_bytes);

  std::printf("\n== alloc on dev1 stream, used from dev0 ==\n");
  if (ndev > 1) {
    void* b = nullptr; CK(cudaSetDevice(1));
    CK(cudaMallocFromPoolAsync(&b, 4 * MiB, pool, st[1])); CK(cudaStreamSynchronize(st[1]));
    gpu_roundtrip((uint32_t*)b, MiB, 0, st[0], 7);
    CK(cudaFreeAsync(b, st[1])); CK(cudaStreamSynchronize(st[1])); CK(cudaSetDevice(0));
  }

  std::printf("\n== H2D bandwidth, 64MiB ==\n");
  {
    void* d = nullptr; CK(cudaMalloc(&d, n_bytes));
    void* pageable = std::malloc(n_bytes); std::memset(pageable, 1, n_bytes);
    void* hostalloc = nullptr; CK(cudaHostAlloc(&hostalloc, n_bytes, cudaHostAllocPortable));
    std::printf("  pool=%.1f GB/s  cudaHostAlloc=%.1f GB/s  pageable=%.1f GB/s\n", bw_gbs(a, d, n_bytes, st[0]),
                bw_gbs(hostalloc, d, n_bytes, st[0]), bw_gbs(pageable, d, n_bytes, st[0]));
    CK(cudaFree(d)); CK(cudaFreeHost(hostalloc)); std::free(pageable);
  }

  std::printf("\n== same-stream reuse ==\n");
  {
    void* x = nullptr; void* y = nullptr;
    CK(cudaMallocFromPoolAsync(&x, 8 * MiB, pool, st[0])); CK(cudaFreeAsync(x, st[0]));
    CK(cudaMallocFromPoolAsync(&y, 8 * MiB, pool, st[0]));
    std::printf("  reused same ptr without sync: %d\n", (int)(x == y));
    CK(cudaFreeAsync(y, st[0])); CK(cudaStreamSynchronize(st[0]));
    void* z = nullptr; CK(cudaMallocFromPoolAsync(&z, 1, pool, st[0])); CK(cudaStreamSynchronize(st[0]));
    std::printf("  1-byte alloc %p\n", z); print_usage(pool, "after 1B alloc");
    CK(cudaFreeAsync(z, st[0])); CK(cudaStreamSynchronize(st[0]));
  }

  std::printf("\n== hit maxSize ==\n");
  {
    std::vector<void*> v; size_t total = n_bytes; cudaError_t e = cudaSuccess;
    for (;;) { void* q = nullptr; e = cudaMallocFromPoolAsync(&q, 32 * MiB, pool, st[0]); if (e != cudaSuccess) break; v.push_back(q); total += 32 * MiB; }
    std::printf("  failed after %zu MiB live: %s; peek=%s get=%s peek-after=%s\n", total >> 20, cudaGetErrorName(e),
                cudaGetErrorName(cudaPeekAtLastError()), cudaGetErrorName(cudaGetLastError()), cudaGetErrorName(cudaPeekAtLastError()));
    void* big = nullptr; e = cudaMallocFromPoolAsync(&big, 512 * MiB, pool, st[0]); cudaGetLastError();
    std::printf("  single 512MiB alloc (> maxSize): %s\n", cudaGetErrorName(e));
    print_usage(pool, "at limit");
    for (void* q : v) CK(cudaFreeAsync(q, st[0]));
    CK(cudaStreamSynchronize(st[0])); print_usage(pool, "freed, before trim");
    void* q = nullptr; CK(cudaMallocFromPoolAsync(&q, 32 * MiB, pool, st[0])); CK(cudaStreamSynchronize(st[0]));
    std::printf("  context still usable after OOM: yes\n");
    CK(cudaFreeAsync(q, st[0])); CK(cudaStreamSynchronize(st[0]));
  }

  std::printf("\n== trim ==\n");
  CK(cudaMemPoolTrimTo(pool, 0)); print_usage(pool, "trim(0) with 64MiB live");
  CK(cudaFreeAsync(a, st[0])); CK(cudaStreamSynchronize(st[0]));
  print_usage(pool, "freed all, threshold=MAX");
  CK(cudaMemPoolTrimTo(pool, 0)); print_usage(pool, "trim(0), none live");

  std::printf("\n== destroy with live allocation ==\n");
  {
    void* q = nullptr; CK(cudaMallocFromPoolAsync(&q, 16 * MiB, pool, st[0])); CK(cudaStreamSynchronize(st[0]));
    TRY(cudaMemPoolDestroy(pool));
    std::memset(q, 3, 16 * MiB);
    std::printf("  host write to live alloc after destroy: ok\n");
    TRY(cudaFreeAsync(q, st[0])); TRY(cudaStreamSynchronize(st[0]));
  }

  std::printf("\n== kernel access WITHOUT cudaMemPoolSetAccess (child process) ==\n");
  fflush(stdout);
  { std::string cmd = std::string(argv[0]) + " child"; int rc = std::system(cmd.c_str()); std::printf("  child rc=%d\n", rc); }

  std::printf("\nDONE\n");
}

static int no_access_child(int ndev) {
  CK(cudaSetDevice(0)); cudaMemPool_t p2 = make_pool(0, 0, false, ndev);
  cudaStream_t s; CK(cudaStreamCreate(&s)); void* q = nullptr;
  CK(cudaMallocFromPoolAsync(&q, MiB_, p2, s)); CK(cudaStreamSynchronize(s));
  for (size_t i = 0; i < MiB_ / 4; ++i) ((uint32_t*)q)[i] = (uint32_t)i * 3;
  cudaPointerAttributes pa{}; cudaError_t e = cudaPointerGetAttributes(&pa, q);
  std::printf("  ptrattr -> %s type=%d devicePointer=%p\n", cudaGetErrorName(e), (int)pa.type, pa.devicePointer);
  void* d; CK(cudaMalloc(&d, MiB_)); e = cudaMemcpy(d, q, MiB_, cudaMemcpyHostToDevice);
  std::printf("  memcpy H2D -> %s\n", cudaGetErrorName(e)); std::fflush(stdout);
  int* bad; CK(cudaMallocManaged(&bad, sizeof(int))); *bad = 0;
  verify_and_flip<<<64, 256, 0, s>>>((uint32_t*)q, MiB_ / 4, 3, bad);
  e = cudaStreamSynchronize(s);
  std::printf("  kernel sync -> %s, bad=%d\n", cudaGetErrorName(e), e == cudaSuccess ? *bad : -1);
  return 0;
}
