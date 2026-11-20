// Throwaway: which GPUs get touched by pinned-pool construction, route A vs B.
// usage: grant_probe <mode> [alloc] [own]     env: WARM=1 (cuInit before timing), LISTED=1
// modes: C (CCCL ctor only), A (ctor + disable unlisted), B (rt create + CCCL enable listed),
//        Brt (rt create + cudaMemPoolSetAccess), Bdrv (driver cuMemPoolCreate + CCCL enable)
#include <cuda/devices>
#include <cuda/memory_pool>

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <string>

namespace {
void* g_drv = nullptr;
template <class F>
F* sym(const char* n)
{
  if (!g_drv) { g_drv = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL); }
  void* p = dlsym(g_drv, n);
  if (!p) { std::printf("missing %s\n", n); std::exit(2); }
  return reinterpret_cast<F*>(p);
}
int ndev()
{
  int n = 0;
  sym<CUresult(int*)>("cuDeviceGetCount")(&n);
  return n;
}
void report(const char* stage)
{
  std::printf("== %s\n", stage);
  const int n = ndev();
  for (int d = 0; d < n; ++d) {
    unsigned flags = 0;
    int active     = -1;
    CUresult r = sym<CUresult(CUdevice, unsigned*, int*)>("cuDevicePrimaryCtxGetState")(d, &flags, &active);
    char name[128] = {};
    sym<CUresult(char*, int, CUdevice)>("cuDeviceGetName")(name, 127, d);
    CUuuid u{};
    sym<CUresult(CUuuid*, CUdevice)>("cuDeviceGetUuid_v2")(&u, d);
    std::printf("  cuda dev %d (%s, uuid GPU-%02x%02x%02x%02x): primary ctx active=%d (rc=%d)\n", d, name,
                static_cast<unsigned char>(u.bytes[0]), static_cast<unsigned char>(u.bytes[1]),
                static_cast<unsigned char>(u.bytes[2]), static_cast<unsigned char>(u.bytes[3]), active,
                static_cast<int>(r));
  }
  std::fflush(stdout);
  std::string cmd = "timeout 20 nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | grep '^" +
                    std::to_string(getpid()) + ",' | sed 's/^/  smi-app: /'; "
                    "timeout 20 nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader | sed 's/^/  smi-gpu: /'";
  if (std::system(cmd.c_str()) != 0) { std::printf("  (nvidia-smi failed)\n"); }
  std::fflush(stdout);
}
double ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
  return std::chrono::duration<double, std::milli>(b - a).count();
}
}  // namespace

int main(int argc, char** argv)
try {
  const std::string mode = argc > 1 ? argv[1] : "A";
  bool do_alloc = false, do_own = false, quiet = false;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "alloc")) { do_alloc = true; }
    if (!std::strcmp(argv[i], "own")) { do_own = true; }
    if (!std::strcmp(argv[i], "quiet")) { quiet = true; }
  }
  const int listed = std::getenv("LISTED") ? std::atoi(std::getenv("LISTED")) : 1;
  const bool warm  = std::getenv("WARM") != nullptr;
  if (warm) { sym<CUresult(unsigned)>("cuInit")(0); }
  std::printf("mode=%s pid=%d listed={%d} warm=%d\n", mode.c_str(), getpid(), listed, warm);

  cuda::memory_pool_properties props{};
  props.max_pool_size = std::size_t{256} << 20;
  cudaMemPool_t raw{};
  std::optional<cuda::shared_pinned_memory_pool> pool;

  const auto t0 = std::chrono::steady_clock::now();
  auto t1 = t0;
  if (mode == "C" || mode == "A") {
    pool.emplace(0, props);
    t1 = std::chrono::steady_clock::now();
    if (mode == "A") {
      for (int d = 0; d < cuda::devices.size(); ++d) {
        if (d != listed) { pool->disable_access_from(cuda::device_ref{d}); }
      }
    }
  } else {
    if (mode == "Bdrv") {
      CUmemPoolProps cp{};
      cp.allocType     = CU_MEM_ALLOCATION_TYPE_PINNED;
      cp.handleTypes   = CU_MEM_HANDLE_TYPE_NONE;
      cp.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      cp.location.id   = 0;
      cp.maxSize       = props.max_pool_size;
      sym<CUresult(unsigned)>("cuInit")(0);
      CUmemoryPool h{};
      CUresult r = sym<CUresult(CUmemoryPool*, const CUmemPoolProps*)>("cuMemPoolCreate")(&h, &cp);
      if (r != CUDA_SUCCESS) { std::printf("cuMemPoolCreate rc=%d\n", static_cast<int>(r)); return 1; }
      raw = reinterpret_cast<cudaMemPool_t>(h);
    } else {
      cudaMemPoolProps cp{};
      cp.allocType     = cudaMemAllocationTypePinned;
      cp.handleTypes   = cudaMemHandleTypeNone;
      cp.location.type = cudaMemLocationTypeHostNuma;
      cp.location.id   = 0;
      cp.maxSize       = props.max_pool_size;
      cudaError_t e = cudaMemPoolCreate(&raw, &cp);
      if (e != cudaSuccess) { std::printf("cudaMemPoolCreate: %s\n", cudaGetErrorString(e)); return 1; }
    }
    t1 = std::chrono::steady_clock::now();
    if (mode == "Brt") {
      cudaMemAccessDesc desc{};
      desc.location.type = cudaMemLocationTypeDevice;
      desc.location.id   = listed;
      desc.flags         = cudaMemAccessFlagsProtReadWrite;
      cudaError_t e      = cudaMemPoolSetAccess(raw, &desc, 1);
      if (e != cudaSuccess) { std::printf("cudaMemPoolSetAccess: %s\n", cudaGetErrorString(e)); return 1; }
      pool.emplace(cuda::shared_pinned_memory_pool::from_native_handle(raw));
    } else {
      pool.emplace(cuda::shared_pinned_memory_pool::from_native_handle(raw));
      pool->enable_access_from(cuda::device_ref{listed});
    }
  }
  const auto t2 = std::chrono::steady_clock::now();
  std::printf("TIME mode=%s warm=%d create_ms=%.3f narrow_ms=%.3f total_ms=%.3f\n", mode.c_str(), warm, ms(t0, t1),
              ms(t1, t2), ms(t0, t2));
  if (quiet) { return 0; }
  report("after construction + narrowing, before any allocation");
  for (int d = 0; d < cuda::devices.size(); ++d) {
    std::printf("  is_accessible_from(%d)=%d\n", d, pool->is_accessible_from(cuda::device_ref{d}));
  }

  if (do_alloc) {
    void* p = pool->allocate_sync(std::size_t{1} << 20);
    std::memset(p, 1, std::size_t{1} << 20);
    report("after allocate_sync(1MiB) from pool (no cudaSetDevice by us)");
    cudaError_t e = cudaSetDevice(listed);
    void* dptr    = nullptr;
    if (e == cudaSuccess) { e = cudaMalloc(&dptr, std::size_t{1} << 20); }
    if (e == cudaSuccess) { e = cudaMemcpy(dptr, p, std::size_t{1} << 20, cudaMemcpyHostToDevice); }
    std::printf("  copy pool mem -> listed device %d: %s\n", listed, cudaGetErrorString(e));
    report("after cudaSetDevice(listed) + H2D copy");
    cudaFree(dptr);
    pool->deallocate_sync(p, std::size_t{1} << 20);
  }
  if (do_own && raw != nullptr) {
    auto copy = *pool;
    pool.reset();
    std::size_t v = 0;
    std::printf("== ownership: one handle dropped, copy alive: GetAttribute -> %s\n",
                cudaGetErrorString(cudaMemPoolGetAttribute(raw, cudaMemPoolAttrReservedMemCurrent, &v)));
    { auto sink = std::move(copy); }
    std::fflush(stdout);
    std::printf("== ownership: last handle dropped: cudaMemPoolDestroy(raw) -> ");
    std::fflush(stdout);
    std::printf("%s (success would mean CCCL had NOT destroyed it)\n", cudaGetErrorString(cudaMemPoolDestroy(raw)));
  }
  return 0;
} catch (const std::exception& e) {
  std::printf("EXCEPTION: %s\n", e.what());
  return 1;
}
