// Throwaway: private-stream pinned pool wrapper feasibility.
// usage: stream_probe [raw] [nophaseb]      env: LISTED=<cuda dev> (default 1), ITERS (default 2000)
//   default: private stream = cuda::stream{device_ref{LISTED}}
//   raw:     private stream = cudaSetDevice(LISTED) on a helper thread + cudaStreamCreateWithFlags
// Phase A never makes any device current on a calling thread (bare std::threads, driver API only).
// Phase B makes the "other" device current (cudaSetDevice) on the calling threads.
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

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
void report(const char* stage, bool smi = true)
{
  std::printf("== %s\n", stage);
  int n = 0;
  sym<CUresult(int*)>("cuDeviceGetCount")(&n);
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
  if (!smi) { return; }
  std::string cmd = "timeout 20 nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | grep '^" +
                    std::to_string(getpid()) + ",' | sed 's/^/  smi-app: /'";
  (void)std::system(cmd.c_str());
  std::fflush(stdout);
}
void current_ctx(const char* who)
{
  CUcontext c = nullptr;
  CUresult r  = sym<CUresult(CUcontext*)>("cuCtxGetCurrent")(&c);
  std::printf("  [%s] cuCtxGetCurrent rc=%d ctx=%p\n", who, static_cast<int>(r), static_cast<void*>(c));
}
void pool_stats(const char* stage, cuda::shared_pinned_memory_pool& pool)
{
  namespace a = cuda::memory_pool_attributes;
  std::printf("  pool[%s]: reserved=%zu KiB used=%zu KiB reserved_high=%zu KiB threshold=%zu\n", stage,
              static_cast<std::size_t>(pool.attribute(a::reserved_mem_current)) >> 10,
              static_cast<std::size_t>(pool.attribute(a::used_mem_current)) >> 10,
              static_cast<std::size_t>(pool.attribute(a::reserved_mem_high)) >> 10,
              static_cast<std::size_t>(pool.attribute(a::release_threshold)));
  std::fflush(stdout);
}

std::mutex g_err_mutex;
std::string g_first_err;
void note_err(const std::string& s)
{
  std::lock_guard lk(g_err_mutex);
  if (g_first_err.empty()) { g_first_err = s; }
}

// The wrapper's synchronous paths, on the shared private stream.
// other_dev >= 0: make that device current on this thread first (runtime API).
long worker(cuda::shared_pinned_memory_pool& pool, cuda::stream_ref s, int tid, int iters, int other_dev)
{
  long errors = 0;
  if (other_dev >= 0) {
    if (cudaError_t e = cudaSetDevice(other_dev); e != cudaSuccess) {
      note_err(std::string("cudaSetDevice(other): ") + cudaGetErrorString(e));
      return iters;
    }
  }
  auto* free_async = sym<CUresult(CUdeviceptr, CUstream)>("cuMemFreeAsync");
  for (int i = 0; i < iters; ++i) {
    const std::size_t bytes = std::size_t{4096} << ((i + tid) % 9);  // 4 KiB .. 1 MiB
    try {
      void* p = pool.allocate(s, bytes, 256);
      s.sync();
      if (p == nullptr) { ++errors; note_err("null pointer"); continue; }
      std::memset(p, tid + 1, bytes);
      if (static_cast<unsigned char*>(p)[bytes - 1] != static_cast<unsigned char>(tid + 1)) { ++errors; }
      if (tid % 2 == 0) {
        pool.deallocate(s, p, bytes, 256);  // noexcept; failures are silent in release
      } else {
        // same driver call CCCL makes, but with the status visible
        CUresult r = free_async(reinterpret_cast<CUdeviceptr>(p), s.get());
        if (r != CUDA_SUCCESS) { ++errors; note_err("cuMemFreeAsync rc=" + std::to_string(static_cast<int>(r))); }
      }
      s.sync();
    } catch (const std::exception& e) {
      ++errors;
      note_err(std::string(typeid(e).name()) + ": " + e.what());
    }
  }
  return errors;
}

void run_phase(const char* name, cuda::shared_pinned_memory_pool& pool, cuda::stream_ref s, int iters, int other_dev)
{
  g_first_err.clear();
  std::atomic<long> errors{0};
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> ts;
  for (int t = 0; t < 8; ++t) {
    ts.emplace_back([&, t] { errors += worker(pool, s, t, iters, other_dev); });
  }
  for (auto& t : ts) { t.join(); }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("PHASE %s: 8 threads x %d iters, other_dev_current=%d: errors=%ld (%.2fs, %.1f us/iter) first_err='%s'\n",
              name, iters, other_dev, errors.load(), sec, sec * 1e6 / iters, g_first_err.c_str());
  pool_stats(name, pool);
}
}  // namespace

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  bool raw = false, phaseb = true;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "raw")) { raw = true; }
    if (!std::strcmp(argv[i], "nophaseb")) { phaseb = false; }
  }
  const int listed = std::getenv("LISTED") ? std::atoi(std::getenv("LISTED")) : 1;
  const int iters  = std::getenv("ITERS") ? std::atoi(std::getenv("ITERS")) : 2000;
  const int other  = listed == 0 ? 1 : 0;
  std::printf("pid=%d listed={%d} raw_stream=%d iters=%d\n", getpid(), listed, raw, iters);

  try {
    cuda::memory_pool_properties props{};
    props.max_pool_size = std::size_t{256} << 20;
    cuda::shared_pinned_memory_pool pool(0, props);
    for (int d = 0; d < cuda::devices.size(); ++d) {
      if (d != listed) { pool.disable_access_from(cuda::device_ref{d}); }
    }
    report("after pool construction + narrowing", false);

    // ---- private stream creation
    cudaStream_t handle{};
    try {
      if (raw) {
        cudaError_t e = cudaSuccess;
        std::thread([&] {
          e = cudaSetDevice(listed);
          if (e == cudaSuccess) { e = cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking); }
        }).join();
        if (e != cudaSuccess) {
          std::printf("STREAM CREATE FAILED (raw): cudaError=%d %s\n", static_cast<int>(e), cudaGetErrorString(e));
          return 3;
        }
      } else {
        handle = cuda::stream{cuda::device_ref{listed}}.release();
      }
    } catch (const cuda::cuda_error& e) {
      std::printf("STREAM CREATE FAILED: cuda::cuda_error status=%d what=%s\n", static_cast<int>(e.status()), e.what());
      report("after failed stream creation", false);
      return 3;
    } catch (const std::exception& e) {
      std::printf("STREAM CREATE FAILED: %s what=%s\n", typeid(e).name(), e.what());
      return 3;
    }
    cuda::stream_ref s{handle};
    current_ctx("main after stream creation");
    report("after private stream creation");

    // ---- pre-grow on the private stream (main thread: no current context)
    pool_stats("before pre-grow", pool);
    {
      const std::size_t grow = std::size_t{48} << 20;
      void* p                = pool.allocate(s, grow, 256);
      s.sync();
      pool_stats("pre-grow allocated", pool);
      pool.deallocate(s, p, grow, 256);
      s.sync();
      pool_stats("pre-grow freed+synced", pool);
      // does a later allocation reuse the reservation (reserved must not rise)?
      void* q = pool.allocate(s, std::size_t{32} << 20, 256);
      s.sync();
      std::memset(q, 7, std::size_t{32} << 20);
      pool_stats("32MiB allocated after pre-grow", pool);
      std::printf("  pre-grow ptr=%p later ptr=%p\n", p, q);
      pool.deallocate(s, q, std::size_t{32} << 20, 256);
      s.sync();
    }

    // ---- phase A: bare threads, nothing current
    std::thread([] { current_ctx("bare thread"); }).join();
    run_phase("A(bare threads)", pool, s, iters, -1);
    report("after phase A (no device ever made current by callers)");

    // ---- destruction checks, phase A flavour: second stream destroyed from a bare thread
    {
      cudaStream_t h2 = raw ? nullptr : cuda::stream{cuda::device_ref{listed}}.release();
      if (h2 != nullptr) {
        CUresult r = CUDA_SUCCESS;
        std::thread([&] { r = sym<CUresult(CUstream)>("cuStreamDestroy_v2")(h2); }).join();
        std::printf("DESTROY from bare thread (no current ctx): cuStreamDestroy rc=%d\n", static_cast<int>(r));
      }
    }

    if (phaseb) {
      if (cudaError_t e = cudaSetDevice(other); e != cudaSuccess) {
        std::printf("main cudaSetDevice(%d): %s\n", other, cudaGetErrorString(e));
      }
      current_ctx("main with other device current");
      // single-threaded on main with other device current
      g_first_err.clear();
      long e1 = worker(pool, s, 0, 200, -1) + worker(pool, s, 1, 200, -1);
      std::printf("PHASE B0 (main thread, dev %d current): errors=%ld first_err='%s'\n", other, e1, g_first_err.c_str());
      run_phase("B(other device current)", pool, s, iters, other);
      report("after phase B");

      // cuda::stream destructor from a thread with the other device current
      if (!raw) {
        std::optional<cuda::stream> owned;
        owned.emplace(cuda::device_ref{listed});
        std::thread([&] {
          cudaSetDevice(other);
          owned.reset();
          int dev = -1;
          cudaGetDevice(&dev);
          std::printf("DESTROY cuda::stream dtor from thread with dev %d current: survived, current dev after=%d\n", other,
                      dev);
        }).join();
      }
      // the private stream itself, destroyed from a thread with the other device current
      std::thread([&] {
        cudaSetDevice(other);
        if (raw) {
          cudaError_t e = cudaStreamDestroy(handle);
          std::printf("DESTROY private stream (cudaStreamDestroy) with dev %d current: rc=%d %s\n", other,
                      static_cast<int>(e), cudaGetErrorString(e));
        } else {
          CUresult r = sym<CUresult(CUstream)>("cuStreamDestroy_v2")(handle);
          std::printf("DESTROY private stream (cuStreamDestroy) with dev %d current: rc=%d\n", other, static_cast<int>(r));
        }
      }).join();
    } else {
      CUresult r = sym<CUresult(CUstream)>("cuStreamDestroy_v2")(handle);
      std::printf("DESTROY private stream from main (no current ctx): rc=%d\n", static_cast<int>(r));
    }
    pool_stats("end", pool);
    report("end", false);
  } catch (const cuda::cuda_error& e) {
    std::printf("EXCEPTION cuda::cuda_error status=%d: %s\n", static_cast<int>(e.status()), e.what());
    return 1;
  } catch (const std::exception& e) {
    std::printf("EXCEPTION %s: %s\n", typeid(e).name(), e.what());
    return 1;
  }
  std::printf("DONE\n");
  return 0;
}
