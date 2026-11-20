// Throwaway probe: CCCL shared_pinned_memory_pool behind a shared_resource handle.
// Output avoids pointers and timings so runs from different envs can be diffed.
#include <cucascade/cuda/stream.hpp>

#if __has_include(<rmm/cuda_stream_view.hpp>)
#include <rmm/cuda_stream_view.hpp>
#define PROBE_RMM_STREAM(s) rmm::cuda_stream_view{s}
#else
#define PROBE_RMM_STREAM(s) cuda::stream_ref{s}
#endif
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>
#include <rmm/version_config.hpp>

#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/memory_resource>

#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

cudaError_t launch_verify_and_flip(uint32_t* p, size_t n, uint32_t seed, int* bad, cudaStream_t s);

namespace {

constexpr std::size_t MiB = std::size_t{1} << 20;

void say(const std::string& s)
{
  std::puts(s.c_str());
  std::fflush(stdout);
}
std::string num(std::size_t v) { return std::to_string(v); }
std::string err(cudaError_t e) { return cudaGetErrorName(e); }

template <class F>
std::string outcome(F&& f)
{
  try {
    std::forward<F>(f)();
    return "ok";
  } catch (const cuda::cuda_error& e) {
    return "cuda::cuda_error(" + err(static_cast<cudaError_t>(e.status())) + ")";
  } catch (const std::invalid_argument& e) {
    return std::string("std::invalid_argument(") + e.what() + ")";
  } catch (const std::bad_alloc& e) {
    return std::string("std::bad_alloc family: ") + typeid(e).name();
  } catch (const std::exception& e) {
    return std::string("other exception: ") + typeid(e).name() + ": " + e.what();
  }
}

// The shape the spec proposes: an impl that owns the CCCL pool, shared through shared_resource.
struct pool_impl {
  cuda::shared_pinned_memory_pool pool;

  explicit pool_impl(cuda::shared_pinned_memory_pool p) : pool(std::move(p)) {}

  [[nodiscard]] void* allocate(cuda::stream_ref s, std::size_t bytes, std::size_t align)
  {
    return pool.allocate(s, bytes, align);
  }
  void deallocate(cuda::stream_ref s, void* p, std::size_t bytes, std::size_t align) noexcept
  {
    pool.deallocate(s, p, bytes, align);
  }
  [[nodiscard]] void* allocate_sync(std::size_t bytes, std::size_t align = 256)
  {
    return pool.allocate_sync(bytes, align);
  }
  void deallocate_sync(void* p, std::size_t bytes, std::size_t align = 256) noexcept
  {
    pool.deallocate_sync(p, bytes, align);
  }
  [[nodiscard]] bool operator==(const pool_impl& o) const noexcept { return pool.get() == o.pool.get(); }
  friend constexpr void get_property(const pool_impl&, cuda::mr::device_accessible) noexcept {}
  friend constexpr void get_property(const pool_impl&, cuda::mr::host_accessible) noexcept {}
};
using handle = cuda::mr::shared_resource<pool_impl>;

static_assert(cuda::mr::resource_with<handle, cuda::mr::device_accessible, cuda::mr::host_accessible>);
static_assert(std::is_constructible_v<rmm::host_device_async_resource_ref, handle&>);
static_assert(std::is_copy_constructible_v<handle>);

struct probe_pool {
  cuda::shared_pinned_memory_pool pool;  // second reference to the same pool, for attributes
  handle h;
  explicit probe_pool(cuda::shared_pinned_memory_pool p)
    : pool(p), h(cuda::mr::make_shared_resource<pool_impl>(std::move(p)))
  {}
  [[nodiscard]] std::size_t reserved() const
  {
    return pool.attribute(cuda::memory_pool_attributes::reserved_mem_current);
  }
  [[nodiscard]] std::size_t used() const
  {
    return pool.attribute(cuda::memory_pool_attributes::used_mem_current);
  }
  [[nodiscard]] std::string usage() const
  {
    return "reserved_MiB=" + num(reserved() / MiB) + " used_MiB=" + num(used() / MiB);
  }
};

int device_count()
{
  int n = 0;
  cudaGetDeviceCount(&n);
  return n;
}

// The parent must never initialize CUDA (children are forked), so count devices in a child.
int device_count_in_child()
{
  const pid_t pid = fork();
  if (pid == 0) { _exit(device_count()); }
  int status = 0;
  waitpid(pid, &status, 0);
  return WEXITSTATUS(status);
}

enum class route { grant_all, grant_then_revoke, raw_create };

cuda::memory_pool_properties props(std::size_t max_bytes, std::size_t initial = 0)
{
  cuda::memory_pool_properties p{};
  p.max_pool_size     = max_bytes;
  p.initial_pool_size = initial;
  return p;
}

probe_pool make(route r, int numa, std::size_t max_bytes, const std::vector<int>& listed)
{
  auto is_listed = [&](int d) {
    for (int l : listed) {
      if (l == d) { return true; }
    }
    return false;
  };
  if (r == route::raw_create) {
    cudaMemPoolProps cp{};
    cp.allocType     = cudaMemAllocationTypePinned;
    cp.handleTypes   = cudaMemHandleTypeNone;
    cp.location.type = cudaMemLocationTypeHostNuma;
    cp.location.id   = numa;
    cp.maxSize       = max_bytes;
    cudaMemPool_t raw{};
    const cudaError_t e = cudaMemPoolCreate(&raw, &cp);
    if (e != cudaSuccess) { throw std::runtime_error("cudaMemPoolCreate: " + err(e)); }
    auto pool = cuda::shared_pinned_memory_pool::from_native_handle(raw);
    for (int d : listed) {
      pool.enable_access_from(cuda::device_ref{d});
    }
    return probe_pool{std::move(pool)};
  }
  cuda::shared_pinned_memory_pool pool{numa, props(max_bytes)};
  if (r == route::grant_then_revoke) {
    for (int d = 0; d < device_count(); ++d) {
      if (!is_listed(d)) { pool.disable_access_from(cuda::device_ref{d}); }
    }
  }
  return probe_pool{std::move(pool)};
}

template <class F>
void run_case(const char* name, F&& f)
{
  say(std::string("\n== ") + name);
  const pid_t pid = fork();
  if (pid == 0) {
    say("  result: " + outcome(std::forward<F>(f)));
    std::fflush(stdout);
    _exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  if (WIFSIGNALED(status)) {
    say("  CHILD KILLED BY SIGNAL " + std::to_string(WTERMSIG(status)));
  } else if (WEXITSTATUS(status) != 0) {
    say("  CHILD EXIT " + std::to_string(WEXITSTATUS(status)));
  }
}

// Allocate from the pool on a stream of `dev`, fill on the host, verify+flip in a kernel.
void kernel_on(int dev, probe_pool& pp, const char* tag)
{
  const std::size_t n     = std::size_t{1} << 20;
  const std::size_t bytes = n * sizeof(uint32_t);
  const uint32_t seed     = 2654435761U;
  std::string line        = std::string("  kernel dev=") + std::to_string(dev) + " (" + tag + "):";

  cudaSetDevice(dev);
  cudaStream_t s{};
  line += " stream_create=" + err(cudaStreamCreate(&s));
  void* raw = nullptr;
  line += " allocate=" + outcome([&] { raw = pp.h.allocate(cuda::stream_ref{s}, bytes, 256); });
  if (raw == nullptr) {
    say(line);
    return;
  }
  line += " sync_after_alloc=" + err(cudaStreamSynchronize(s));
  auto* p = static_cast<uint32_t*>(raw);
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = static_cast<uint32_t>(i) * seed;
  }
  int* bad = nullptr;
  cudaMalloc(&bad, sizeof(int));
  cudaMemset(bad, 0, sizeof(int));
  line += " launch=" + err(launch_verify_and_flip(p, n, seed, bad, s));
  line += " stream_sync=" + err(cudaStreamSynchronize(s));
  line += " device_sync=" + err(cudaDeviceSynchronize());
  line += " device_sync_again=" + err(cudaDeviceSynchronize());
  int bad_host   = -1;
  line += " copy_back=" + err(cudaMemcpy(&bad_host, bad, sizeof(int), cudaMemcpyDeviceToHost));
  void* later    = nullptr;
  line += " malloc_after=" + err(cudaMalloc(&later, 4096));
  bool flipped = true;
  for (std::size_t i = 0; i < n; i += 4097) {
    flipped = flipped && (p[i] == ~(static_cast<uint32_t>(i) * seed));
  }
  line += " mismatches=" + std::to_string(bad_host) + " host_sees_flip=" + (flipped ? "1" : "0");
  say(line);
  pp.h.deallocate(cuda::stream_ref{s}, raw, bytes, 256);
  cudaStreamSynchronize(s);
}

// Plain copy from pool memory to device memory of `dev`; no kernel dereference.
void memcpy_on(int dev, probe_pool& pp, const char* tag)
{
  const std::size_t bytes = 4 * MiB;
  cudaSetDevice(dev);
  cudaStream_t s{};
  cudaStreamCreate(&s);
  void* host = pp.h.allocate(cuda::stream_ref{s}, bytes, 256);
  cudaStreamSynchronize(s);
  auto* hp = static_cast<unsigned char*>(host);
  for (std::size_t i = 0; i < bytes; ++i) {
    hp[i] = static_cast<unsigned char>(i * 31U);
  }
  void* devbuf = nullptr;
  cudaMalloc(&devbuf, bytes);
  std::string line = std::string("  memcpy dev=") + std::to_string(dev) + " (" + tag + "):";
  line += " h2d=" + err(cudaMemcpyAsync(devbuf, host, bytes, cudaMemcpyHostToDevice, s));
  line += " sync=" + err(cudaStreamSynchronize(s));
  for (std::size_t i = 0; i < bytes; ++i) {
    hp[i] = 0;
  }
  line += " d2h=" + err(cudaMemcpyAsync(host, devbuf, bytes, cudaMemcpyDeviceToHost, s));
  line += " sync=" + err(cudaStreamSynchronize(s));
  bool same = true;
  for (std::size_t i = 0; i < bytes; i += 257) {
    same = same && (hp[i] == static_cast<unsigned char>(i * 31U));
  }
  line += std::string(" round_trip_ok=") + (same ? "1" : "0");
  say(line);
  pp.h.deallocate(cuda::stream_ref{s}, host, bytes, 256);
  cudaStreamSynchronize(s);
}

std::string access_flags(probe_pool& pp)
{
  std::string s = "  is_accessible_from:";
  for (int d = 0; d < device_count(); ++d) {
    s += " dev" + std::to_string(d) + "=" + (pp.pool.is_accessible_from(cuda::device_ref{d}) ? "1" : "0");
  }
  return s;
}

void versions()
{
  say("CCCL_VERSION=" + num(CCCL_VERSION));
  say("CUDART_VERSION=" + num(CUDART_VERSION));
#if __has_include(<rmm/cuda_stream_view.hpp>)
  say("rmm::cuda_stream_view header: present");
#else
  say("rmm::cuda_stream_view header: absent");
#endif
  say(std::string("RMM_VERSION=") + std::to_string(RMM_VERSION_MAJOR) + "." + std::to_string(RMM_VERSION_MINOR) +
      "." + std::to_string(RMM_VERSION_PATCH));
#if _CCCL_CTK_AT_LEAST(13, 0)
  say("gate CTK>=13.0 (default-location ctor, managed pools): on");
#else
  say("gate CTK>=13.0 (default-location ctor, managed pools): off");
#endif
#if _CCCL_CTK_AT_LEAST(13, 3)
  say("gate CTK>=13.3 (max_pool_size attribute readback): on");
#else
  say("gate CTK>=13.3 (max_pool_size attribute readback): off");
#endif
  run_case("runtime versions and devices", [] {
    int rt = 0;
    int drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    say("  cudaRuntimeGetVersion=" + std::to_string(rt) + " cudaDriverGetVersion=" + std::to_string(drv));
    for (int d = 0; d < device_count(); ++d) {
      cudaDeviceProp p{};
      cudaGetDeviceProperties(&p, d);
      int numa_pools = -1;
      int numa_id    = -2;
      cudaDeviceGetAttribute(&numa_pools, cudaDevAttrHostNumaMemoryPoolsSupported, d);
      cudaDeviceGetAttribute(&numa_id, cudaDevAttrHostNumaId, d);
      say("  dev" + std::to_string(d) + ": " + p.name + " sm_" + std::to_string(p.major) + std::to_string(p.minor) +
          " HostNumaMemoryPoolsSupported=" + std::to_string(numa_pools) + " HostNumaId=" + std::to_string(numa_id));
    }
  });
}

void basic_cases()
{
  run_case("node 0 with a 64 MiB limit", [] {
    auto pp = make(route::grant_all, 0, 64 * MiB, {});
    say("  fresh: " + pp.usage());
    cudaStream_t s{};
    cudaStreamCreate(&s);
    void* p = pp.h.allocate(cuda::stream_ref{s}, 16 * MiB, 256);
    cudaStreamSynchronize(s);
    static_cast<char*>(p)[0]            = 1;
    static_cast<char*>(p)[16 * MiB - 1] = 1;
    say("  after 16 MiB alloc: " + pp.usage() + " aligned256=" +
        std::to_string(reinterpret_cast<std::uintptr_t>(p) % 256 == 0) + " aligned4096=" +
        std::to_string(reinterpret_cast<std::uintptr_t>(p) % 4096 == 0));
    pp.h.deallocate(cuda::stream_ref{s}, p, 16 * MiB, 256);
    say("  after free, before sync: " + pp.usage());
    cudaStreamSynchronize(s);
    say("  after free and sync: " + pp.usage());
    say(access_flags(pp));
  });

  run_case("nonexistent NUMA node (4096)", [] {
    say("  CCCL ctor: " + outcome([] { cuda::shared_pinned_memory_pool p{4096, props(64 * MiB)}; }));
    cudaMemPoolProps cp{};
    cp.allocType     = cudaMemAllocationTypePinned;
    cp.location.type = cudaMemLocationTypeHostNuma;
    cp.location.id   = 4096;
    cp.maxSize       = 64 * MiB;
    cudaMemPool_t raw{};
    say("  raw cudaMemPoolCreate: " + err(cudaMemPoolCreate(&raw, &cp)));
    say("  CCCL ctor node 1 (machine has one node): " +
        outcome([] { cuda::shared_pinned_memory_pool p{1, props(64 * MiB)}; }));
    say("  CCCL ctor node -1: " + outcome([] { cuda::shared_pinned_memory_pool p{-1, props(64 * MiB)}; }));
  });

  run_case("over-limit allocation and recovery (64 MiB limit)", [] {
    auto pp = make(route::grant_all, 0, 64 * MiB, {});
    cudaStream_t s{};
    cudaStreamCreate(&s);
    const cuda::stream_ref sr{s};
    void* a = nullptr;
    say("  48 MiB: " + outcome([&] { a = pp.h.allocate(sr, 48 * MiB, 256); }));
    say("  +32 MiB (over): " + outcome([&] { (void)pp.h.allocate(sr, 32 * MiB, 256); }));
    say("  cudaGetLastError after failure: " + err(cudaGetLastError()));
    say("  +8 MiB (still fits): " + outcome([&] {
      void* b = pp.h.allocate(sr, 8 * MiB, 256);
      pp.h.deallocate(sr, b, 8 * MiB, 256);
    }));
    pp.h.deallocate(sr, a, 48 * MiB, 256);
    say("  32 MiB after free, no sync, same stream: " + outcome([&] {
      void* b = pp.h.allocate(sr, 32 * MiB, 256);
      pp.h.deallocate(sr, b, 32 * MiB, 256);
    }));
    cudaStreamSynchronize(s);
    say("  64 MiB after free and sync: " + outcome([&] {
      void* b = pp.h.allocate(sr, 64 * MiB, 256);
      pp.h.deallocate(sr, b, 64 * MiB, 256);
    }));
    say("  128 MiB single request (above limit): " + outcome([&] { (void)pp.h.allocate(sr, 128 * MiB, 256); }));
    say("  allocate_sync 128 MiB: " + outcome([&] { (void)pp.h.allocate_sync(128 * MiB); }));
    cudaStreamSynchronize(s);
    say("  end: " + pp.usage());
  });

  run_case("zero-byte allocate", [] {
    auto pp = make(route::grant_all, 0, 64 * MiB, {});
    cudaStream_t s{};
    cudaStreamCreate(&s);
    void* p = reinterpret_cast<void*>(std::uintptr_t{1});
    say("  allocate(stream, 0): " + outcome([&] { p = pp.h.allocate(cuda::stream_ref{s}, 0, 256); }) +
        " ptr_is_null=" + std::to_string(p == nullptr));
    void* q = reinterpret_cast<void*>(std::uintptr_t{1});
    say("  allocate_sync(0): " + outcome([&] { q = pp.h.allocate_sync(0); }) +
        " ptr_is_null=" + std::to_string(q == nullptr));
    say("  cudaGetLastError: " + err(cudaGetLastError()));
    pp.h.deallocate(cuda::stream_ref{s}, nullptr, 0, 256);
    pp.h.deallocate_sync(nullptr, 0);
    say("  deallocate(nullptr, 0) both forms returned; cudaGetLastError: " + err(cudaGetLastError()));
    say("  " + pp.usage());
  });

  run_case("alignment", [] {
    auto pp = make(route::grant_all, 0, 64 * MiB, {});
    cudaStream_t s{};
    cudaStreamCreate(&s);
    const cuda::stream_ref sr{s};
    std::string line = "  pointer alignment by size:";
    std::vector<std::pair<void*, std::size_t>> held;
    for (std::size_t bytes : {std::size_t{1}, std::size_t{100}, std::size_t{4096}, std::size_t{70000}, MiB}) {
      void* p = pp.h.allocate(sr, bytes, 256);
      held.emplace_back(p, bytes);
      const auto v = reinterpret_cast<std::uintptr_t>(p);
      line += " " + num(bytes) + ":" + (v % 4096 == 0 ? "4096" : v % 256 == 0 ? "256" : "LESS");
    }
    say(line);
    say("  used after 1+100+4096+70000+1MiB bytes: " + num(pp.used()));
    for (auto [p, b] : held) {
      pp.h.deallocate(sr, p, b, 256);
    }
    for (std::size_t a : {std::size_t{1}, std::size_t{8}, std::size_t{64}, std::size_t{256}, std::size_t{512},
                          std::size_t{4096}, std::size_t{3}}) {
      say("  allocate(align=" + num(a) + "): " + outcome([&] {
        void* p = pp.h.allocate(sr, 4096, a);
        pp.h.deallocate(sr, p, 4096, 256);
      }));
    }
    cudaStreamSynchronize(s);
  });

  run_case("alignment 0", [] {
    auto pp = make(route::grant_all, 0, 64 * MiB, {});
    cudaStream_t s{};
    cudaStreamCreate(&s);
    say("  allocate(align=0): " + outcome([&] { (void)pp.h.allocate(cuda::stream_ref{s}, 4096, 0); }));
  });

  run_case("trim_to", [] {
    auto pp = make(route::grant_all, 0, 64 * MiB, {});
    cudaStream_t s{};
    cudaStreamCreate(&s);
    const cuda::stream_ref sr{s};
    void* a = pp.h.allocate(sr, 32 * MiB, 256);
    void* b = pp.h.allocate(sr, 8 * MiB, 256);
    cudaStreamSynchronize(s);
    say("  two live (32+8): " + pp.usage());
    pp.h.deallocate(sr, a, 32 * MiB, 256);
    cudaStreamSynchronize(s);
    say("  32 freed and synced: " + pp.usage());
    say("  trim_to(16 MiB): " + outcome([&] { pp.pool.trim_to(16 * MiB); }) + " -> " + pp.usage());
    say("  trim_to(0): " + outcome([&] { pp.pool.trim_to(0); }) + " -> " + pp.usage());
    pp.h.deallocate(sr, b, 8 * MiB, 256);
    cudaStreamSynchronize(s);
    say("  all freed and synced: " + pp.usage());
    say("  trim_to(0): " + outcome([&] { pp.pool.trim_to(0); }) + " -> " + pp.usage());
    say("  reserved_mem_high_MiB=" + num(pp.pool.attribute(cuda::memory_pool_attributes::reserved_mem_high) / MiB) +
        " used_mem_high_MiB=" + num(pp.pool.attribute(cuda::memory_pool_attributes::used_mem_high) / MiB));
  });

  run_case("config rules (CCCL-side checks)", [] {
    auto mk = [](std::size_t max, std::size_t initial, std::size_t thresh) {
      return outcome([=] {
        cuda::memory_pool_properties p{};
        p.max_pool_size     = max;
        p.initial_pool_size = initial;
        p.release_threshold = thresh;
        probe_pool pp{cuda::shared_pinned_memory_pool{0, p}};
        say("    -> " + pp.usage());
      });
    };
    const std::size_t inf = std::numeric_limits<std::size_t>::max();
    say("  initial 16 MiB, max 64 MiB: " + mk(64 * MiB, 16 * MiB, inf));
    say("  initial 128 MiB > max 64 MiB: " + mk(64 * MiB, 128 * MiB, inf));
    say("  initial 16 MiB > release_threshold 8 MiB: " + mk(64 * MiB, 16 * MiB, 8 * MiB));
    say("  release_threshold 0: " + mk(64 * MiB, 0, 0));
    say("  max 0 (unlimited): " + mk(0, 0, inf));
  });

  run_case("handle semantics and rmm ref", [] {
    auto pp    = make(route::grant_all, 0, 64 * MiB, {});
    auto other = make(route::grant_all, 0, 64 * MiB, {});
    handle copy = pp.h;
    say(std::string("  copy == original: ") + (copy == pp.h ? "1" : "0") + "  other == original: " +
        (other.h == pp.h ? "1" : "0"));
    rmm::host_device_async_resource_ref ref{pp.h};
    rmm::device_async_resource_ref dref{ref};
    cudaStream_t s{};
    cudaStreamCreate(&s);
    {
      rmm::device_buffer buf{8 * MiB, PROBE_RMM_STREAM(s), dref};
      say("  rmm::device_buffer(8 MiB) through the ref: " + pp.usage() + " other pool used_MiB=" +
          num(other.used() / MiB));
    }
    cudaStreamSynchronize(s);
    say("  buffer destroyed and synced: " + pp.usage());
  });
}

void route_cases(route r, const char* rname)
{
  const int n = device_count_in_child();
  if (n < 2) {
    say(std::string("\n== ") + rname + ": SKIPPED, needs 2+ GPUs");
    return;
  }
  const int listed   = 1;
  const int unlisted = 0;
  const std::string R = std::string(rname) + ", listed={1}: ";
  auto mk = [r] { return make(r, 0, 64 * MiB, {1}); };

  run_case((R + "construct and access flags").c_str(), [&] {
    auto pp = mk();
    say(access_flags(pp));
  });
  run_case((R + "kernel on listed device").c_str(), [&] {
    auto pp = mk();
    kernel_on(listed, pp, "listed");
  });
  run_case((R + "kernel on unlisted device 0, then listed device in the same process").c_str(), [&] {
    auto pp = mk();
    kernel_on(unlisted, pp, "unlisted");
    kernel_on(listed, pp, "listed, after the failure");
    kernel_on(unlisted, pp, "unlisted, second attempt");
  });
  if (n > 2) {
    run_case((R + "kernel on unlisted device 2").c_str(), [&] {
      auto pp = mk();
      kernel_on(2, pp, "unlisted");
    });
  }
  run_case((R + "memcpy (no kernel) on unlisted device 0").c_str(), [&] {
    auto pp = mk();
    memcpy_on(unlisted, pp, "unlisted");
    memcpy_on(listed, pp, "listed");
  });
  run_case((R + "allocate_sync / deallocate_sync with device 0 unlisted").c_str(), [&] {
    auto pp  = mk();
    void* p  = nullptr;
    say("  allocate_sync(4 MiB): " + outcome([&] { p = pp.h.allocate_sync(4 * MiB); }));
    if (p != nullptr) {
      static_cast<char*>(p)[0]           = 1;
      static_cast<char*>(p)[4 * MiB - 1] = 1;
      say("  host write ok; " + pp.usage());
      pp.h.deallocate_sync(p, 4 * MiB);
      say("  deallocate_sync returned; cudaGetLastError: " + err(cudaGetLastError()) + "; " + pp.usage());
    }
  });
  run_case((R + "grant an unlisted device later, then kernel on it").c_str(), [&] {
    auto pp = mk();
    say("  enable_access_from(dev0): " + outcome([&] { pp.pool.enable_access_from(cuda::device_ref{0}); }));
    say(access_flags(pp));
    kernel_on(0, pp, "granted late");
  });
  run_case((R + "allocate while granted, revoke, then kernel on the revoked device").c_str(), [&] {
    auto pp = mk();
    cudaSetDevice(listed);
    cudaStream_t s{};
    cudaStreamCreate(&s);
    void* live = pp.h.allocate(cuda::stream_ref{s}, 4 * MiB, 256);
    cudaStreamSynchronize(s);
    say("  disable_access_from(dev1) with a live allocation: " +
        outcome([&] { pp.pool.disable_access_from(cuda::device_ref{1}); }));
    say(access_flags(pp));
    kernel_on(listed, pp, "revoked after use, new allocation");
    (void)live;
  });
}

}  // namespace

int main()
{
  versions();
  basic_cases();
  route_cases(route::grant_then_revoke, "route A grant-all then disable_access_from");
  route_cases(route::raw_create, "route B raw cudaMemPoolCreate + from_native_handle");
  return 0;
}
