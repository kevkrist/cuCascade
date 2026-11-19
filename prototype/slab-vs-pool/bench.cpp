// PROTOTYPE, throwaway. Answers: should the slab carve blocks out of big chunks it caches itself
// ("chunked"), or make one pool allocation per block and let cudaMemPool do the caching
// ("per-block")? Both designs sit on the same HostNuma cudaMemPool_t (node 0, release threshold
// = max, access granted to all GPUs), as decided in "Pool resource API and semantic promise".
//
// Build: nvcc -O2 -std=c++20 -o bench bench.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

static constexpr size_t KiB = size_t{1} << 10;
static constexpr size_t MiB = size_t{1} << 20;
static constexpr size_t GiB = size_t{1} << 30;
static constexpr size_t ALIGN_PAD = 4096 - 256;  // over-allocation for 4096-aligned blocks

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static cudaMemPool_t make_pool(size_t max_bytes)
{
  cudaMemPoolProps props{};
  props.allocType     = cudaMemAllocationTypePinned;
  props.handleTypes   = cudaMemHandleTypeNone;
  props.location.type = cudaMemLocationTypeHostNuma;
  props.location.id   = 0;
  props.maxSize       = max_bytes;
  cudaMemPool_t pool{};
  CK(cudaMemPoolCreate(&pool, &props));
  uint64_t thr = UINT64_MAX;
  CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
  int n = 0;
  CK(cudaGetDeviceCount(&n));
  std::vector<cudaMemAccessDesc> d(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    d[static_cast<size_t>(i)].location = {cudaMemLocationTypeDevice, i};
    d[static_cast<size_t>(i)].flags    = cudaMemAccessFlagsProtReadWrite;
  }
  CK(cudaMemPoolSetAccess(pool, d.data(), d.size()));
  return pool;
}

static uint64_t pool_attr(cudaMemPool_t p, cudaMemPoolAttr a)
{
  uint64_t v = 0;
  CK(cudaMemPoolGetAttribute(p, a, &v));
  return v;
}
static void print_pool(char const* tag, cudaMemPool_t p)
{
  std::printf("    %-34s pool reserved %6.2f GiB, pool used %6.2f GiB\n", tag,
              double(pool_attr(p, cudaMemPoolAttrReservedMemCurrent)) / double(GiB),
              double(pool_attr(p, cudaMemPoolAttrUsedMemCurrent)) / double(GiB));
}

// ---------------------------------------------------------------------------------------------
// Design A: one pool allocation per block. Stream-ordered allocs on the slab's stream, one sync
// per allocate_blocks call; frees go straight back to the pool (stream-ordered, no sync).
struct per_block_slab {
  cudaMemPool_t pool;
  size_t bs;
  cudaStream_t shared_stream;  // design per decision: one private stream per slab
  bool per_thread_stream;      // variant: each thread brings its own stream

  // returns false on OOM (and frees what it got: strong guarantee)
  bool allocate(size_t k, std::vector<void*>& out, cudaStream_t s)
  {
    out.clear();
    for (size_t i = 0; i < k; ++i) {
      void* p = nullptr;
      cudaError_t e = cudaMallocFromPoolAsync(&p, bs + ALIGN_PAD, pool, s);
      if (e != cudaSuccess) {
        (void)cudaGetLastError();
        for (void* q : out) CK(cudaFreeAsync(q, s));
        out.clear();
        return false;
      }
      out.push_back(p);
    }
    CK(cudaStreamSynchronize(s));
    return true;
  }
  void deallocate(std::vector<void*>& blocks, cudaStream_t s)
  {
    for (void* p : blocks) CK(cudaFreeAsync(p, s));
    blocks.clear();
  }
};

// Design B: chunked slab. Mutex + LIFO free list; grows one chunk (blocks_per_chunk blocks) at a
// time from the pool; returning a block only pushes it onto the free list. (No per-chunk
// occupancy tracking here, so its measured cost is a lower bound for the real design.)
struct chunked_slab {
  cudaMemPool_t pool;
  size_t bs;
  size_t blocks_per_chunk;
  cudaStream_t stream;
  std::mutex mu;
  std::vector<void*> free_list;
  std::vector<void*> chunks;
  size_t total_blocks = 0;

  bool grow_locked()
  {
    void* c = nullptr;
    cudaError_t e = cudaMallocFromPoolAsync(&c, bs * blocks_per_chunk + ALIGN_PAD, pool, stream);
    if (e != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    CK(cudaStreamSynchronize(stream));
    chunks.push_back(c);
    auto base = (reinterpret_cast<uintptr_t>(c) + 4095) & ~uintptr_t{4095};
    for (size_t i = 0; i < blocks_per_chunk; ++i)
      free_list.push_back(reinterpret_cast<void*>(base + i * bs));
    total_blocks += blocks_per_chunk;
    return true;
  }
  bool allocate(size_t k, std::vector<void*>& out)
  {
    out.clear();
    std::lock_guard lk(mu);
    while (free_list.size() < k)
      if (!grow_locked()) return false;
    out.assign(free_list.end() - static_cast<std::ptrdiff_t>(k), free_list.end());
    free_list.resize(free_list.size() - k);
    return true;
  }
  void deallocate(std::vector<void*>& blocks)
  {
    std::lock_guard lk(mu);
    free_list.insert(free_list.end(), blocks.begin(), blocks.end());
    blocks.clear();
  }
  size_t release_free_chunks()  // only valid here when *all* blocks are free (prototype shortcut)
  {
    std::lock_guard lk(mu);
    if (free_list.size() != total_blocks) return 0;
    size_t n = chunks.size();
    for (void* c : chunks) CK(cudaFreeAsync(c, stream));
    CK(cudaStreamSynchronize(stream));
    chunks.clear();
    free_list.clear();
    total_blocks = 0;
    return n * (bs * blocks_per_chunk + ALIGN_PAD);
  }
  void destroy()
  {
    for (void* c : chunks) CK(cudaFreeAsync(c, stream));
    CK(cudaStreamSynchronize(stream));
    chunks.clear();
    free_list.clear();
    total_blocks = 0;
  }
};

// ---------------------------------------------------------------------------------------------
// Scenario 1: steady-state cost per block. T threads loop: allocate k blocks, host-write one
// byte per block, return them. Pool and slab are warmed first.
enum class design { per_block_shared, per_block_per_thread, chunked };
static char const* name(design d)
{
  switch (d) {
    case design::per_block_shared: return "per-block (1 slab stream)";
    case design::per_block_per_thread: return "per-block (stream/thread)";
    case design::chunked: return "chunked slab";
  }
  return "?";
}

static double run_steady(design d, size_t bs, size_t k, int threads, double seconds)
{
  cudaMemPool_t pool = make_pool(64 * GiB);
  cudaStream_t shared{};
  CK(cudaStreamCreateWithFlags(&shared, cudaStreamNonBlocking));
  per_block_slab A{pool, bs, shared, d == design::per_block_per_thread};
  chunked_slab B{pool, bs, 128, shared, {}, {}, {}, 0};

  std::vector<cudaStream_t> ts(static_cast<size_t>(threads));
  for (auto& s : ts) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  auto one_iter = [&](int t, std::vector<void*>& v) {
    cudaStream_t s = d == design::per_block_per_thread ? ts[static_cast<size_t>(t)] : shared;
    bool ok = d == design::chunked ? B.allocate(k, v) : A.allocate(k, v, s);
    if (!ok) { std::fprintf(stderr, "unexpected OOM\n"); std::exit(1); }
    for (void* p : v) *static_cast<volatile char*>(p) = 1;
    if (d == design::chunked) B.deallocate(v); else A.deallocate(v, s);
  };

  // warm up: every thread holds its peak at once so the pool/slab reach steady size
  {
    std::vector<std::thread> w;
    for (int t = 0; t < threads; ++t)
      w.emplace_back([&, t] { std::vector<void*> v; for (int i = 0; i < 20; ++i) one_iter(t, v); });
    for (auto& x : w) x.join();
    CK(cudaDeviceSynchronize());
  }

  std::atomic<bool> stop{false};
  std::atomic<uint64_t> blocks{0};
  std::vector<std::thread> w;
  auto t0 = clk::now();
  for (int t = 0; t < threads; ++t)
    w.emplace_back([&, t] {
      std::vector<void*> v;
      v.reserve(k);
      uint64_t n = 0;
      while (!stop.load(std::memory_order_relaxed)) { one_iter(t, v); n += k; }
      blocks += n;
    });
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
  stop = true;
  for (auto& x : w) x.join();
  auto t1 = clk::now();
  CK(cudaDeviceSynchronize());

  double ns_per_block_per_thread = secs(t0, t1) * 1e9 * threads / double(blocks.load());
  if (d == design::chunked) B.destroy();
  CK(cudaDeviceSynchronize());
  for (auto& s : ts) CK(cudaStreamDestroy(s));
  CK(cudaStreamDestroy(shared));
  CK(cudaMemPoolDestroy(pool));
  return ns_per_block_per_thread;
}

// What one block costs to fill with a host memcpy (for scale).
static double memcpy_ns(size_t bs)
{
  std::vector<char> a(bs, 1), b(bs);
  for (int i = 0; i < 3; ++i) std::memcpy(b.data(), a.data(), bs);
  int iters = std::max(10, int(256 * MiB / bs));
  auto t0 = clk::now();
  for (int i = 0; i < iters; ++i) { std::memcpy(b.data(), a.data(), bs); asm volatile("" ::: "memory"); }
  return secs(t0, clk::now()) * 1e9 / iters;
}

// ---------------------------------------------------------------------------------------------
// Scenario 2: cold growth. Fresh pool, T threads each allocate `per_thread` bytes in 1 MiB blocks
// at the same time (first touch of brand-new pinned memory).
static void run_cold(design d, int threads, size_t per_thread)
{
  size_t bs = MiB, k = per_thread / bs;
  cudaMemPool_t pool = make_pool(64 * GiB);
  cudaStream_t shared{};
  CK(cudaStreamCreateWithFlags(&shared, cudaStreamNonBlocking));
  per_block_slab A{pool, bs, shared, false};
  chunked_slab B{pool, bs, 128, shared, {}, {}, {}, 0};
  std::vector<std::vector<void*>> held(static_cast<size_t>(threads));
  std::vector<std::thread> w;
  auto t0 = clk::now();
  for (int t = 0; t < threads; ++t)
    w.emplace_back([&, t] {
      auto& v = held[static_cast<size_t>(t)];
      bool ok = d == design::chunked ? B.allocate(k, v) : A.allocate(k, v, shared);
      if (!ok) { std::fprintf(stderr, "unexpected OOM\n"); std::exit(1); }
    });
  for (auto& x : w) x.join();
  double s = secs(t0, clk::now());
  std::printf("    %-28s %2d threads x %4zu MiB: %7.1f ms  (%5.2f GiB/s)\n", name(d), threads,
              per_thread / MiB, s * 1e3, double(per_thread) * threads / double(GiB) / s);
  for (auto& v : held) {
    if (d == design::chunked) B.deallocate(v); else A.deallocate(v, shared);
  }
  if (d == design::chunked) B.destroy();
  CK(cudaStreamSynchronize(shared));
  CK(cudaStreamDestroy(shared));
  CK(cudaMemPoolDestroy(pool));
}

// ---------------------------------------------------------------------------------------------
// Scenario 3: memory left idle after a spike. Pool limit 10 GiB. The slab spikes to 8 GiB of
// 1 MiB blocks, returns them all, then "cudf" asks the same pool for one contiguous buffer.
static bool try_contig(cudaMemPool_t pool, cudaStream_t s, size_t bytes)
{
  void* p = nullptr;
  cudaError_t e = cudaMallocFromPoolAsync(&p, bytes, pool, s);
  if (e != cudaSuccess) { (void)cudaGetLastError(); return false; }
  CK(cudaFreeAsync(p, s));
  CK(cudaStreamSynchronize(s));
  return true;
}

static void run_spike(design d, bool call_release, size_t contig)
{
  size_t bs = MiB;
  cudaMemPool_t pool = make_pool(10 * GiB);
  cudaStream_t s{};
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  per_block_slab A{pool, bs, s, false};
  chunked_slab B{pool, bs, 128, s, {}, {}, {}, 0};
  std::vector<void*> v;
  size_t k = 8 * GiB / bs;
  bool ok = d == design::chunked ? B.allocate(k, v) : A.allocate(k, v, s);
  if (!ok) { std::fprintf(stderr, "unexpected OOM in spike\n"); std::exit(1); }
  std::printf("  %s%s, contiguous request %zu GiB:\n", name(d),
              d == design::chunked ? (call_release ? " + release_free_chunks()" : " (no release)") : "",
              contig / GiB);
  print_pool("after 8 GiB spike:", pool);
  if (d == design::chunked) { B.deallocate(v); if (call_release) B.release_free_chunks(); }
  else A.deallocate(v, s);
  CK(cudaStreamSynchronize(s));
  print_pool("after all blocks returned:", pool);
  bool got = try_contig(pool, s, contig);
  std::printf("    cudf %zu GiB contiguous: %s\n", contig / GiB, got ? "OK" : "OUT OF MEMORY");
  print_pool("after cudf request:", pool);
  if (d == design::chunked) B.destroy();
  CK(cudaStreamSynchronize(s));
  CK(cudaStreamDestroy(s));
  CK(cudaMemPoolDestroy(pool));
}

int main(int argc, char** argv)
{
  double secs_per = argc > 1 ? std::atof(argv[1]) : 1.0;
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));

  std::printf("== Scenario 1: steady-state cost per block (ns per block per thread; lower is better)\n");
  for (size_t bs : {64 * KiB, MiB, 4 * MiB}) {
    std::printf("  block %zu KiB  (host memcpy of one block: %.0f ns)\n", bs / KiB, memcpy_ns(bs));
    std::printf("    %-28s %8s %8s %8s %8s\n", "k blocks/call x threads ->", "1x1", "16x1", "16x8", "256x8");
    std::printf("    %-28s %8s %8s %8s %8s\n", "", "", "", "", "");
    for (design d : {design::per_block_shared, design::per_block_per_thread, design::chunked}) {
      std::printf("    %-28s", name(d));
      std::fflush(stdout);
      for (auto [k, t] : {std::pair<size_t, int>{1, 1}, {16, 1}, {16, 8}, {256, 8}}) {
        if (bs * k * size_t(t) > 16 * GiB) { std::printf(" %8s", "-"); continue; }
        std::printf(" %8.0f", run_steady(d, bs, k, t, secs_per));
        std::fflush(stdout);
      }
      std::printf("\n");
    }
    // high-contention row
    std::printf("    32 threads, 16 blocks/call:");
    for (design d : {design::per_block_shared, design::per_block_per_thread, design::chunked})
      if (bs * 16 * 32 <= 16 * GiB) std::printf("  %s=%0.f", name(d), run_steady(d, bs, 16, 32, secs_per));
    std::printf("\n");
  }

  std::printf("\n== Scenario 2: cold growth (fresh pool; first touch of new pinned memory)\n");
  for (int t : {1, 8, 32})
    for (design d : {design::per_block_shared, design::chunked}) run_cold(d, t, 512 * MiB);

  std::printf("\n== Scenario 3: memory idle after a spike (pool limit 10 GiB, spike 8 GiB of 1 MiB blocks)\n");
  for (size_t c : {size_t{1} * GiB, size_t{6} * GiB}) {
    run_spike(design::per_block_shared, false, c);
    run_spike(design::chunked, false, c);
    run_spike(design::chunked, true, c);
  }
  return 0;
}
