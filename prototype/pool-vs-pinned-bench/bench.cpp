// PROTOTYPE, throwaway, never merged. Answers "Benchmark pool vs current pinned path":
// does a HostNuma cudaMemPool_t beat today's small_pinned_host_memory_resource (slabs up to
// 8 KiB, cudaHostAlloc above that) as cudf's pinned provider?
//
// "current" is the real small_pinned_host_memory_resource from libcucascade.so.
// "pool" is a raw cudaMemPool_t configured as decided in "Pool resource API and semantic
// promise" (HostNuma node 0, pinned, release threshold = max, access for every visible GPU).
// The real resource wraps CCCL's pool around the same driver calls.
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static constexpr std::size_t KiB = std::size_t{1} << 10;
static constexpr std::size_t MiB = std::size_t{1} << 20;
static constexpr std::size_t GiB = std::size_t{1} << 30;

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b)
{
  return std::chrono::duration<double, std::micro>(b - a).count();
}

static cudaMemPool_t make_pool(std::size_t max_bytes)
{
  cudaMemPoolProps props{};
  props.allocType     = cudaMemAllocationTypePinned;
  props.handleTypes   = cudaMemHandleTypeNone;
  props.location.type = cudaMemLocationTypeHostNuma;
  props.location.id   = 0;
  props.maxSize       = max_bytes;
  cudaMemPool_t pool{};
  CK(cudaMemPoolCreate(&pool, &props));
  std::uint64_t thr = UINT64_MAX;
  CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
  int n = 0;
  CK(cudaGetDeviceCount(&n));
  std::vector<cudaMemAccessDesc> d(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    d[static_cast<std::size_t>(i)].location = {cudaMemLocationTypeDevice, i};
    d[static_cast<std::size_t>(i)].flags    = cudaMemAccessFlagsProtReadWrite;
  }
  CK(cudaMemPoolSetAccess(pool, d.data(), d.size()));
  return pool;
}

static std::uint64_t pool_attr(cudaMemPool_t p, cudaMemPoolAttr a)
{
  std::uint64_t v = 0;
  CK(cudaMemPoolGetAttribute(p, a, &v));
  return v;
}

struct allocator {
  virtual ~allocator()                                         = default;
  virtual void* alloc(cudaStream_t s, std::size_t bytes)       = 0;
  virtual void free(cudaStream_t s, void* p, std::size_t bytes) = 0;
};

struct current_alloc final : allocator {
  cucascade::memory::small_pinned_host_memory_resource& mr;
  explicit current_alloc(cucascade::memory::small_pinned_host_memory_resource& m) : mr(m) {}
  void* alloc(cudaStream_t s, std::size_t bytes) override
  {
    return mr.allocate(::cuda::stream_ref{s}, bytes);
  }
  void free(cudaStream_t s, void* p, std::size_t bytes) override
  {
    mr.deallocate(::cuda::stream_ref{s}, p, bytes);
  }
};

struct pool_alloc final : allocator {
  cudaMemPool_t pool;
  explicit pool_alloc(cudaMemPool_t p) : pool(p) {}
  void* alloc(cudaStream_t s, std::size_t bytes) override
  {
    void* p = nullptr;
    CK(cudaMallocFromPoolAsync(&p, bytes, pool, s));
    return p;
  }
  void free(cudaStream_t s, void* p, std::size_t) override { CK(cudaFreeAsync(p, s)); }
};

struct stats {
  double mean{}, p50{}, p99{};
  std::size_t n{};
};
static stats summarize(std::vector<double>& v)
{
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  return {sum / double(v.size()), v[v.size() / 2], v[(v.size() * 99) / 100], v.size()};
}

// One cycle = allocate, [sync the stream, as cudf does before host use], write one byte, free.
// Runs `warm` untimed cycles, then timed cycles until `budget_s` or `max_iters`.
static stats run_cycles(allocator& a, cudaStream_t s, std::size_t bytes, bool sync,
                        double budget_s = 1.0, std::size_t max_iters = 200000,
                        std::size_t warm = 16)
{
  auto cycle = [&] {
    void* p = a.alloc(s, bytes);
    if (sync) { CK(cudaStreamSynchronize(s)); *static_cast<volatile char*>(p) = 1; }
    a.free(s, p, bytes);
  };
  for (std::size_t i = 0; i < warm; ++i) cycle();
  CK(cudaStreamSynchronize(s));
  std::vector<double> t;
  t.reserve(max_iters);
  auto const start = clk::now();
  while (t.size() < max_iters) {
    auto const t0 = clk::now();
    cycle();
    auto const t1 = clk::now();
    t.push_back(us(t0, t1));
    if (us(start, t1) > budget_s * 1e6 && t.size() >= 10) break;
  }
  CK(cudaStreamSynchronize(s));
  return summarize(t);
}

// Hold `batch` allocations live, sync once, free them all. Reported per allocation.
static stats run_batches(allocator& a, cudaStream_t s, std::size_t bytes, std::size_t batch,
                         double budget_s = 1.0)
{
  std::vector<void*> live(batch);
  auto round = [&] {
    for (auto& p : live) p = a.alloc(s, bytes);
    CK(cudaStreamSynchronize(s));
    for (auto& p : live) { *static_cast<volatile char*>(p) = 1; a.free(s, p, bytes); }
  };
  for (int i = 0; i < 8; ++i) round();
  std::vector<double> t;
  auto const start = clk::now();
  while (t.size() < 100000) {
    auto const t0 = clk::now();
    round();
    auto const t1 = clk::now();
    t.push_back(us(t0, t1) / double(batch));
    if (us(start, t1) > budget_s * 1e6 && t.size() >= 10) break;
  }
  return summarize(t);
}

// Varying sizes with a few buffers live at once: each step frees the oldest of `ring` buffers
// and allocates a new one of pseudo-random size in [lo, hi] (log-uniform), then syncs.
static stats run_mixed(allocator& a, cudaStream_t s, std::size_t lo, std::size_t hi,
                       std::size_t ring, std::size_t steps, std::size_t* peak_live)
{
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> u(std::log2(double(lo)), std::log2(double(hi)));
  std::vector<std::pair<void*, std::size_t>> live(ring, {nullptr, 0});
  std::vector<double> t;
  std::size_t cur = 0, peak = 0;
  for (std::size_t i = 0; i < steps + ring; ++i) {
    auto const bytes = static_cast<std::size_t>(std::exp2(u(rng)));
    auto& slot       = live[i % ring];
    auto const t0    = clk::now();
    if (slot.first != nullptr) { a.free(s, slot.first, slot.second); cur -= slot.second; }
    slot = {a.alloc(s, bytes), bytes};
    CK(cudaStreamSynchronize(s));
    *static_cast<volatile char*>(slot.first) = 1;
    auto const t1 = clk::now();
    cur += bytes;
    peak = std::max(peak, cur);
    if (i >= ring) t.push_back(us(t0, t1));
  }
  for (auto& [p, b] : live) a.free(s, p, b);
  CK(cudaStreamSynchronize(s));
  if (peak_live != nullptr) *peak_live = peak;
  return summarize(t);
}

struct mt_result {
  double us_per_cycle;  // wall time one thread spends per cycle
  double cycles_per_s;  // all threads together
};
static mt_result run_mt(allocator& a, int threads, std::size_t bytes, bool sync, int num_devices)
{
  std::atomic<int> phase{0};  // 0 warm-up, 1 measured, 2 stop
  std::atomic<int> ready{0};
  std::vector<std::uint64_t> counts(static_cast<std::size_t>(threads), 0);
  std::vector<std::thread> ts;
  for (int i = 0; i < threads; ++i) {
    ts.emplace_back([&, i] {
      CK(cudaSetDevice(i % num_devices));
      cudaStream_t s{};
      CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
      ready.fetch_add(1);
      std::uint64_t n = 0;
      for (;;) {
        int const ph = phase.load(std::memory_order_relaxed);
        if (ph == 2) break;
        void* p = a.alloc(s, bytes);
        if (sync) { CK(cudaStreamSynchronize(s)); *static_cast<volatile char*>(p) = 1; }
        a.free(s, p, bytes);
        if (ph == 1) ++n;
      }
      CK(cudaStreamSynchronize(s));
      CK(cudaStreamDestroy(s));
      counts[static_cast<std::size_t>(i)] = n;
    });
  }
  while (ready.load() < threads) std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  auto const t0 = clk::now();
  phase = 1;
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  phase         = 2;
  auto const t1 = clk::now();
  for (auto& t : ts) t.join();
  std::uint64_t total = 0;
  for (auto c : counts) total += c;
  double const el = us(t0, t1);
  return {el * threads / double(std::max<std::uint64_t>(total, 1)), double(total) / (el / 1e6)};
}

static char const* human(std::size_t b)
{
  static thread_local char buf[4][32];
  static thread_local int k = 0;
  char* o                   = buf[k++ % 4];
  if (b >= MiB) std::snprintf(o, 32, "%zu MiB", b / MiB);
  else if (b >= KiB) std::snprintf(o, 32, "%zu KiB", b / KiB);
  else std::snprintf(o, 32, "%zu B", b);
  return o;
}

// ---------------------------------------------------------------------------------------------
static void scenario_large(allocator& cur, allocator& pool, cudaStream_t s)
{
  std::printf("\n== 1. Repeated large requests, 1 thread: allocate(stream) + sync + free(stream)\n");
  std::printf("   µs per cycle. 'pool cold' = first allocation of that size from a new, empty pool.\n");
  std::printf("   %-9s | %-30s | %-30s | %-10s | %s\n", "size", "current mean / p50 / p99",
              "pool warm mean / p50 / p99", "pool cold", "current/pool");
  for (std::size_t bytes : {16 * KiB, 64 * KiB, 1 * MiB, 16 * MiB, 64 * MiB, 256 * MiB}) {
    auto const c = run_cycles(cur, s, bytes, true);
    auto const p = run_cycles(pool, s, bytes, true);
    cudaMemPool_t fresh = make_pool(1 * GiB);
    auto const t0       = clk::now();
    void* q             = nullptr;
    CK(cudaMallocFromPoolAsync(&q, bytes, fresh, s));
    CK(cudaStreamSynchronize(s));
    *static_cast<volatile char*>(q) = 1;
    auto const cold                 = us(t0, clk::now());
    CK(cudaFreeAsync(q, s));
    CK(cudaStreamSynchronize(s));
    CK(cudaMemPoolDestroy(fresh));
    std::printf("   %-9s | %9.1f /%9.1f /%9.1f | %9.2f /%9.2f /%9.2f | %10.1f | %8.0fx\n",
                human(bytes), c.mean, c.p50, c.p99, p.mean, p.p50, p.p99, cold, c.mean / p.mean);
  }

  std::printf("\n   Varying sizes (log-uniform 64 KiB..32 MiB), 4 buffers live, 2000 steps, µs per step\n");
  std::size_t peak = 0;
  auto const c     = run_mixed(cur, s, 64 * KiB, 32 * MiB, 4, 2000, &peak);
  pool_alloc fresh{make_pool(8 * GiB)};  // new pool, so 'reserved' reflects this run only
  auto const p = run_mixed(fresh, s, 64 * KiB, 32 * MiB, 4, 2000, nullptr);
  std::printf("   current mean %.1f p99 %.1f | pool mean %.2f p99 %.2f (includes growth) | %.0fx\n",
              c.mean, c.p99, p.mean, p.p99, c.mean / p.mean);
  std::printf("   peak live %.1f MiB; pool reserved after run %.1f MiB\n",
              double(peak) / double(MiB),
              double(pool_attr(fresh.pool, cudaMemPoolAttrReservedMemCurrent)) / double(MiB));
  CK(cudaMemPoolDestroy(fresh.pool));
  (void)pool;
}

static void scenario_small(allocator& cur, allocator& pool, cudaStream_t s)
{
  std::printf("\n== 2. Small allocations (current = slab path, <= 8 KiB), 1 thread, µs per allocation\n");
  std::printf("   A: alloc + sync + free.  B: alloc + free, no sync.  C: 64 live, one sync, free all.\n");
  std::printf("   %-7s | %-21s | %-21s | %-21s\n", "size", "A current / pool", "B current / pool",
              "C current / pool");
  for (std::size_t bytes : {std::size_t{64}, std::size_t{512}, 4 * KiB, 8 * KiB}) {
    auto const ca = run_cycles(cur, s, bytes, true);
    auto const pa = run_cycles(pool, s, bytes, true);
    auto const cb = run_cycles(cur, s, bytes, false);
    auto const pb = run_cycles(pool, s, bytes, false);
    auto const cc = run_batches(cur, s, bytes, 64);
    auto const pc = run_batches(pool, s, bytes, 64);
    std::printf("   %-7s | %9.2f / %9.2f | %9.2f / %9.2f | %9.2f / %9.2f\n", human(bytes), ca.mean,
                pa.mean, cb.mean, pb.mean, cc.mean, pc.mean);
  }
}

static double copy_gbps(void* dst, void const* src, std::size_t bytes, cudaMemcpyKind kind,
                        cudaStream_t s)
{
  std::vector<double> t;
  for (int i = 0; i < 12; ++i) {
    auto const t0 = clk::now();
    CK(cudaMemcpyAsync(dst, src, bytes, kind, s));
    CK(cudaStreamSynchronize(s));
    t.push_back(us(t0, clk::now()));
  }
  std::sort(t.begin(), t.end());
  return double(bytes) / 1e9 / (t[t.size() / 2] / 1e6);  // median
}

static void scenario_copy(cudaMemPool_t pool, int num_devices)
{
  std::size_t const bytes = 256 * MiB;
  std::printf("\n== 3. Copy bandwidth, %s buffer, cudaMemcpyAsync + sync, median of 12, GB/s\n",
              human(bytes));
  std::printf("   %-44s", "host buffer");
  for (int d = 0; d < num_devices; ++d) std::printf(" | dev%d H2D  dev%d D2H", d, d);
  std::printf("\n");

  cudaStream_t s0{};
  CK(cudaSetDevice(0));
  CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
  void* from_pool = nullptr;
  CK(cudaMallocFromPoolAsync(&from_pool, bytes, pool, s0));
  CK(cudaStreamSynchronize(s0));
  void* host_alloc = nullptr;
  CK(cudaHostAlloc(&host_alloc, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
  void* registered = nullptr;
  if (posix_memalign(&registered, 2 * MiB, bytes) != 0) std::exit(1);
  std::memset(registered, 1, bytes);
  CK(cudaHostRegister(registered, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
  void* pageable = nullptr;
  if (posix_memalign(&pageable, 2 * MiB, bytes) != 0) std::exit(1);
  std::memset(pageable, 1, bytes);
  std::memset(from_pool, 1, bytes);
  std::memset(host_alloc, 1, bytes);

  struct row {
    char const* name;
    void* p;
  } rows[] = {{"pool (HostNuma cudaMemPool)", from_pool},
              {"cudaHostAlloc Portable|Mapped (today >8 KiB)", host_alloc},
              {"malloc + cudaHostRegister (today's blocks)", registered},
              {"pageable", pageable}};
  for (auto const& r : rows) {
    std::printf("   %-44s", r.name);
    for (int d = 0; d < num_devices; ++d) {
      CK(cudaSetDevice(d));
      cudaStream_t s{};
      CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
      void* dev = nullptr;
      CK(cudaMalloc(&dev, bytes));
      double const h2d = copy_gbps(dev, r.p, bytes, cudaMemcpyHostToDevice, s);
      double const d2h = copy_gbps(r.p, dev, bytes, cudaMemcpyDeviceToHost, s);
      std::printf(" | %8.1f  %8.1f", h2d, d2h);
      CK(cudaFree(dev));
      CK(cudaStreamDestroy(s));
    }
    std::printf("\n");
  }
  CK(cudaSetDevice(0));
  CK(cudaFreeAsync(from_pool, s0));
  CK(cudaStreamSynchronize(s0));
  CK(cudaStreamDestroy(s0));
  CK(cudaFreeHost(host_alloc));
  CK(cudaHostUnregister(registered));
  std::free(registered);
  std::free(pageable);
}

static void scenario_mt(allocator& cur, allocator& pool, int num_devices)
{
  std::printf("\n== 4. Multi-threaded allocate/free, one stream per thread, 1.5 s per cell\n");
  std::printf("   µs per cycle per thread (aggregate cycles/s in brackets)\n");
  struct cfg {
    std::size_t bytes;
    bool sync;
    int devices;
    char const* label;
  } cfgs[] = {
    {4 * KiB, true, 1, "4 KiB, alloc+sync+free, all threads on dev0 (current = mutex slab path)"},
    {4 * KiB, false, 1, "4 KiB, alloc+free no sync, all threads on dev0"},
    {4 * KiB, true, num_devices, "4 KiB, alloc+sync+free, threads spread over all GPUs"},
    {1 * MiB, true, 1, "1 MiB, alloc+sync+free, dev0 (current = cudaHostAlloc)"},
    {16 * MiB, true, 1, "16 MiB, alloc+sync+free, dev0 (current = cudaHostAlloc)"},
  };
  for (auto const& c : cfgs) {
    std::printf("\n   %s\n   %-7s | %-28s | %-28s | %s\n", c.label, "threads", "current", "pool",
                "current/pool");
    for (int t : {1, 2, 4, 8, 16, 32}) {
      auto const a = run_mt(cur, t, c.bytes, c.sync, c.devices);
      auto const b = run_mt(pool, t, c.bytes, c.sync, c.devices);
      std::printf("   %-7d | %10.2f [%12.0f/s] | %10.2f [%12.0f/s] | %8.2fx\n", t, a.us_per_cycle,
                  a.cycles_per_s, b.us_per_cycle, b.cycles_per_s, a.us_per_cycle / b.us_per_cycle);
      std::fflush(stdout);
    }
  }
}

int main()
{
  int num_devices = 0;
  CK(cudaGetDeviceCount(&num_devices));
  int rt = 0, drv = 0;
  CK(cudaRuntimeGetVersion(&rt));
  CK(cudaDriverGetVersion(&drv));
  std::printf("CUDA runtime %d, driver %d, %d visible device(s), %u hardware threads\n", rt, drv,
              num_devices, std::thread::hardware_concurrency());
  for (int d = 0; d < num_devices; ++d) {
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, d));
    std::printf("  dev%d: %s (sm_%d%d)\n", d, p.name, p.major, p.minor);
    CK(cudaSetDevice(d));
    CK(cudaFree(nullptr));  // create the context up front
  }
  CK(cudaSetDevice(0));

  // Today's stack: slabs carved from 1 MiB blocks of a fixed-size pinned resource.
  rmm::mr::pinned_host_memory_resource pinned_mr;
  cucascade::memory::fixed_size_host_memory_resource upstream{0, pinned_mr, 1 * GiB, 1 * GiB,
                                                             1 * MiB, 128, 1};
  cucascade::memory::small_pinned_host_memory_resource small{upstream};
  current_alloc cur{small};

  cudaMemPool_t pool = make_pool(8 * GiB);
  pool_alloc pl{pool};

  cudaStream_t s{};
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  scenario_large(cur, pl, s);
  std::fflush(stdout);
  scenario_small(cur, pl, s);
  std::fflush(stdout);
  scenario_copy(pool, num_devices);
  std::fflush(stdout);
  scenario_mt(cur, pl, num_devices);

  std::printf("\npool at exit: reserved %.1f MiB, high watermark %.1f MiB\n",
              double(pool_attr(pool, cudaMemPoolAttrReservedMemCurrent)) / double(MiB),
              double(pool_attr(pool, cudaMemPoolAttrReservedMemHigh)) / double(MiB));
  CK(cudaStreamSynchronize(s));
  CK(cudaStreamDestroy(s));
  CK(cudaMemPoolDestroy(pool));
  return 0;
}
