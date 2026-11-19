# PROTOTYPE (throwaway, never merged): chunked slab vs one-pool-allocation-per-block

Question (from "Slab class API"): should the slab carve blocks from big chunks it caches
itself, or make one HostNuma `cudaMemPool_t` allocation per block and let the pool do the caching?

- `bench.cpp`: both designs on the same pool. Scenario 1: steady-state cost per block. Scenario 2:
  cold growth. Scenario 3: memory left idle after a spike, against a contiguous request on the same pool.
  Output: `out.txt`.
- `diag.cpp`: splits per-block cost into alloc, sync and free, with reuse policies on and off,
  plus a device-pool control. Output: `diag.txt`.

Machine: 2x RTX PRO 4500 Blackwell (sm_120), 1 NUMA node, 48 cores, CUDA 13.2 (pixi default env).
Build: `$CXX -O2 -std=c++20 -I$CONDA_PREFIX/targets/x86_64-linux/include -L$CONDA_PREFIX/lib bench.cpp -lcudart -lpthread`
