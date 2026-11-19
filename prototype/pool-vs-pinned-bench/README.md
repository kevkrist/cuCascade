# PROTOTYPE (throwaway, never merged): HostNuma `cudaMemPool_t` vs today's pinned path

Task from "Benchmark pool vs current pinned path": does the pool beat
`small_pinned_host_memory_resource` (slabs up to 8 KiB, `cudaHostAlloc` above that)?

- `bench.cpp`: "current" is the real `small_pinned_host_memory_resource` from `libcucascade.so`
  (main at 9e6cdc8). "pool" is a raw HostNuma `cudaMemPool_t` (node 0, pinned, release threshold
  max, access for all visible GPUs, 8 GiB limit), the same driver calls the planned resource wraps.
  1. Repeated large requests: allocate(stream) + stream sync + free(stream), as cudf does.
  2. Small allocations (up to 8 KiB), with and without the sync, and 64 live at once.
  3. H2D/D2H copy bandwidth for each kind of host buffer.
  4. Multi-threaded allocate/free at 1 to 32 threads, one stream per thread.
- `out.txt`: full output of the run below.
- `build.sh`: builds against an existing `build/release` (`REPO=<checkout> ./build.sh`).

Machine: 2x RTX PRO 6000 Blackwell (sm_120), 1 NUMA node, 48 hardware threads, 64 GB RAM,
CUDA runtime 13.2, driver 13.4 (pixi default env), machine otherwise idle.
Run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 ./bench` (a third, small GPU is hidden).

Not covered: cold pool growth from many threads at once, multi-NUMA, sm_75, the CCCL wrapper's
own overhead, real cudf workloads.
