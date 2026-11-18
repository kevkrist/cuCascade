# Can a CUDA pool hold NUMA-placed host memory?

Research for [kevkrist/cuCascade#24](https://github.com/kevkrist/cuCascade/issues/24). Checked
against CUDA 12.9 and 13.2 headers plus NVIDIA docs, 2026-11-18.

## Answer

| Question | Verdict |
| --- | --- |
| (a) Can a `cudaMemPool_t` be backed by our own `cudaHostRegister`ed, `mbind`-placed buffers? | **No.** No API adds or imports caller-owned memory into a pool. The driver allocates the backing memory itself. |
| (b) Host / `HostNuma` pool support | `HostNuma` pools: CUDA 12.2+, driver >= 535.54.03, Linux only until 12.9. `Host` pools (no NUMA node): CUDA 13.0+, driver >= 580.65.06. `HostNumaCurrent` is never valid for pools. Gate on `cudaDevAttrHostNumaMemoryPoolsSupported` (142) / `cudaDevAttrHostMemoryPoolsSupported` (144, 13.x only). No document gives a GPU-architecture limit, so sm_75 support is unknown until the runtime probe. |
| (c) Pinned / accessibility / VA | Pinned (`cudaMemAllocationTypePinned`). Only the host CPU can access it by default; each GPU needs `cudaMemPoolSetAccess` first, and a device access without it is undefined behaviour. The docs imply one pointer for host and device (unconfirmed). What `cudaPointerGetAttributes` reports is not documented, so the probe must check it. |
| (d) Pool knobs | `maxSize` caps the pool (0 = system default). Release threshold defaults to 0, so the pool shrinks at every sync. `TrimTo` never frees backing that outstanding allocations still use. Destroying a pool with allocations still live returns at once; the pool is freed later. Frees are stream-ordered, and the three `Reuse*` policies are on by default. |

**Recommendation:** to get NUMA-placed pinned host memory with stream-ordered alloc/free, let
CUDA allocate it with a `cudaMemLocationTypeHostNuma` pool, or with VMM (`cuMemCreate` +
`HOST_NUMA`). If we must keep our own `mbind` + `cudaHostRegister` buffers, we need our own
suballocator. See [Alternatives](#alternatives-to-a).

Notation: `[13.2]` = `.pixi/envs/default/targets/x86_64-linux/include/` (CUDART_VERSION 13020;
`cuda-13-stable` is identical). `[12.9]` = `.pixi/envs/cuda-12-stable/targets/x86_64-linux/include/`
(CUDART_VERSION 12090; install with `pixi install -e cuda-12-stable`).

## (a) Importing existing registered host buffers into a pool

Evidence that there is no import path:

- The pool API consists of only `Create/Destroy/Get|SetAttribute/Get|SetAccess/TrimTo/
  ExportToShareableHandle/ImportFromShareableHandle/ExportPointer/ImportPointer`. CUDA 13 adds
  `cuMemGet|SetMemPool` and `cuMemGetDefaultMemPool` (`grep 'CUresult CUDAAPI cuMemPool' cuda.h`
  in both envs). None of these takes a pointer to memory the caller already owns.
- `cudaMemPoolProps` has only `allocType`, `handleTypes`, `location`, `win32SecurityAttributes`,
  `maxSize`, `usage` and `reserved[54]`. None of them names caller memory (`[13.2] driver_types.h:2289-2302`,
  `[12.9] driver_types.h:2199-2213`).
- `cudaMemPoolImportFromShareableHandle` / `cudaMemPoolImportPointer` are cross-process IPC only.
  An imported pool "do[es] not support creating new allocations", and `ImportPointer` imports
  "a memory pool allocation from another process" (`[13.2] cuda_runtime_api.h:8414`, `:8456`).
- The 13.5 Runtime API reference describes no API that adds already-allocated local memory to a
  pool ([CUDART_MEMORY_POOLS](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html)).
- The VMM route is also closed. `cuMemMap` maps only a `CUmemGenericAllocationHandle`
  (`[13.2] cuda.h:13213`). Such a handle comes from `cuMemCreate`, where the driver allocates
  (`[13.2] cuda.h:13066-13129`), or from `cuMemImportFromShareableHandle`, which takes a handle
  exported by `cuMemExportToShareableHandle` or a graphics API (`[13.2] cuda.h:13446-13494`).
  `cuMemRetainAllocationHandle` works only on ranges "previously mapped by ::cuMemMap"
  (`[13.2] cuda.h:13537-13557`). A `cudaHostRegister` / `cuMemHostRegister` range is not one of
  these, so no generic handle exists for it.
- Pools are not built on top of registered memory either. `cudaHostRegister` memory has its own
  pointer and lifetime model: it is pinned in place and released with `cudaHostUnregister`
  (`[13.2] cuda_runtime_api.h:5000-5089`).

## (b) `Host` / `HostNuma` / `HostNumaCurrent` pool support

| Item | CUDA 12.9 | CUDA 13.2 |
| --- | --- | --- |
| `cudaMemLocationTypeHost` (2), `HostNuma` (3), `HostNumaCurrent` (4) enums exist | yes, `[12.9] driver_types.h:2134-2136` | yes, `[13.2] driver_types.h:2220-2222` |
| `HostNuma` pool | allowed; `id` = host NUMA node id, `[12.9] cuda_runtime_api.h:8984-8986` | allowed, `[13.2] cuda_runtime_api.h:8200-8210` |
| `Host` pool (any node) | **rejected**: "Specifying ::cudaMemLocationTypeHostNumaCurrent or ::cudaMemLocationTypeHost ... will result in ::cudaErrorInvalidValue", `[12.9] cuda_runtime_api.h:8987-8988` | allowed; not IPC capable and `handleTypes` must be 0, `[13.2] cuda_runtime_api.h:8200-8205` |
| `HostNumaCurrent` pool | rejected (same line) | rejected, `[13.2] cuda_runtime_api.h:8209-8210` |
| `cudaDevAttrMemoryPoolsSupported` (115) | `[12.9] driver_types.h:2030` | `[13.2] driver_types.h:2068` |
| `cudaDevAttrHostNumaId` (134), "-1 when system does not support NUMA" | `[12.9] driver_types.h:2049` | `[13.2] driver_types.h:2087` |
| `cudaDevAttrHostNumaMemoryPoolsSupported` (142) | `[12.9] driver_types.h:2055` | `[13.2] driver_types.h:2093` |
| `cudaDevAttrHostMemoryPoolsSupported` (144) | **absent** (compile error, probe below) | `[13.2] driver_types.h:2095` |
| VMM: `CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` (141) | `[12.9] cuda.h:857` | `[13.2] cuda.h:958` |
| VMM: `cuMemCreate` with `HOST` (no node) | not documented; only `HOST_NUMA`, `[12.9] cuda.h:13081` | allowed; `HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` (145), `[13.2] cuda.h:13080-13090`, `:962` |
| `cudaMemGetDefaultMemPool` / `cudaMemSetMemPool` for host locations | absent | present, `[13.2] cuda_runtime_api.h:8291`, `:8342` |
| `cudaMemPoolAttrLocationType` / `MaxPoolSize` pool attributes | absent (enum ends at `UsedMemHigh`, `[12.9] driver_types.h:2125`) | `[13.2] driver_types.h:2195`, `:2204` |

**Versions:**

- CUDA 12.2 added "Host NUMA memory allocation: Allocate a CPU memory targeting a specific NUMA
  node using either the CUDA virtual memory management APIs or the CUDA stream-ordered memory
  allocator." Its minimum Linux driver is >= 535.54.03
  ([12.2 release notes](https://docs.nvidia.com/cuda/archive/12.2.0/cuda-toolkit-release-notes/index.html)).
- Until 12.9 this was Linux-only. 12.9 added Windows WDDM/MCDM: "CPU NUMA allocation support
  through cuMemCreate and cuMemAllocAsync is now available on Windows ... expanding this
  previously Linux-only feature." 12.9 needs driver >= 575.51.03
  ([12.9 release notes](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html)).
- CUDA 13.0 "Added support for cuMemCreate and cudaMallocAsync on the host, with
  CU_MEM_LOCATION_TYPE_HOST". 13.0 needs driver >= 580.65.06
  ([13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)).
- CCCL gates its own host-pool wrapper on CTK >= 12.6 (`[13.2] cccl/cuda/__memory_pool/pinned_memory_pool.h:39`).
  It enables the `Host` (no node) constructor only on CTK >= 13.0 (`:110-129`).

**Turing (sm_75):** no NVIDIA document gives an architecture minimum for host or HostNuma pools.
CUDA 13.0 still supports "all NVIDIA architectures from Turing through Grace Blackwell" (13.0
release notes, above). Support is reported per device and platform through attributes 142 and
144. Example: on Jetson AGX Orin, `host_memory_pools_supported` is False, and a
missing-support case failed only at the first copy
([NVIDIA/cuda-python#2486](https://github.com/NVIDIA/cuda-python/issues/2486)). **Always query
the attribute; never assume.** The sm_75 CI GPU must be checked by the runtime probe ticket.

**Linux:** `cudaDevAttrHostNumaId` returns -1 when the system has no NUMA. For a VMM allocation,
use id 0 on systems without NUMA (`[13.2] cuda.h:13088`). The pool docs give no such fallback.

Local probe (minimal, not the full runtime probe): RTX PRO 4500 Blackwell (sm_120), driver
595.112.07 (CUDA 13.2 driver), single NUMA node 0. Source is in [Appendix](#appendix-probe).

```
CUDA 13.2: rt=13020 drv=13020 pools=1 hostNumaId=0 hostNumaPools=1 hostPools=1
CUDA 12.9: rt=12090 drv=13020 pools=1 hostNumaId=0 hostNumaPools=1 hostPools=0  (attr 144 not compiled: identifier undefined in 12.9)
```

## (c) Properties of the allocations

- **Pinned:** pools must use `allocType = cudaMemAllocationTypePinned`, documented as "cannot
  migrate from its current location while the application is actively using it" (`[13.2]
  driver_types.h:2259-2263`, `:2290`). Managed pools are a separate type (13.x).
- **Host-accessible:** yes. "pools created with ::cudaMemLocationTypeHostNuma or
  ::cudaMemLocationTypeHost, their default accessibility will be from the host CPU" (`[13.2]
  cuda_runtime_api.h:8211-8214`; `[12.9] cuda_runtime_api.h:8990`).
- **Device-accessible only after `cudaMemPoolSetAccess`.** The 12.2 release notes say:
  "Applications must ensure device accesses to pointer backed by HOST allocations from these
  APIs are performed only after they have explicitly requested accessibility for the memory on
  the accessing device. It is undefined behavior to access these host allocations from a device
  without accessibility for the address range, regardless of whether the device supports
  pageable memory access or not." CCCL agrees: "memory is not accessible from devices until
  `cudaMemPoolSetAccess` is called". That is why `cuda::pinned_memory_pool` calls
  `enable_access_from(cuda::devices)` in its constructor
  (`[13.2] cccl/cuda/__memory_pool/pinned_memory_pool.h:116-127`, `:137-150`). Access can be
  granted to every GPU; one pool serves all devices, unlike a per-device copy.
- **Host VA == device VA:** the docs strongly imply it. The 12.2 note speaks of device access
  "to pointer backed by HOST allocations", and `cudaMallocFromPoolAsync` returns a single
  pointer (`[13.2] cuda_runtime_api.h:8378`). No document states it outright, so **the runtime
  probe must confirm it**.
- **Contrast with registered memory:** on UVA systems, `cudaHostRegister` memory may have a
  device address different from its host address ("disjoint addressing",
  `[13.2] cuda_runtime_api.h:8557-8566`). The device pointer equals the host pointer only when
  every visible device has `cudaDevAttrCanUseHostPointerForRegisteredMem`
  (`[13.2] cuda_runtime_api.h:5055-5067`).
- **`cudaPointerGetAttributes`:** documented types are `Unregistered / Host / Device / Managed`.
  For `Host` memory, `device` is the device current at allocation or registration
  (`[13.2] cuda_runtime_api.h:8590-8606`, `driver_types.h:1264-1267`). No document says what a
  host-pool allocation reports. Expect `cudaMemoryTypeHost`, but **the probe must check it**.
  The probe must also check `devicePointer` / `hostPointer` and `device`.
- **vs `cudaHostAlloc` / `cudaHostRegister` flags:**
  - With `cudaHostRegisterDefault` on UVA, registered memory is mapped and portable to every
    context at once (`[13.2] cuda_runtime_api.h:5020-5028`).
  - A host pool instead grants device access explicitly through `cudaMemPoolSetAccess`, for the
    whole pool.
  - The Programming Guide says such access applies to all existing allocations and "cannot be
    revoked"; set it once at pool creation
    ([Programming Guide §4.3.3.3](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#device-accessibility-for-multi-gpu-support)).
  - Host pools have no write-combined, read-only or IO-memory flags.
  - Placement on the NUMA node is done by the driver, not by `mbind` (`location.id`).

## (d) Pool knobs and semantics

- **`cudaMemPoolSetAccess`:** "Controls visibility of pools between devices"; each descriptor
  enables access for one GPU (`[13.2] cuda_runtime_api.h:8166-8179`). The Programming Guide adds:
  - Access applies to all allocations, current and future.
  - For device-located pools, the accessing GPU must be peer capable (`cudaDeviceCanAccessPeer`).
  - Once granted, access should not be toggled
    ([§4.3.3.3](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#device-accessibility-for-multi-gpu-support)).
  - CCCL nevertheless exposes `disable_access_from` (PROT_NONE)
    (`[13.2] cccl/cuda/__memory_pool/memory_pool_base.h:593-596`).
- **`maxSize`:** "When set to 0, defaults to a system dependent value." A non-zero value caps the
  pool (`[13.2] driver_types.h:2300`, `cuda_runtime_api.h:8215-8216`). On 13.x the cap can be
  read back with `cudaMemPoolAttrMaxPoolSize`, which "may be higher than what was initially
  passed ... due to alignment" (`[13.2] driver_types.h:2197-2204`). When the cap is hit,
  allocation fails with `cudaErrorOutOfMemory` (`[13.2] cuda_runtime_api.h:8370-8374`).
- **`cudaMemPoolAttrReleaseThreshold`:** bytes to keep before trying to release memory "on the
  next call to stream, event or context synchronize"; "(default 0)" (`[13.2]
  driver_types.h:2133-2141`). The Programming Guide recommends `UINT64_MAX` to stop the pool
  shrinking at every sync
  ([§4.3.4.2](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#physical-page-caching-behavior)).
  CCCL defaults to `SIZE_MAX` (`[13.2] cccl/cuda/__memory_pool/memory_pool_base.h:278`).
- **`cudaMemPoolTrimTo(pool, minBytesToKeep)`** (`[13.2] cuda_runtime_api.h:8026-8049`):
  - It releases memory to the OS while respecting `minBytesToKeep`.
  - It "cannot release OS allocations that back outstanding asynchronous allocations".
  - Frees whose completion the host has not yet observed "can count as outstanding".
  - Trim works at OS-allocation granularity, not at user-allocation granularity.
- **Destroy with live allocations:** "the function will return immediately and the resources
  associated with the pool will be released automatically once there are no more outstanding
  allocations." Destroying a device's current pool reverts that device to its default pool
  (`[13.2] cuda_runtime_api.h:8256-8273`; `[12.9] :9025-9042`).
- **Stream ordering and reuse:**
  - The allocation "must not be accessed until the allocation operation completes". After a
    free, the pointer must not be used by later work (`[13.2] cuda_runtime_api.h:7967-8024`).
  - Memory freed in a stream can be reused at once by a later allocation on the same stream
    ([§4.3.4.4](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#memory-reuse-policies)).
  - Cross-stream reuse follows three attributes, all "default enabled": `ReuseFollowEventDependencies`,
    `ReuseAllowOpportunistic` and `ReuseAllowInternalDependencies` (`[13.2] driver_types.h:2107-2130`).
  - The last one can insert hidden cross-stream dependencies. The Guide notes that opportunistic
    reuse causes run-to-run variance and internal dependencies can serialize work.
  - Synchronize calls also apply the release threshold
    ([§4.3.4.5](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#synchronization-api-actions)).
  - Host code that touches a host-pool allocation must itself wait for the allocating stream.
    Stream ordering does not cover CPU access.

## Alternatives to (a)

| Option | Pros | Cons |
| --- | --- | --- |
| **CUDA `HostNuma` pool** (`cudaMemPoolCreate`, `location = {HostNuma, node}`, `SetAccess` for all GPUs, release threshold `UINT64_MAX`, optional `maxSize`). Ready-made: CCCL `cuda::pinned_memory_pool(numa_id)`, a `cuda::mr` resource that is both `device_accessible` and `host_accessible` (`[13.2] cccl/cuda/__memory_pool/pinned_memory_pool.h:146-150`, `:182-185`). | Stream-ordered alloc/free, graph capture, and reuse across streams handled by the driver. One pool per NUMA node. Works on 12.2+ (12.9 and 13.2 both fine). | The driver does placement and reservation, so we lose control of `mbind` policy, hugepages and pre-faulting. There is no way to see fragmentation. Up front we cannot rule out device or platform lacking support (attribute 142), and sm_75 is unverified. Pinned growth is set by `maxSize`, but the actual backing granularity is undocumented. |
| **VMM**: `cuMemCreate` with `HOST_NUMA`, plus `cuMemAddressReserve`, `cuMemMap` and `cuMemSetAccess` (12.2+; `HOST` without a node needs 13.0+). | Full control of reservation size, growth and granularity. Same VA on host and device. Can back a growable arena. | Driver API (`libcuda` link or `cudaGetDriverEntryPoint`). No stream ordering: we build our own allocator on top. Allocation granularity comes from `cuMemGetAllocationGranularity`. Needs attribute 141 (or 145 for `HOST`). `cuMemSetAccess` cannot grant `HOST_NUMA` access to non-host allocations (`[13.2] cuda.h:13386`). |
| **Custom suballocator over our `mbind` + `cudaHostRegister` slabs**, exposing `cuda::mr` `async_resource` concepts. | Keeps current placement control and existing buffers; host-only allocation is possible. | We must write stream-ordered free and reuse ourselves (event per free, or a per-stream free list). Registered memory may have a separate device address unless `CanUseHostPointerForRegisteredMem` holds. `cudaHostRegister` of large ranges is slow. More code to maintain. |
| **`cudaMallocHost` / `cudaHostAlloc` slabs + suballocator** (what RMM's pinned resources do). | Simple, mature. | No NUMA control beyond the first-touch / `numactl` policy of the allocating thread. Same suballocator work as above. |

## Open items for the runtime probe ticket

On sm_75 (CI) and multi-NUMA hosts, confirm:

1. Attributes 115, 134, 142 and 144.
2. That `cudaMemPoolCreate(HostNuma)` succeeds.
3. That a device kernel faults or fails without `SetAccess`.
4. Pointer equality between host and device.
5. `cudaPointerGetAttributes` fields.
6. Actual NUMA placement (`move_pages`/`get_mempolicy`).
7. `maxSize` OOM behaviour and the backing chunk size (`ReservedMemCurrent`).

## Appendix: probe

Built with `pixi run -e <env> nvcc -x cu probe.cpp`:

```cpp
#include <cuda_runtime_api.h>
#include <cstdio>
int main() {
  cudaMemPoolProps p{};
  p.allocType = cudaMemAllocationTypePinned;
  p.location  = {cudaMemLocationTypeHostNuma, 0};
  int v[4] = {};
  cudaDeviceGetAttribute(&v[0], cudaDevAttrMemoryPoolsSupported, 0);
  cudaDeviceGetAttribute(&v[1], cudaDevAttrHostNumaId, 0);
  cudaDeviceGetAttribute(&v[2], cudaDevAttrHostNumaMemoryPoolsSupported, 0);
#if CUDART_VERSION >= 13000
  cudaDeviceGetAttribute(&v[3], cudaDevAttrHostMemoryPoolsSupported, 0);
#endif
  int drv = 0; cudaDriverGetVersion(&drv);
  std::printf("rt=%d drv=%d pools=%d hostNumaId=%d hostNumaPools=%d hostPools=%d\n",
              CUDART_VERSION, drv, v[0], v[1], v[2], v[3]);
  (void)p;
}
```
