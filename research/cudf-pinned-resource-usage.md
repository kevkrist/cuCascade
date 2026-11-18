# How libcudf uses its pinned host memory resource

Question: [kevkrist/cuCascade#25](https://github.com/kevkrist/cuCascade/issues/25). Does libcudf
synchronize before host code writes to memory from its pinned resource, or does it write right
after a stream-ordered `allocate(stream)`? What does it require of the resource, and does a
`cudaMemPool`-backed resource have to return memory the host can use immediately?

## Answer

**No. cudf never writes on the host straight after `allocate(stream)`.** Every path where the host
touches pinned memory goes through `cudf::detail::rmm_host_allocator::allocate`, which calls
`mr.allocate(stream, ...)` and then `cudf::detail::sync_stream(stream)` before it returns the
pointer. The one other user (tdigest, through `device_uvector`) writes the memory on the GPU in
stream order and syncs explicitly before the host touches it. Deallocation is the opposite: cudf
calls `deallocate(stream)` **without** synchronizing, sometimes while an async copy that reads
the buffer is still pending. The resource therefore has to be stream-ordered on free.

So a `cudaMemPool`-backed (or event-ordered) resource does **not** have to make `allocate(stream)`
host-immediately-usable. It only has to make the memory safe to use **once `stream` has been
synchronized**, which means any dependency on earlier users is enqueued on `stream`, for example
with `cudaStreamWaitEvent`. It must also order reuse after `deallocate(stream)` behind work
already submitted to that stream. cudf's own default pinned pool (an RMM `pool_memory_resource`)
gives exactly this contract and nothing stronger.

## Versions examined

The installed environments ship headers only, so `.cpp`/`.cu` sources were read from upstream at
the exact commits in the conda build strings:

| Env | libcudf | cudf commit | librmm | rmm commit |
| --- | --- | --- | --- | --- |
| `cuda-13-stable` | 26.10.02 | [`2cd6c6c0`](https://github.com/rapidsai/cudf/tree/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9) (= tag `v26.10.02`) | 26.10.00 | [`5ab04250`](https://github.com/rapidsai/rmm/tree/5ab042503a8c01efa3dba058cbdc9405b84aacfb) (= tag `v26.10.00`) |
| `default` (cuda-13-nightly) | 26.12.00a725 | [`d71b3b74`](https://github.com/rapidsai/cudf/tree/d71b3b743ca6fc9d8d4b1718d845eec4bc63dee6) | 26.12.00a38 | [`cd54ddc5`](https://github.com/rapidsai/rmm/tree/cd54ddc5a7905df21b02ac88497d6dfda26b4249) |

Sources: `.pixi/envs/*/conda-meta/libcudf-*.json` and `librmm-*.json` (the `build` string ends in
the short sha). In both versions these files are byte-identical: `host_vector.hpp`,
`host_memory.hpp`, `vector_factories.hpp` and `src/utilities/host_memory.cpp` in cudf, and
`pinned_host_memory_resource.cpp` and `stream_ordered_memory_resource.hpp` in RMM. The installed
headers match the upstream ones. Below, `cudf@2610` means
`https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/`,
`cudf@2612` means `.../blob/d71b3b743ca6fc9d8d4b1718d845eec4bc63dee6/`, and `rmm@2610` means
`https://github.com/rapidsai/rmm/blob/5ab042503a8c01efa3dba058cbdc9405b84aacfb/`.

## 1. Required ref and property types

- `set_pinned_memory_resource` and `get_pinned_memory_resource` take and return
  `rmm::host_device_async_resource_ref`
  ([`cudf@2610 cpp/include/cudf/utilities/pinned_memory.hpp:21-29`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/include/cudf/utilities/pinned_memory.hpp#L21-L29)).
- `rmm::host_device_async_resource_ref` =
  `cuda::mr::resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>`, the
  stream-ordered `resource_ref` and not `synchronous_resource_ref`
  ([`rmm@2610 cpp/include/rmm/resource_ref.hpp:55-56`](https://github.com/rapidsai/rmm/blob/5ab042503a8c01efa3dba058cbdc9405b84aacfb/cpp/include/rmm/resource_ref.hpp#L55-L56)).
  The resource must therefore provide `allocate(stream, bytes, align)`,
  `deallocate(stream, p, bytes, align)`, `allocate_sync`, `deallocate_sync` and `operator==`,
  plus `get_property` friends for both properties. cudf's own default does exactly this
  ([`cudf@2610 cpp/src/utilities/host_memory.cpp:116-202`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/utilities/host_memory.cpp#L116-L202)).
- `set_pinned_memory_resource` only swaps a global ref under a mutex. It takes no ownership and
  performs no synchronization
  ([`host_memory.cpp:312-325`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/utilities/host_memory.cpp#L312-L325)).
  The caller must keep the resource alive.
- Internally, `rmm_host_allocator` stores the resource as `rmm::host_async_resource_ref`, which
  needs only `host_accessible`. It records `is_device_accessible` from the static type it was
  constructed with
  ([`cudf@2610 cpp/include/cudf/detail/utilities/host_vector.hpp:103-110,199-204`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/include/cudf/detail/utilities/host_vector.hpp#L103-L110)).
  For the pinned resource that flag is true, so `host_span`s are tagged device-accessible and
  `cuda_memcpy*` takes the `copy_pinned` path. Below
  `LIBCUDF_KERNEL_PINNED_COPY_THRESHOLD` (default 0) that path is a GPU kernel that reads the
  host pointer directly; otherwise it is `cudaMemcpyAsync`
  ([`cudf@2610 cpp/src/utilities/cuda_memcpy.cu:32-46`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/utilities/cuda_memcpy.cu#L32-L46),
  [`host_memory.cpp:335-348`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/utilities/host_memory.cpp#L335-L348)).
  The `device_accessible` claim must therefore be true (mapped or portable pinned memory).

## 2. The allocate path: sync after `allocate(stream)`

`rmm_host_allocator<T>::allocate`
([`cudf@2610 host_vector.hpp:141-149`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/include/cudf/detail/utilities/host_vector.hpp#L141-L149), same in 2612):

```cpp
auto const result = mr.allocate(stream, cnt * sizeof(value_type), alignof(value_type));
// Synchronize to ensure the memory is allocated before thrust::host_vector initialization
// TODO: replace thrust::host_vector with a type that does not require synchronization
cudf::detail::sync_stream(stream);
return static_cast<pointer>(result);
```

This was added in cudf PR #15895 ("Pinned vector factory that uses the global pool", commit
`f7ba6ab47a`, 2024-06-11; found with `git log -S`). `thrust::host_vector` value-initializes
elements on the host inside the constructor, so the sync is necessary. Every growth of a
`host_vector` (`reserve`, `push_back` reallocation) goes through `allocate` and syncs again.

Every host-side pinned allocation in cudf goes through this allocator:

| Entry point | Where | Uses pinned resource via |
| --- | --- | --- |
| `cudf::detail::host_vector<T>` | `host_vector.hpp:210-239` | `rmm_host_allocator` |
| `make_empty_pinned_vector`, `make_pinned_vector_async(size)` | [`vector_factories.hpp:488-509`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/include/cudf/detail/utilities/vector_factories.hpp#L488-L509) | `host_vector<T>({get_pinned_memory_resource(), stream})` |
| `make_pinned_vector(size)` (syncs again after), `make_pinned_vector[_async](device_span)` (async D2H after alloc), `make_pinned_vector(host_span)` | [`vector_factories.hpp:521-620`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/include/cudf/detail/utilities/vector_factories.hpp#L521-L620) | `make_pinned_vector_async(size)` |
| `make_host_vector[_async]`, `make_empty_host_vector` | `vector_factories.hpp:376-476` | `get_host_allocator<T>(size, stream)`: pinned if `size*sizeof(T) <= get_allocate_host_as_pinned_threshold()`, else pageable `new`/`delete` ([`host_memory.hpp:30-37`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/include/cudf/detail/utilities/host_memory.hpp#L30-L37)) |
| `hostdevice_vector` / `hostdevice_2dvector` (all IO readers/writers) | [`cudf@2610 cpp/src/io/utilities/hostdevice_vector.hpp:47-51`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/io/utilities/hostdevice_vector.hpp#L47-L51) ([2612: `:59`](https://github.com/rapidsai/cudf/blob/d71b3b743ca6fc9d8d4b1718d845eec4bc63dee6/cpp/src/io/utilities/hostdevice_vector.hpp#L59)) | `h_data{make_pinned_vector_async<T>(size, stream)}` |
| `device_scalar` bounce buffer | [`cudf@2610 cpp/include/cudf/detail/device_scalar.hpp:46-92`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/include/cudf/detail/device_scalar.hpp#L46-L92) | `make_pinned_vector<T>(1, stream)` |
| `streaming_hash_join` comparators | [`cudf@2610 cpp/src/join/streaming_hash_join.cu:62-76`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/join/streaming_hash_join.cu#L62-L76) | `std::vector<Eq, rmm_host_allocator<Eq>>{{get_pinned_memory_resource(), stream}}`, then `stream.sync()` before freeing |
| `transform` (JIT) argument staging | [`cudf@2610 cpp/src/transform/transform.cu:511-539`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/transform/transform.cu#L511-L539) | `host_vector({get_pinned_memory_resource(), stream})`, then sync `make_device_uvector` |

There are 91 `make_pinned_vector*` call sites in `cpp/src` + `cpp/include` at 26.10. Typical
patterns write on the host right after the factory returns:
`null_mask.cu:203-224` (`thrust::tabulate(thrust::host, ...)` into
`make_pinned_vector_async` output),
[`concatenate.cu:82-93`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/copying/concatenate.cu#L82-L93),
and [`column_device_view.cu:73-91`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/column/column_device_view.cu#L73-L91).
These writes are safe only because of the sync inside `rmm_host_allocator::allocate`. The
factory's `_async` suffix refers to the subsequent copy, not to the allocation.

**Non-host_vector use (tdigest).** `get_pinned_memory_resource()` also feeds `device_uvector`s
in tdigest when the CPU computes clusters
([`cudf@2610 cpp/src/quantiles/tdigest/tdigest_aggregation.cu:743-745, 1138-1155, 1538-1567`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/quantiles/tdigest/tdigest_aggregation.cu#L743-L745)).
Those buffers are filled by GPU work in stream order, and the code calls
`cudf::detail::sync_stream(stream)` before host code touches them. Line 794 says so explicitly:
"sync required after compute_cluster_starts() and before generate_cluster_limits()"
([`:794-795`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/quantiles/tdigest/tdigest_aggregation.cu#L794-L795);
2612 at [`:802`](https://github.com/rapidsai/cudf/blob/d71b3b743ca6fc9d8d4b1718d845eec4bc63dee6/cpp/src/quantiles/tdigest/tdigest_aggregation.cu#L802)).

**`allocate_sync` is never called** on the pinned resource by these paths. All of them use
`allocate(stream)` + `sync_stream(stream)`. cudf's own default `allocate_sync` doesn't even
synchronize: it is `allocate(cudaStreamDefault, ...)`
([`host_memory.cpp:126-129`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/utilities/host_memory.cpp#L126-L129)),
and so is the test fixture's `pinned_pool`
([`cudf@2610 cpp/include/cudf_test/testing_main.hpp:56-59`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/include/cudf_test/testing_main.hpp#L56-L59)).

Complete list of `get_pinned_memory_resource()` callers in `cpp/src` and `cpp/include` (26.10;
26.12 has the same set at shifted lines): `host_memory.hpp:34`, `vector_factories.hpp:490,508`,
`tdigest_aggregation.cu:744,1139,1539`, `streaming_hash_join.cu:64`, `transform.cu:513`.

## 3. The deallocate path: stream-ordered, no sync

`rmm_host_allocator::deallocate` is just `mr.deallocate(stream, p, bytes, align)`, using the
stream captured at construction, with no sync
([`host_vector.hpp:161-164`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/include/cudf/detail/utilities/host_vector.hpp#L161-L164)).
cudf does free pinned buffers while async work that reads them is still pending. In
`set_null_masks`, `h_number_of_mask_words` (pinned, `null_mask.cu:203-204`) is the source of
`make_device_uvector_async` (`:235-236`, an async `cudaMemcpyAsync` or copy kernel); the kernel
then launches and the function returns without a sync, which destroys the host vector
([`cudf@2610 cpp/src/bitmask/null_mask.cu:203-250`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/bitmask/null_mask.cu#L203-L250);
same in 2612 at [`:217-263`](https://github.com/rapidsai/cudf/blob/d71b3b743ca6fc9d8d4b1718d845eec4bc63dee6/cpp/src/bitmask/null_mask.cu#L217-L263)).
`device_scalar::set_value_async` has the same shape: it writes the pinned bounce buffer, then
does an async H2D copy, and the scalar may be destroyed before that copy finishes
(`device_scalar.hpp:72-82`). The pinned resource must therefore not let the host or another
stream reuse a freed block before work already queued on the freeing stream completes. Some
callers sync before freeing anyway (`streaming_hash_join.cu:76`, `partitioning.cu:560`,
`concatenate.cu:93`), but cudf does not do this consistently.

## 4. cudf's default pinned resource and whether it syncs

The default is `pinned_pool_with_fallback_memory_resource`: an RMM `pool_memory_resource` over
`rmm::mr::pinned_host_memory_resource`, falling back to direct pinned allocation when the pool is
exhausted. Initial size is `min(0.5% of device memory, 64 MiB)` (env `LIBCUDF_PINNED_POOL_SIZE`)
and max size is 16x that (env `LIBCUDF_PINNED_POOL_MAX_SIZE`)
([`host_memory.cpp:79-232`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/utilities/host_memory.cpp#L79-L232)).
It does **not** sync on `allocate(stream)`:

- `pinned_host_memory_resource::allocate(stream)` ignores the stream and calls
  `cudaHostAlloc(..., cudaHostAllocDefault)`, which is synchronous host memory. `deallocate` calls
  `cudaFreeHost`. Only `allocate_sync` adds a `cudaStreamSynchronize(nullptr)`
  ([`rmm@2610 cpp/src/mr/pinned_host_memory_resource.cpp:18-50`](https://github.com/rapidsai/rmm/blob/5ab042503a8c01efa3dba058cbdc9405b84aacfb/cpp/src/mr/pinned_host_memory_resource.cpp#L18-L50)).
- The pool (`stream_ordered_memory_resource`) is stream-ordered.
  `deallocate(stream)` records an event on `stream` and puts the block on that stream's free list
  ([`rmm@2610 cpp/include/rmm/mr/detail/stream_ordered_memory_resource.hpp:128-149`](https://github.com/rapidsai/rmm/blob/5ab042503a8c01efa3dba058cbdc9405b84aacfb/cpp/include/rmm/mr/detail/stream_ordered_memory_resource.hpp#L128-L149)).
  `allocate(stream)` first reuses a block from the same stream's list ("no sync required",
  `:387-393`). Otherwise it takes a block from another stream after
  `cudaStreamWaitEvent(stream, other_event)`, a **GPU-side** wait (`:432-460`, `:487-492`), or
  grows the pool. No host sync happens anywhere on `allocate(stream)`. Only `allocate_sync` and
  `deallocate_sync` call `cudaStreamSynchronize` (`:162-186`).

So a block returned by cudf's default `allocate(stream)` can still be read by a pending
same-stream copy, or by another stream's work that `stream` was only told to wait for. Host
writes are safe only after `stream` is synchronized, and that is exactly what
`rmm_host_allocator::allocate` does. The contract cudf relies on is therefore "ordered on
`stream`", not "host-ready on return".

## 5. Threshold knobs

- `set_allocate_host_as_pinned_threshold` / `get_allocate_host_as_pinned_threshold`: an atomic,
  default `LIBCUDF_ALLOCATE_HOST_AS_PINNED_THRESHOLD` or 0, meaning "all `make_host_vector*` use
  pageable" ([`host_memory.cpp:350-363`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/utilities/host_memory.cpp#L350-L363)).
  It is consulted only by `get_host_allocator`
  ([`host_memory.hpp:33`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/include/cudf/detail/utilities/host_memory.hpp#L33)),
  i.e. `make_host_vector*` / `make_empty_host_vector`. `make_pinned_vector*`, `hostdevice_vector`,
  `device_scalar`, tdigest, streaming join and transform **always** use the pinned resource,
  whatever their size. The pageable fallback resource is `new`/`delete` and ignores the stream
  (`host_memory.cpp:265-308,367-372`).
- `set_kernel_pinned_copy_threshold`: see section 1. Below it, pinned copies are kernels that
  dereference the host pointer from the GPU.

## 6. kvikio / IO paths

cudf's file datasource and sink hand pointers to kvikio
(`FileHandle::pread`/`pwrite`, `MmapHandle`) and never pass cudf's pinned resource to kvikio
([`cudf@2610 cpp/src/io/utilities/datasource.cpp:43-186`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/io/utilities/datasource.cpp#L43-L186),
[`data_sink.cpp:27-81`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/io/utilities/data_sink.cpp#L27-L81);
grep finds no `get_pinned_memory_resource` in either file). The IO layers use cudf's pinned
resource only through `hostdevice_vector` and `make_pinned_vector*`, for example the ORC and
Parquet writer bounce buffers
([`orc/writer_impl.cu:2621`](https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/io/orc/writer_impl.cu#L2621),
`parquet/writer_impl.cu:2418`). Section 2 covers those, so they get the same
allocate-then-sync behaviour.

## 7. cuCascade context

- cuCascade does not call `cudf::set_pinned_memory_resource` anywhere in `src/`, `include/`,
  `test/` or `benchmark/` (grep at `origin/main` `08aaf04`). The only mention is the usage example
  in the docs of
  [`include/cucascade/memory/small_pinned_host_memory_resource.hpp:47-53`](../include/cucascade/memory/small_pinned_host_memory_resource.hpp),
  which shows a host program (e.g. Sirius) installing it together with
  `set_allocate_host_as_pinned_threshold(MAX_SLAB_SIZE)`.
- `small_pinned_host_memory_resource` already matches the cudf contract.
  `deallocate(stream)` records an event on the freeing stream, or calls `cudaStreamSynchronize`
  if no event is available. `allocate(stream)` makes `stream` `cudaStreamWaitEvent` on that event
  before handing the slab out
  ([`src/memory/small_pinned_host_memory_resource.cpp:143-230`](../src/memory/small_pinned_host_memory_resource.cpp)).
  Like RMM's pool, it is ready only after `stream` syncs, not host-ready on return, and that is
  enough for cudf because of section 2. Allocations above 8 KiB use synchronous
  `cudaHostAlloc(Portable|Mapped)` / `cudaFreeHost` (`:157-163`, `:196-199`). Its
  `allocate_sync` does sync the default stream (`.hpp:104-109`), which is stricter than cudf's
  default resource.

## Conclusion

A `cudaMemPool`-backed pinned resource (`cudaMallocFromPoolAsync` on a host-located pool)
installed with `cudf::set_pinned_memory_resource` is correct for cudf 26.10 and 26.12 if:

1. **`allocate(stream)`**: the memory is safe for host and device use once `stream` has
   completed. Any dependency on earlier frees must be expressed on `stream` (stream-ordered
   allocation semantics already do this). Host-immediate readiness is **not** required, because
   cudf syncs `stream` immediately after every allocation it touches from the host.
2. **`deallocate(stream)`**: it must be stream-ordered (`cudaFreeAsync` is), because cudf frees
   pinned buffers with async copies still pending on `stream`.
3. It satisfies `host_accessible` and `device_accessible`, and the memory really is
   device-accessible. cudf may read it from kernels.
4. It is thread-safe and outlives every cudf call. `set_pinned_memory_resource` stores a
   non-owning ref.

`allocate_sync` is not called by cudf's pinned paths, and cudf's own default implements it
without a sync. Do not rely on it for correctness either way.
