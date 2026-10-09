/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/cuda/stream.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/notification_channel.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <rmm/aligned.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace cucascade {
namespace memory {

using impl_type                    = detail::reservation_aware_resource_adaptor_impl;
using stream_ordered_tracker_state = impl_type::stream_ordered_tracker_state;
using device_reserved_arena        = impl_type::device_reserved_arena;

namespace {

struct stream_ordered_allocation_tracker : public impl_type::allocation_tracker_iface {
  mutable std::mutex mutex;
  std::unordered_map<cudaStream_t, std::shared_ptr<stream_ordered_tracker_state>> states;

  ~stream_ordered_allocation_tracker() override
  {
    for (auto const& [stream, state] : states) {
      state->close();
    }
  }

  std::shared_ptr<stream_ordered_tracker_state> reset_tracker_state(
    ::cuda::stream_ref stream) override
  {
    std::lock_guard lock(mutex);
    auto node = states.extract(stream.get());
    return node.empty() ? nullptr : std::move(node.mapped());
  }

  void assign_reservation_to_tracker(::cuda::stream_ref stream,
                                     std::unique_ptr<device_reserved_arena> arena,
                                     std::unique_ptr<reservation_limit_policy> policy,
                                     std::unique_ptr<oom_handling_policy> oom_policy) override
  {
    auto state = stream_ordered_tracker_state::create(
      std::move(arena), std::move(policy), std::move(oom_policy));
    std::lock_guard lock(mutex);
    if (!states.emplace(stream.get(), state).second) {
      throw rmm::logic_error("Stream already has reservation state set");
    }
  }

  std::shared_ptr<stream_ordered_tracker_state> get_tracker_state(
    ::cuda::stream_ref stream) const override
  {
    std::lock_guard lock(mutex);
    auto it = states.find(stream.get());
    return it == states.end() ? nullptr : it->second;
  }
};

struct ptds_allocation_tracker : public impl_type::allocation_tracker_iface {
  // A unique identity prevents a closed TLS entry from matching a later adaptor at the same
  // address.
  static inline std::atomic<std::uint64_t> next_identity{0};
  std::uint64_t const identity{next_identity.fetch_add(1)};
  struct tls_entry {
    std::shared_ptr<stream_ordered_tracker_state> state;
    explicit tls_entry(std::shared_ptr<stream_ordered_tracker_state> value)
      : state(std::move(value))
    {
    }
    tls_entry(tls_entry&&) noexcept        = default;
    tls_entry(tls_entry const&)            = delete;
    tls_entry& operator=(tls_entry const&) = delete;
    ~tls_entry()
    {
      if (state) { state->close(); }
    }
  };
  using state_map = std::unordered_map<std::uint64_t, tls_entry>;
  static state_map& tls_states()
  {
    thread_local state_map map;
    return map;
  }

  mutable std::mutex registry_mutex;
  std::unordered_map<std::thread::id, std::shared_ptr<stream_ordered_tracker_state>> registry;

  ~ptds_allocation_tracker() override
  {
    // Other threads can retain TLS entries after this resource dies. Close their arenas first.
    for (auto const& [thread, state] : registry) {
      state->close();
    }
  }

  std::shared_ptr<stream_ordered_tracker_state> reset_tracker_state(::cuda::stream_ref) override
  {
    auto node = tls_states().extract(identity);
    {
      std::lock_guard lock(registry_mutex);
      registry.erase(std::this_thread::get_id());
    }
    return node.empty() ? nullptr : std::move(node.mapped().state);
  }

  void assign_reservation_to_tracker(::cuda::stream_ref,
                                     std::unique_ptr<device_reserved_arena> arena,
                                     std::unique_ptr<reservation_limit_policy> policy,
                                     std::unique_ptr<oom_handling_policy> oom_policy) override
  {
    auto state = stream_ordered_tracker_state::create(
      std::move(arena), std::move(policy), std::move(oom_policy));
    auto& map = tls_states();
    if (!map.emplace(identity, tls_entry{state}).second) {
      throw rmm::logic_error("Thread already has reservation state set");
    }
    std::shared_ptr<stream_ordered_tracker_state> replaced;
    try {
      std::lock_guard lock(registry_mutex);
      auto [entry, inserted] = registry.try_emplace(std::this_thread::get_id(), state);
      if (!inserted) { replaced = std::exchange(entry->second, state); }
    } catch (...) {
      map.erase(identity);
      throw;
    }
  }

  std::shared_ptr<stream_ordered_tracker_state> get_tracker_state(::cuda::stream_ref) const override
  {
    auto const& map = tls_states();
    auto it         = map.find(identity);
    return it == map.end() ? nullptr : it->second.state;
  }
};

thread_local stream_ordered_tracker_state* policy_locked_state = nullptr;

// Recovers the CUDA memory pool backing an upstream resource, for accurate OOM
// diagnostics. cuda::mr::resource_cast does an exact-type downcast of the wrapped
// concrete resource (returning nullptr on mismatch), so we probe the RMM resources
// that own a pool and expose pool_handle(); any other upstream (e.g. a plain
// cudaMalloc-backed resource) yields a null handle.
[[nodiscard]] cudaMemPool_t extract_pool_handle(rmm::device_async_resource_ref upstream) noexcept
{
  if (auto* mr = ::cuda::mr::resource_cast<rmm::mr::cuda_async_memory_resource>(&upstream)) {
    return mr->pool_handle();
  }
  if (auto* mr = ::cuda::mr::resource_cast<rmm::mr::cuda_async_view_memory_resource>(&upstream)) {
    return mr->pool_handle();
  }
  if (auto* mr =
        ::cuda::mr::resource_cast<rmm::mr::cuda_async_managed_memory_resource>(&upstream)) {
    return mr->pool_handle();
  }
  return nullptr;
}

}  // namespace

stream_ordered_tracker_state::stream_ordered_tracker_state(
  std::unique_ptr<device_reserved_arena> arena,
  std::unique_ptr<reservation_limit_policy> res_policy,
  std::unique_ptr<oom_handling_policy> oom_policy)
  : memory_reservation(std::move(arena)),
    reservation_policy(std::move(res_policy)),
    oom_policy(std::move(oom_policy))
{
}

std::shared_ptr<stream_ordered_tracker_state> stream_ordered_tracker_state::create(
  std::unique_ptr<device_reserved_arena> arena,
  std::unique_ptr<reservation_limit_policy> policy,
  std::unique_ptr<oom_handling_policy> oom_policy)
{
  auto state = std::make_shared<stream_ordered_tracker_state>(
    std::move(arena), std::move(policy), std::move(oom_policy));
  state->memory_reservation->_owner = state;
  return state;
}

stream_ordered_tracker_state::~stream_ordered_tracker_state() noexcept { close(); }

void stream_ordered_tracker_state::close() noexcept
{
  std::unique_ptr<device_reserved_arena> detached;
  {
    std::lock_guard lock(arbitration_mutex);
    if (closed) { return; }
    // Release counters before publishing closure. Deferred notifier destruction cannot use the
    // adaptor.
    memory_reservation->_impl->do_release_reservation(memory_reservation.get());
    memory_reservation->_impl = nullptr;
    closed                    = true;
    detached                  = std::move(memory_reservation);
  }
  // The arena's remaining destructor only invokes its notifier, outside every accounting lock.
}

thread_local impl_type::allocation_context* impl_type::allocation_context::_current = nullptr;

impl_type::allocation_context::allocation_context(impl_type& owner, ::cuda::stream_ref stream)
  : _owner(&owner),
    _origin(owner._allocation_tracker->get_tracker_state(stream)),
    _previous(_current),
    _thread(std::this_thread::get_id())
{
  _current = this;
}

impl_type::allocation_context::allocation_context(allocation_context&& other) noexcept
  : _owner(other._owner),
    _origin(std::move(other._origin)),
    _previous(other._previous),
    _thread(other._thread)
{
  if (!_owner) { return; }
  if (_thread != std::this_thread::get_id()) { std::terminate(); }
  auto** link = &_current;
  while (*link && *link != &other) {
    link = &(*link)->_previous;
  }
  if (*link != &other) { std::terminate(); }
  *link        = this;
  other._owner = nullptr;
}

impl_type::allocation_context::~allocation_context() noexcept
{
  if (!_owner) { return; }
  if (_thread != std::this_thread::get_id()) { std::terminate(); }
  auto** link = &_current;
  while (*link && *link != this) {
    link = &(*link)->_previous;
  }
  if (*link != this) { std::terminate(); }
  *link = _previous;
}

std::shared_ptr<stream_ordered_tracker_state> impl_type::allocation_origin(
  ::cuda::stream_ref stream) const
{
  for (auto* context = allocation_context::_current; context; context = context->_previous) {
    if (context->_owner == this) { return context->_origin; }
  }
  return _allocation_tracker->get_tracker_state(stream);
}

impl_type::reservation_aware_resource_adaptor_impl(
  memory_space_id space_id,
  rmm::device_async_resource_ref upstream,
  std::size_t capacity,
  std::unique_ptr<reservation_limit_policy> default_reservation_policy,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  AllocationTrackingScope tracking_scope,
  cudaMemPool_t pool_handle)
  : _space_id(space_id),
    _upstream(std::move(upstream)),
    _pool_handle(pool_handle != nullptr ? pool_handle : extract_pool_handle(_upstream)),
    _memory_limit(capacity),
    _capacity(capacity),
    _allocation_tracker([&]() -> std::unique_ptr<allocation_tracker_iface> {
      if (tracking_scope == AllocationTrackingScope::PER_STREAM) {
        return std::make_unique<stream_ordered_allocation_tracker>();
      } else {
        return std::make_unique<ptds_allocation_tracker>();
      }
    }()),
    _default_reservation_policy(default_reservation_policy
                                  ? std::move(default_reservation_policy)
                                  : make_default_reservation_limit_policy()),
    _default_oom_policy(default_oom_policy ? std::move(default_oom_policy)
                                           : make_default_oom_policy())
{
}

impl_type::reservation_aware_resource_adaptor_impl(
  memory_space_id space_id,
  rmm::device_async_resource_ref upstream,
  std::size_t memory_limit,
  std::size_t capacity,
  std::unique_ptr<reservation_limit_policy> default_reservation_policy,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  AllocationTrackingScope tracking_scope,
  cudaMemPool_t pool_handle)
  : _space_id(space_id),
    _upstream(std::move(upstream)),
    _pool_handle(pool_handle != nullptr ? pool_handle : extract_pool_handle(_upstream)),
    _memory_limit(memory_limit),
    _capacity(capacity),
    _allocation_tracker([&]() -> std::unique_ptr<allocation_tracker_iface> {
      if (tracking_scope == AllocationTrackingScope::PER_STREAM) {
        return std::make_unique<stream_ordered_allocation_tracker>();
      } else {
        return std::make_unique<ptds_allocation_tracker>();
      }
    }()),
    _default_reservation_policy(default_reservation_policy
                                  ? std::move(default_reservation_policy)
                                  : make_default_reservation_limit_policy()),
    _default_oom_policy(default_oom_policy ? std::move(default_oom_policy)
                                           : make_default_oom_policy())
{
}

impl_type::~reservation_aware_resource_adaptor_impl()
{
  // Arenas refer back to these counters, so detach them before member destruction begins.
  _allocation_tracker.reset();
}

rmm::device_async_resource_ref impl_type::get_upstream_resource() const noexcept
{
  return _upstream;
}

std::size_t impl_type::get_available_memory() const noexcept
{
  auto current_bytes = _total_allocated_bytes.load();
  return _capacity > current_bytes ? _capacity - current_bytes : 0;
}

std::size_t impl_type::get_available_memory(::cuda::stream_ref stream) const noexcept
{
  auto upstream_available_memory = get_available_memory();
  if (auto state = _allocation_tracker->get_tracker_state(stream)) {
    std::lock_guard lock(state->arbitration_mutex);
    if (state->memory_reservation) {
      upstream_available_memory += state->memory_reservation->get_available_memory();
    }
  }
  return upstream_available_memory;
}

std::size_t impl_type::get_available_memory_print(::cuda::stream_ref stream) const noexcept
{
  auto upstream_available_memory = get_available_memory();
  if (auto state = _allocation_tracker->get_tracker_state(stream)) {
    std::lock_guard lock(state->arbitration_mutex);
    if (state->memory_reservation) {
      upstream_available_memory += state->memory_reservation->get_available_memory();
    }
  }
  return upstream_available_memory;
}

std::size_t impl_type::get_allocated_bytes(::cuda::stream_ref stream) const
{
  auto stats = _allocation_tracker->get_tracker_state(stream);
  if (!stats) { return 0; }
  std::lock_guard lock(stats->arbitration_mutex);
  return stats->memory_reservation
           ? static_cast<std::size_t>(
               std::max(int64_t{0}, stats->memory_reservation->allocated_bytes.load()))
           : 0;
}

std::size_t impl_type::get_peak_allocated_bytes(::cuda::stream_ref stream) const
{
  auto stats = _allocation_tracker->get_tracker_state(stream);
  if (!stats) { return 0; }
  std::lock_guard lock(stats->arbitration_mutex);
  return stats->memory_reservation
           ? static_cast<std::size_t>(
               std::max(int64_t{0}, stats->memory_reservation->peak_allocated_bytes.peak()))
           : 0;
}

std::size_t impl_type::get_total_allocated_bytes() const { return _total_allocated_bytes.load(); }

std::size_t impl_type::get_peak_total_allocated_bytes() const
{
  return _peak_total_allocated_bytes.peak();
}

void impl_type::reset_peak_allocated_bytes(::cuda::stream_ref stream)
{
  auto stats = _allocation_tracker->get_tracker_state(stream);
  if (stats) {
    std::lock_guard lock(stats->arbitration_mutex);
    if (stats->memory_reservation) { stats->memory_reservation->peak_allocated_bytes.reset(0); }
  }
}

std::size_t impl_type::get_total_reserved_bytes() const { return _total_reserved_bytes.load(); }

bool impl_type::is_stream_tracked(::cuda::stream_ref stream) const
{
  return _allocation_tracker->get_tracker_state(stream) != nullptr;
}

bool impl_type::attach_reservation_to_tracker(
  ::cuda::stream_ref stream,
  std::unique_ptr<reservation> reserved_bytes,
  std::unique_ptr<reservation_limit_policy> stream_reservation_policy,
  std::unique_ptr<oom_handling_policy> stream_oom_policy)
{
  if (!reserved_bytes) { return false; }
  auto* arena = dynamic_cast<device_reserved_arena*>(reserved_bytes->_arena.get());
  if (!arena || arena->_impl != this) { return false; }
  auto stats = _allocation_tracker->get_tracker_state(stream);
  if (stats) { return false; }

  if (!stream_reservation_policy) {
    stream_reservation_policy = make_default_reservation_limit_policy();
  }

  if (!stream_oom_policy) { stream_oom_policy = make_default_oom_policy(); }

  _allocation_tracker->assign_reservation_to_tracker(
    stream,
    std::unique_ptr<device_reserved_arena>(
      dynamic_cast<device_reserved_arena*>(reserved_bytes->_arena.release())),
    std::move(stream_reservation_policy),
    std::move(stream_oom_policy));

  return true;
}
void impl_type::reset_stream_reservation(::cuda::stream_ref stream)
{
  if (auto state = _allocation_tracker->reset_tracker_state(stream)) { state->close(); }
}

std::unique_ptr<reserved_arena> impl_type::reserve(std::size_t bytes,
                                                   std::unique_ptr<event_notifier> release_notifer)
{
  if (do_reserve(bytes, _memory_limit)) {
    _number_of_allocations.fetch_add(1);
    try {
      return std::make_unique<device_reserved_arena>(*this, bytes, std::move(release_notifer));
    } catch (...) {
      _number_of_allocations.fetch_sub(1);
      _total_reserved_bytes.fetch_sub(bytes);
      _total_allocated_bytes.sub(bytes);
      throw;
    }
  }
  return nullptr;
}

std::unique_ptr<reserved_arena> impl_type::reserve_upto(
  std::size_t bytes, std::unique_ptr<event_notifier> release_notifer)
{
  auto reserved_size = do_reserve_upto(bytes, _memory_limit);
  _number_of_allocations.fetch_add(1);
  try {
    return std::make_unique<device_reserved_arena>(
      *this, reserved_size, std::move(release_notifer));
  } catch (...) {
    _number_of_allocations.fetch_sub(1);
    _total_reserved_bytes.fetch_sub(reserved_size);
    _total_allocated_bytes.sub(reserved_size);
    throw;
  }
}

bool impl_type::grow_reservation_by(device_reserved_arena& arena, std::size_t bytes)
{
  auto owner = arena._owner.lock();
  std::unique_lock<std::mutex> lock;
  if (owner && policy_locked_state != owner.get()) {
    lock = std::unique_lock(owner->arbitration_mutex);
  }
  if (owner && owner->closed) { return false; }
  auto old_size = static_cast<std::size_t>(arena.size());
  if (bytes > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) - old_size) {
    return false;
  }
  auto live                 = static_cast<std::size_t>(arena.allocated_bytes.load());
  auto new_size             = old_size + bytes;
  auto delta                = std::max(new_size, live) - std::max(old_size, live);
  auto [success, committed] = _total_allocated_bytes.try_add(delta, _memory_limit);
  if (!success) { return false; }
  _total_reserved_bytes.fetch_add(bytes);
  arena._size = static_cast<int64_t>(new_size);
  _peak_total_allocated_bytes.update_peak(committed);
  return true;
}

void impl_type::shrink_reservation_to_fit(device_reserved_arena& arena)
{
  auto owner = arena._owner.lock();
  std::unique_lock<std::mutex> lock;
  if (owner && policy_locked_state != owner.get()) {
    lock = std::unique_lock(owner->arbitration_mutex);
  }
  if (owner && owner->closed) { return; }
  auto live = arena.allocated_bytes.load();
  if (live < arena.size()) {
    auto reclaimed = static_cast<std::size_t>(std::exchange(arena._size, live) - live);
    _total_allocated_bytes.sub(reclaimed);
    _total_reserved_bytes.fetch_sub(reclaimed);
  }
}

std::size_t impl_type::get_active_reservation_count() const noexcept
{
  return _number_of_allocations.load();
}

void impl_type::admit(std::shared_ptr<stream_ordered_tracker_state> const& origin,
                      std::size_t padded_bytes,
                      ::cuda::stream_ref stream)
{
  std::unique_lock<std::mutex> lock;
  std::size_t global_delta     = padded_bytes;
  device_reserved_arena* arena = nullptr;
  std::size_t live             = 0;
  if (origin) {
    lock = std::unique_lock(origin->arbitration_mutex);
    if (origin->closed) { throw rmm::logic_error("Allocation context reservation has been reset"); }
    arena = origin->memory_reservation.get();
    live  = static_cast<std::size_t>(arena->allocated_bytes.load());
    if (padded_bytes > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) - live) {
      throw rmm::out_of_memory("Reservation allocation counter overflow");
    }
    if (live + padded_bytes > static_cast<std::size_t>(arena->size()) &&
        origin->reservation_policy) {
      auto* previous = std::exchange(policy_locked_state, origin.get());
      try {
        origin->reservation_policy->handle_over_reservation(stream, padded_bytes, live, arena);
      } catch (...) {
        policy_locked_state = previous;
        throw;
      }
      policy_locked_state = previous;
    }
    auto reserved = static_cast<std::size_t>(arena->size());
    global_delta  = std::max(reserved, live + padded_bytes) - std::max(reserved, live);
  }
  auto [success, committed] = _total_allocated_bytes.try_add(global_delta, _capacity);
  if (!success) {
    throw cucascade_out_of_memory("not enough capacity to allocate memory",
                                  MemoryError::LIMIT_EXCEEDED,
                                  padded_bytes,
                                  committed,
                                  _pool_handle);
  }
  if (arena) {
    auto next = arena->allocated_bytes.add(static_cast<int64_t>(padded_bytes));
    arena->peak_allocated_bytes.update_peak(next);
  }
  _peak_total_allocated_bytes.update_peak(committed);
}

void impl_type::release_charge(std::shared_ptr<stream_ordered_tracker_state> const& origin,
                               std::size_t padded_bytes) noexcept
{
  std::size_t global_delta = padded_bytes;
  if (origin) {
    std::lock_guard lock(origin->arbitration_mutex);
    if (!origin->closed) {
      auto& arena = *origin->memory_reservation;
      auto before = static_cast<std::size_t>(arena.allocated_bytes.load());
      if (before < padded_bytes) { std::terminate(); }
      auto after    = before - padded_bytes;
      auto reserved = static_cast<std::size_t>(arena.size());
      global_delta  = std::max(reserved, before) - std::max(reserved, after);
      arena.allocated_bytes.sub(static_cast<int64_t>(padded_bytes));
    }
    _total_allocated_bytes.sub(global_delta);
    return;
  }
  _total_allocated_bytes.sub(global_delta);
}

void* impl_type::allocate_attempt(std::shared_ptr<stream_ordered_tracker_state> const& origin,
                                  std::size_t bytes,
                                  std::size_t alignment,
                                  ::cuda::stream_ref stream)
{
  constexpr auto quantum = rmm::CUDA_ALLOCATION_ALIGNMENT;
  if (bytes > std::numeric_limits<std::size_t>::max() - (quantum - 1)) {
    throw rmm::out_of_memory("Allocation size cannot be aligned without overflow");
  }
  auto padded = rmm::align_up(bytes, quantum);
  admit(origin, padded, stream);
  void* pointer = nullptr;
  try {
    try {
      pointer = _upstream.allocate(stream, bytes, alignment);
    } catch (std::exception const& e) {
      throw cucascade_out_of_memory(e.what(),
                                    MemoryError::ALLOCATION_FAILED,
                                    bytes,
                                    _total_allocated_bytes.load(),
                                    _pool_handle);
    }
    if (bytes != 0 && !pointer) { throw std::bad_alloc{}; }
    if (pointer) {
      std::lock_guard lock(_ledger_mutex);
      if (!_allocations.emplace(pointer, allocation_record{bytes, alignment, padded, origin})
             .second) {
        std::terminate();
      }
    }
    return pointer;
  } catch (...) {
    if (pointer) { _upstream.deallocate(stream, pointer, bytes, alignment); }
    release_charge(origin, padded);
    throw;
  }
}

void* impl_type::allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
{
  auto origin = allocation_origin(stream);
  try {
    return allocate_attempt(origin, bytes, alignment, stream);
  } catch (...) {
    // The origin stays fixed, while each retry admits its actual byte count and execution stream.
    auto* policy          = origin ? origin->oom_policy.get() : _default_oom_policy.get();
    bool retried          = false;
    void* published       = nullptr;
    auto published_stream = stream;
    try {
      auto* result = policy->handle_oom(
        bytes,
        stream,
        std::current_exception(),
        [this, origin, alignment, &retried, &published, &published_stream](
          std::size_t retry_bytes, ::cuda::stream_ref retry_stream) {
          if (retried) {
            throw rmm::logic_error("OOM policy retried after a successful allocation");
          }
          published        = allocate_attempt(origin, retry_bytes, alignment, retry_stream);
          published_stream = retry_stream;
          retried          = true;
          return published;
        });
      if (!retried || result != published) {
        throw rmm::logic_error("OOM policy must return a pointer from its successful retry");
      }
      return result;
    } catch (...) {
      if (retried) { deallocate(published_stream, published, 0, alignment); }
      throw;
    }
  }
}

void impl_type::deallocate(::cuda::stream_ref stream,
                           void* ptr,
                           std::size_t bytes,
                           std::size_t alignment) noexcept
{
  if (!ptr) {
    _upstream.deallocate(stream, ptr, 0, alignment);
    return;
  }
  allocation_record record;
  bool recorded = false;
  {
    std::lock_guard lock(_ledger_mutex);
    auto node = _allocations.extract(ptr);
    if (!node.empty()) {
      record   = std::move(node.mapped());
      recorded = true;
    }
  }
  if (!recorded) {
    if (bytes != 0) { std::terminate(); }
    _upstream.deallocate(stream, ptr, bytes, alignment);
    return;
  }
  auto origin = record.origin.lock();
  _upstream.deallocate(stream, ptr, record.bytes, record.alignment);
  release_charge(origin, record.padded_bytes);
}

bool impl_type::operator==(impl_type const& other) const noexcept { return this == &other; }

bool impl_type::do_reserve(std::size_t size_bytes, std::size_t limit_bytes)
{
  if (size_bytes > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) { return false; }
  auto [success, post_increase_bytes] = _total_allocated_bytes.try_add(size_bytes, limit_bytes);
  if (success) {
    _peak_total_allocated_bytes.update_peak(post_increase_bytes);
    _total_reserved_bytes.fetch_add(size_bytes);
  }
  return success;
}

std::size_t impl_type::do_reserve_upto(std::size_t size_bytes, std::size_t limit_bytes)
{
  size_bytes = std::min(size_bytes, static_cast<std::size_t>(std::numeric_limits<int64_t>::max()));
  auto post_increase_bytes = _total_allocated_bytes.add_bounded(size_bytes, limit_bytes);
  if (size_bytes > 0) {
    _peak_total_allocated_bytes.update_peak(post_increase_bytes);
    _total_reserved_bytes.fetch_add(size_bytes);
  }
  return size_bytes;
}

void impl_type::do_release_reservation(device_reserved_arena* arena) noexcept
{
  if (!arena) return;

  int64_t allocation_size    = arena->allocated_bytes.load();
  int64_t arena_size         = arena->size();
  std::size_t released_bytes = 0;
  if (arena_size > allocation_size) {
    released_bytes = static_cast<std::size_t>(arena_size - allocation_size);
  }

  _number_of_allocations.fetch_sub(1);
  _total_reserved_bytes.fetch_sub(static_cast<std::size_t>(std::max(int64_t{0}, arena_size)));
  _total_allocated_bytes.sub(released_bytes);
}

void impl_type::set_default_policy(std::unique_ptr<reservation_limit_policy> policy)
{
  _default_reservation_policy = std::move(policy);
}

const reservation_limit_policy& impl_type::get_default_reservation_policy() const
{
  return *_default_reservation_policy;
}

const oom_handling_policy& impl_type::get_default_oom_handling_policy() const
{
  return *_default_oom_policy;
}

reservation_aware_resource_adaptor::reservation_aware_resource_adaptor(
  memory_space_id space_id,
  rmm::device_async_resource_ref upstream,
  std::size_t capacity,
  std::unique_ptr<reservation_limit_policy> stream_reservation_policy,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  AllocationTrackingScope tracking_scope,
  cudaMemPool_t pool_handle)
  : shared_base(cuda::mr::make_shared_resource<impl_type>(space_id,
                                                          std::move(upstream),
                                                          capacity,
                                                          std::move(stream_reservation_policy),
                                                          std::move(default_oom_policy),
                                                          tracking_scope,
                                                          pool_handle))
{
}

reservation_aware_resource_adaptor::reservation_aware_resource_adaptor(
  memory_space_id space_id,
  rmm::device_async_resource_ref upstream,
  std::size_t memory_limit,
  std::size_t capacity,
  std::unique_ptr<reservation_limit_policy> stream_reservation_policy,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  AllocationTrackingScope tracking_scope,
  cudaMemPool_t pool_handle)
  : shared_base(cuda::mr::make_shared_resource<impl_type>(space_id,
                                                          std::move(upstream),
                                                          memory_limit,
                                                          capacity,
                                                          std::move(stream_reservation_policy),
                                                          std::move(default_oom_policy),
                                                          tracking_scope,
                                                          pool_handle))
{
}

rmm::device_async_resource_ref reservation_aware_resource_adaptor::get_upstream_resource()
  const noexcept
{
  return get().get_upstream_resource();
}

std::size_t reservation_aware_resource_adaptor::get_available_memory() const noexcept
{
  return get().get_available_memory();
}

std::size_t reservation_aware_resource_adaptor::get_available_memory(
  ::cuda::stream_ref stream) const noexcept
{
  return get().get_available_memory(stream);
}

std::size_t reservation_aware_resource_adaptor::get_available_memory_print(
  ::cuda::stream_ref stream) const noexcept
{
  return get().get_available_memory_print(stream);
}

std::size_t reservation_aware_resource_adaptor::get_allocated_bytes(::cuda::stream_ref stream) const
{
  return get().get_allocated_bytes(stream);
}

std::size_t reservation_aware_resource_adaptor::get_peak_allocated_bytes(
  ::cuda::stream_ref stream) const
{
  return get().get_peak_allocated_bytes(stream);
}

std::size_t reservation_aware_resource_adaptor::get_total_allocated_bytes() const
{
  return get().get_total_allocated_bytes();
}

std::size_t reservation_aware_resource_adaptor::get_peak_total_allocated_bytes() const
{
  return get().get_peak_total_allocated_bytes();
}

void reservation_aware_resource_adaptor::reset_peak_allocated_bytes(::cuda::stream_ref stream)
{
  get().reset_peak_allocated_bytes(stream);
}

std::size_t reservation_aware_resource_adaptor::get_total_reserved_bytes() const
{
  return get().get_total_reserved_bytes();
}

bool reservation_aware_resource_adaptor::is_stream_tracked(::cuda::stream_ref stream) const
{
  return get().is_stream_tracked(stream);
}

std::unique_ptr<reserved_arena> reservation_aware_resource_adaptor::reserve(
  std::size_t bytes, std::unique_ptr<event_notifier> release_notifer)
{
  return get().reserve(bytes, std::move(release_notifer));
}

std::unique_ptr<reserved_arena> reservation_aware_resource_adaptor::reserve_upto(
  std::size_t bytes, std::unique_ptr<event_notifier> release_notifer)
{
  return get().reserve_upto(bytes, std::move(release_notifer));
}

std::size_t reservation_aware_resource_adaptor::get_active_reservation_count() const noexcept
{
  return get().get_active_reservation_count();
}

bool reservation_aware_resource_adaptor::attach_reservation_to_tracker(
  ::cuda::stream_ref stream,
  std::unique_ptr<reservation> reserved_bytes,
  std::unique_ptr<reservation_limit_policy> stream_reservation_policy,
  std::unique_ptr<oom_handling_policy> stream_oom_policy)
{
  return get().attach_reservation_to_tracker(stream,
                                             std::move(reserved_bytes),
                                             std::move(stream_reservation_policy),
                                             std::move(stream_oom_policy));
}

void reservation_aware_resource_adaptor::reset_stream_reservation(::cuda::stream_ref stream)
{
  get().reset_stream_reservation(stream);
}

void reservation_aware_resource_adaptor::set_default_policy(
  std::unique_ptr<reservation_limit_policy> policy)
{
  get().set_default_policy(std::move(policy));
}

const reservation_limit_policy& reservation_aware_resource_adaptor::get_default_reservation_policy()
  const
{
  return get().get_default_reservation_policy();
}

const oom_handling_policy& reservation_aware_resource_adaptor::get_default_oom_handling_policy()
  const
{
  return get().get_default_oom_handling_policy();
}

}  // namespace memory
}  // namespace cucascade
