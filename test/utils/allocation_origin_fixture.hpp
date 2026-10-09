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

#pragma once

#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <atomic>
#include <cstdlib>
#include <functional>
#include <stdexcept>

namespace cucascade::test {

/** @brief Host-backed asynchronous-resource interface used without invoking CUDA. */
struct host_mock_resource {
  struct state {
    std::atomic<std::size_t> outstanding{0};
    std::atomic<std::size_t> freed_bytes{0};
    std::atomic<int> failures{0};
    std::function<void()> after_allocate;
  };
  std::shared_ptr<state> shared{std::make_shared<state>()};

  void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t)
  {
    auto remaining = shared->failures.load();
    while (remaining > 0) {
      if (shared->failures.compare_exchange_weak(remaining, remaining - 1)) {
        throw std::bad_alloc{};
      }
    }
    auto* pointer = std::malloc(std::max<std::size_t>(bytes, 1));
    if (!pointer) { throw std::bad_alloc{}; }
    shared->outstanding.fetch_add(1);
    if (shared->after_allocate) { shared->after_allocate(); }
    return pointer;
  }

  void deallocate(::cuda::stream_ref, void* pointer, std::size_t bytes, std::size_t) noexcept
  {
    if (pointer) {
      std::free(pointer);
      shared->outstanding.fetch_sub(1);
      shared->freed_bytes.fetch_add(bytes);
    }
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return allocate(::cuda::stream_ref{cudaStream_t{}}, bytes, alignment);
  }
  void deallocate_sync(void* pointer, std::size_t bytes, std::size_t alignment) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{}}, pointer, bytes, alignment);
  }
  bool operator==(host_mock_resource const& other) const noexcept { return shared == other.shared; }
  friend void get_property(host_mock_resource const&, ::cuda::mr::device_accessible) noexcept {}
};

/** @brief A valid reservation owner with no device setup or disk I/O. */
struct inert_disk_backend : idisk_io_backend {
  void write(std::filesystem::path const&,
             void const*,
             std::size_t,
             std::size_t,
             ::cuda::stream_ref) override
  {
    throw std::logic_error("Unexpected I/O in accounting fixture");
  }
  void read(
    std::filesystem::path const&, void*, std::size_t, std::size_t, ::cuda::stream_ref) override
  {
    throw std::logic_error("Unexpected I/O in accounting fixture");
  }
  void write(std::filesystem::path const&, void const*, std::size_t, std::size_t) override
  {
    throw std::logic_error("Unexpected I/O in accounting fixture");
  }
  void read(std::filesystem::path const&, void*, std::size_t, std::size_t) override
  {
    throw std::logic_error("Unexpected I/O in accounting fixture");
  }
};

inline memory::disk_memory_space_config accounting_owner_config()
{
  memory::disk_memory_space_config config;
  config.disk_id         = 0;
  config.memory_capacity = 1ULL << 30;
  config.mount_paths     = "/tmp";
  return config;
}

struct allocation_origin_fixture {
  using resource_type = memory::reservation_aware_resource_adaptor;
  using scope_type    = resource_type::AllocationTrackingScope;
  host_mock_resource upstream;
  // Only reservation::create's stable owner reference is used; the attached arena comes from
  // resource.
  memory::memory_space owner{accounting_owner_config(), std::make_shared<inert_disk_backend>()};
  resource_type resource;
  ::cuda::stream_ref origin{reinterpret_cast<cudaStream_t>(std::uintptr_t{1})};
  ::cuda::stream_ref lane{reinterpret_cast<cudaStream_t>(std::uintptr_t{2})};
  ::cuda::stream_ref other{reinterpret_cast<cudaStream_t>(std::uintptr_t{3})};

  explicit allocation_origin_fixture(scope_type scope, std::size_t capacity = 8192)
    : resource(memory::memory_space_id{memory::Tier::GPU, 0},
               rmm::device_async_resource_ref{upstream},
               capacity,
               nullptr,
               nullptr,
               scope)
  {
  }

  bool attach(::cuda::stream_ref stream,
              std::size_t bytes,
              std::unique_ptr<memory::reservation_limit_policy> policy = nullptr,
              std::unique_ptr<memory::oom_handling_policy> oom         = nullptr,
              std::unique_ptr<memory::event_notifier> notifier         = nullptr)
  {
    auto arena = resource.reserve(bytes, std::move(notifier));
    if (!arena) { return false; }
    return resource.attach_reservation_to_tracker(
      stream,
      memory::reservation::create(owner, std::move(arena)),
      std::move(policy),
      std::move(oom));
  }
};

}  // namespace cucascade::test
