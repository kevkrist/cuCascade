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

#include "utils/allocation_origin_fixture.hpp"

#include <cucascade/memory/error.hpp>

#include <catch2/catch_all.hpp>

#include <barrier>
#include <latch>
#include <limits>
#include <thread>
#include <vector>

using namespace cucascade::memory;
using fixture = cucascade::test::allocation_origin_fixture;
using scope   = fixture::scope_type;

namespace {
struct changed_retry : oom_handling_policy {
  ::cuda::stream_ref retry_stream;
  std::size_t retry_bytes;
  explicit changed_retry(::cuda::stream_ref stream, std::size_t bytes = 513)
    : retry_stream(stream), retry_bytes(bytes)
  {
  }
  std::string get_policy_name() const noexcept override { return "changed retry"; }
  void* do_handle_oom(std::size_t, ::cuda::stream_ref, std::exception_ptr, RetryFunc retry) override
  {
    return retry(retry_bytes, retry_stream);
  }
};

struct arena_observer_policy : reservation_limit_policy {
  reserved_arena** observed;
  explicit arena_observer_policy(reserved_arena*& pointer) : observed(&pointer) {}
  void handle_over_reservation(::cuda::stream_ref,
                               std::size_t,
                               std::size_t,
                               reserved_arena* arena) override
  {
    *observed = arena;
  }
  std::string get_policy_name() const override { return "observe"; }
};
}  // namespace

TEST_CASE("Allocation context admits against origin and preserves free ownership",
          "[allocation_origin][cpu]")
{
  auto mode = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  fixture f{mode};
  REQUIRE(f.attach(f.origin, 1024, std::make_unique<fail_reservation_limit_policy>()));
  if (mode == scope::PER_STREAM) { REQUIRE(f.attach(f.lane, 2048)); }
  void* pointer;
  {
    auto context = f.resource.allocation_context_for(f.origin);
    pointer      = f.resource.allocate(f.lane, 512, 256);
    CHECK(f.resource.get_allocated_bytes(f.origin) == 512);
    if (mode == scope::PER_STREAM) { CHECK(f.resource.get_allocated_bytes(f.lane) == 0); }
    CHECK_THROWS_AS(f.resource.allocate(f.lane, 768, 256), rmm::out_of_memory);
    CHECK(f.resource.get_allocated_bytes(f.origin) == 512);
  }
  std::jthread free_thread([&] {
    if (mode == scope::PER_THREAD) { f.attach(f.other, 2048); }
    f.resource.deallocate(f.other, pointer, 512, 256);
    if (mode == scope::PER_THREAD) { f.resource.reset_stream_reservation(f.other); }
  });
  free_thread.join();
  CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
  f.resource.reset_stream_reservation(f.origin);
  f.resource.reset_stream_reservation(f.lane);
  CHECK(f.resource.get_total_allocated_bytes() == 0);
  CHECK(f.upstream.shared->outstanding == 0);
}

TEST_CASE("Unreserved allocations cannot debit a reservation attached later",
          "[allocation_origin][cpu]")
{
  auto mode = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  fixture f{mode};
  void* pointer;
  {
    auto null_context = f.resource.allocation_context_for(f.origin);
    REQUIRE(f.attach(f.lane, 1024));
    pointer = f.resource.allocate(f.lane, 1, 256);
    CHECK(f.resource.get_allocated_bytes(f.lane) == 0);
    CHECK(f.resource.get_total_allocated_bytes() == 1280);
  }
  f.resource.deallocate(f.lane, pointer, 1, 256);
  CHECK(f.resource.get_allocated_bytes(f.lane) == 0);
  CHECK(f.resource.get_total_allocated_bytes() == 1024);
  f.resource.reset_stream_reservation(f.lane);
  CHECK(f.resource.get_total_allocated_bytes() == 0);
}

TEST_CASE("Reset releases slack immediately and closes captured origins",
          "[allocation_origin][cpu]")
{
  auto mode  = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  auto bytes = GENERATE(std::size_t{512}, std::size_t{1024}, std::size_t{1536});
  fixture f{mode};
  auto channel = std::make_shared<notification_channel>();
  REQUIRE(f.attach(f.origin, 1024, nullptr, nullptr, channel->get_notifier()));
  auto context  = f.resource.allocation_context_for(f.origin);
  auto* pointer = f.resource.allocate(f.lane, bytes, 256);
  CHECK(f.resource.get_total_allocated_bytes() == std::max(bytes, std::size_t{1024}));
  f.resource.reset_stream_reservation(f.origin);
  CHECK(f.resource.get_total_reserved_bytes() == 0);
  CHECK(f.resource.get_active_reservation_count() == 0);
  CHECK(f.resource.get_total_allocated_bytes() == bytes);
  CHECK(channel->wait() == notification_channel::wait_status::NOTIFIED);
  CHECK_THROWS_AS(f.resource.allocate(f.lane, 256, 256), rmm::logic_error);
  f.resource.reset_stream_reservation(f.origin);
  REQUIRE(f.attach(f.origin, 2048));
  std::jthread free_thread([&] { f.resource.deallocate(f.origin, pointer, bytes, 256); });
  free_thread.join();
  CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
  CHECK(f.resource.get_total_allocated_bytes() == 2048);
  f.resource.reset_stream_reservation(f.origin);
  CHECK(f.resource.get_total_allocated_bytes() == 0);
}

TEST_CASE("Nested contexts and copies isolate adaptor identity", "[allocation_origin][cpu]")
{
  fixture f{scope::PER_STREAM};
  REQUIRE(f.attach(f.origin, 1024));
  REQUIRE(f.attach(f.other, 2048));
  fixture::resource_type different{
    memory_space_id{Tier::GPU, 0}, rmm::device_async_resource_ref{f.upstream}, 8192};
  auto copy = f.resource;
  CHECK(copy.get() == f.resource.get());
  CHECK_FALSE(different.get() == f.resource.get());
  auto outer = f.resource.allocation_context_for(f.origin);
  {
    auto inner    = f.resource.allocation_context_for(f.other);
    auto moved    = std::move(inner);
    auto* pointer = copy.allocate(f.lane, 256, 256);
    CHECK(f.resource.get_allocated_bytes(f.other) == 256);
    auto* unrelated = different.allocate(f.lane, 256, 256);
    CHECK(different.get_total_allocated_bytes() == 256);
    different.deallocate(f.origin, unrelated, 256, 256);
    copy.deallocate(f.origin, pointer, 256, 256);
  }
  auto* pointer = f.resource.allocate(f.lane, 512, 256);
  CHECK(f.resource.get_allocated_bytes(f.origin) == 512);
  CHECK(f.resource.get_allocated_bytes(f.other) == 0);
  f.resource.deallocate(f.other, pointer, 512, 256);
}

TEST_CASE("Overflow growth and shrink preserve committed capacity", "[allocation_origin][cpu]")
{
  auto mode = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  fixture f{mode, 4096};
  reserved_arena* arena = nullptr;
  REQUIRE(f.attach(f.origin, 512, std::make_unique<arena_observer_policy>(arena)));
  auto context  = f.resource.allocation_context_for(f.origin);
  auto* pointer = f.resource.allocate(f.lane, 1024, 256);
  REQUIRE(arena != nullptr);
  CHECK(f.resource.get_total_allocated_bytes() == 1024);
  REQUIRE(arena->grow_by(256));
  CHECK(f.resource.get_total_allocated_bytes() == 1024);
  CHECK(f.resource.get_total_reserved_bytes() == 768);
  REQUIRE(arena->grow_by(512));
  CHECK(f.resource.get_total_allocated_bytes() == 1280);
  CHECK(f.resource.get_total_reserved_bytes() == 1280);
  arena->shrink_to_fit();
  CHECK(f.resource.get_total_allocated_bytes() == 1024);
  CHECK(f.resource.get_total_reserved_bytes() == 1024);
  f.resource.deallocate(f.other, pointer, 1024, 256);
  arena->shrink_to_fit();
  CHECK(f.resource.get_total_allocated_bytes() == 0);
  CHECK(f.resource.get_total_reserved_bytes() == 0);
  CHECK_FALSE(arena->grow_by(std::numeric_limits<std::size_t>::max()));
  f.resource.reset_stream_reservation(f.origin);
  CHECK(f.resource.get_active_reservation_count() == 0);
}

TEST_CASE("Increase policy does not double-charge successful growth", "[allocation_origin][cpu]")
{
  auto mode = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  fixture f{mode};
  REQUIRE(f.attach(f.origin, 512, std::make_unique<increase_reservation_limit_policy>(1.0)));
  auto context  = f.resource.allocation_context_for(f.origin);
  auto* pointer = f.resource.allocate(f.lane, 1024, 256);
  CHECK(f.resource.get_allocated_bytes(f.origin) == 1024);
  CHECK(f.resource.get_total_allocated_bytes() == 1024);
  CHECK(f.resource.get_total_reserved_bytes() == 1024);
  f.resource.deallocate(f.other, pointer, 1024, 256);
  f.resource.reset_stream_reservation(f.origin);
  CHECK(f.resource.get_total_allocated_bytes() == 0);
}

TEST_CASE("Failed attempts roll back and retry retains original ownership",
          "[allocation_origin][cpu]")
{
  auto mode = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  fixture f{mode};
  REQUIRE(f.attach(f.origin, 1024));
  auto context                = f.resource.allocation_context_for(f.origin);
  f.upstream.shared->failures = 1;
  CHECK_THROWS_AS(f.resource.allocate(f.lane, 512, 256), cucascade_out_of_memory);
  CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
  CHECK(f.resource.get_total_allocated_bytes() == 1024);
  CHECK_THROWS_AS(f.resource.allocate(f.lane, 16384, 256), cucascade_out_of_memory);
  CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
  CHECK(f.resource.get_total_allocated_bytes() == 1024);
  CHECK_THROWS_AS(f.resource.allocate(f.lane, std::numeric_limits<std::size_t>::max(), 256),
                  rmm::out_of_memory);
  f.resource.reset_stream_reservation(f.origin);
  REQUIRE(f.attach(f.origin, 1024, nullptr, std::make_unique<changed_retry>(f.other)));
  {
    auto fresh                  = f.resource.allocation_context_for(f.origin);
    f.upstream.shared->failures = 1;
    auto* pointer               = f.resource.allocate(f.lane, 256, 256);
    CHECK(f.resource.get_allocated_bytes(f.origin) == 768);
    f.resource.deallocate(f.lane, pointer, 256, 256);
    CHECK(f.upstream.shared->freed_bytes == 513);
    CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
    f.upstream.shared->failures = 2;
    CHECK_THROWS_AS(f.resource.allocate(f.lane, 256, 256), cucascade_out_of_memory);
    CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
  }
  f.resource.reset_stream_reservation(f.origin);
  CHECK(f.resource.get_total_allocated_bytes() == 0);
}

TEST_CASE("Reset between admission and publication leaves only live bytes",
          "[allocation_origin][cpu]")
{
  auto mode = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  fixture f{mode};
  REQUIRE(f.attach(f.origin, 1024));
  auto context                      = f.resource.allocation_context_for(f.origin);
  f.upstream.shared->after_allocate = [&] { f.resource.reset_stream_reservation(f.origin); };
  auto* pointer                     = f.resource.allocate(f.lane, 512, 256);
  CHECK(f.resource.get_total_allocated_bytes() == 512);
  CHECK(f.resource.get_total_reserved_bytes() == 0);
  CHECK(f.resource.get_active_reservation_count() == 0);
  f.resource.deallocate(f.origin, pointer, 512, 256);
  CHECK(f.resource.get_total_allocated_bytes() == 0);
}

TEST_CASE("Concurrent origin admission and reset cannot charge a replacement",
          "[allocation_origin][cpu][threading]")
{
  fixture f{scope::PER_STREAM, 4096};
  for (int round = 0; round < 50; ++round) {
    REQUIRE(f.attach(f.origin, 1024, std::make_unique<fail_reservation_limit_policy>()));
    std::barrier start{9};
    std::latch attempted{8};
    std::latch release{1};
    std::atomic<std::size_t> admitted{0};
    std::vector<std::jthread> threads;
    for (int worker = 0; worker < 8; ++worker) {
      threads.emplace_back([&] {
        auto context = f.resource.allocation_context_for(f.origin);
        start.arrive_and_wait();
        void* pointer = nullptr;
        try {
          pointer = f.resource.allocate(f.lane, 256, 256);
          admitted.fetch_add(256);
        } catch (rmm::out_of_memory const&) {
        }
        attempted.count_down();
        release.wait();
        if (pointer) { f.resource.deallocate(f.other, pointer, 256, 256); }
      });
    }
    start.arrive_and_wait();
    attempted.wait();
    CHECK(admitted <= 1024);
    CHECK(f.resource.get_allocated_bytes(f.origin) == admitted);
    f.resource.reset_stream_reservation(f.origin);
    auto replaced = f.attach(f.origin, 1024);
    release.count_down();
    threads.clear();
    REQUIRE(replaced);
    CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
    CHECK(f.resource.get_total_allocated_bytes() == 1024);
    f.resource.reset_stream_reservation(f.origin);
    CHECK(f.resource.get_total_allocated_bytes() == 0);
  }
}

TEST_CASE("Zero-sized requests do not create logical charges", "[allocation_origin][cpu]")
{
  auto mode = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  fixture f{mode};
  REQUIRE(f.attach(f.origin, 1024));
  auto context  = f.resource.allocation_context_for(f.origin);
  auto* pointer = f.resource.allocate(f.lane, 0, 256);
  CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
  f.resource.deallocate(f.other, pointer, 0, 256);
  CHECK(f.upstream.shared->outstanding == 0);
  CHECK(f.resource.get_total_allocated_bytes() == 1024);
}

namespace {
struct invalid_retry_result : oom_handling_policy {
  bool throw_after_success;
  explicit invalid_retry_result(bool fail) : throw_after_success(fail) {}
  std::string get_policy_name() const noexcept override { return "invalid retry"; }
  void* do_handle_oom(std::size_t,
                      ::cuda::stream_ref stream,
                      std::exception_ptr,
                      RetryFunc retry) override
  {
    retry(512, stream);
    if (throw_after_success) { throw std::runtime_error("after successful retry"); }
    return nullptr;
  }
};
}  // namespace

TEST_CASE("Retry from zero is freed by its actual allocation size", "[allocation_origin][cpu]")
{
  auto mode = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  fixture f{mode};
  REQUIRE(f.attach(f.origin, 1024, nullptr, std::make_unique<changed_retry>(f.other)));
  auto context                = f.resource.allocation_context_for(f.origin);
  f.upstream.shared->failures = 1;
  auto* pointer               = f.resource.allocate(f.lane, 0, 256);
  CHECK(f.resource.get_allocated_bytes(f.origin) == 768);
  f.resource.deallocate(f.origin, pointer, 0, 256);
  CHECK(f.upstream.shared->freed_bytes == 513);
  CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
}

TEST_CASE("Invalid OOM policy results cannot strand a successful retry", "[allocation_origin][cpu]")
{
  auto mode = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  auto fail = GENERATE(false, true);
  fixture f{mode};
  REQUIRE(f.attach(f.origin, 1024, nullptr, std::make_unique<invalid_retry_result>(fail)));
  auto context                = f.resource.allocation_context_for(f.origin);
  f.upstream.shared->failures = 1;
  CHECK_THROWS(f.resource.allocate(f.lane, 256, 256));
  CHECK(f.upstream.shared->outstanding == 0);
  CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
  CHECK(f.resource.get_total_allocated_bytes() == 1024);
}

TEST_CASE("Reservation attachment rejects foreign adaptor arenas", "[allocation_origin][cpu]")
{
  fixture f{scope::PER_THREAD};
  fixture other{scope::PER_THREAD};
  auto reservation = reservation::create(f.owner, f.resource.reserve(1024));
  CHECK_FALSE(other.resource.attach_reservation_to_tracker(other.origin, std::move(reservation)));
  CHECK(f.resource.get_total_allocated_bytes() == 0);
  CHECK(other.resource.get_total_allocated_bytes() == 0);
  CHECK_FALSE(other.resource.attach_reservation_to_tracker(other.origin, nullptr));
}

TEST_CASE("Thread exit releases unused reservation while outputs retain global ownership",
          "[allocation_origin][cpu][threading]")
{
  fixture f{scope::PER_THREAD};
  void* pointer = nullptr;
  std::jthread worker([&] {
    if (!f.attach(f.origin, 1024)) { return; }
    auto context = f.resource.allocation_context_for(f.origin);
    pointer      = f.resource.allocate(f.lane, 512, 256);
  });
  worker.join();
  REQUIRE(pointer != nullptr);
  CHECK(f.resource.get_active_reservation_count() == 0);
  CHECK(f.resource.get_total_reserved_bytes() == 0);
  CHECK(f.resource.get_total_allocated_bytes() == 512);
  REQUIRE(f.attach(f.origin, 2048));
  f.resource.deallocate(f.origin, pointer, 512, 256);
  CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
  CHECK(f.resource.get_total_allocated_bytes() == 2048);
}

TEST_CASE("Adaptor cleanup closes arenas retained in other threads' TLS",
          "[allocation_origin][cpu][threading]")
{
  fixture f{scope::PER_THREAD};
  auto resource =
    std::make_unique<fixture::resource_type>(memory_space_id{Tier::GPU, 0},
                                             rmm::device_async_resource_ref{f.upstream},
                                             8192,
                                             nullptr,
                                             nullptr,
                                             scope::PER_THREAD);
  auto channel = std::make_shared<notification_channel>();
  std::latch attached{1};
  std::latch destroyed{1};
  bool success = false;
  std::jthread worker([&] {
    auto arena = resource->reserve(1024, channel->get_notifier());
    success    = resource->attach_reservation_to_tracker(
      f.origin, reservation::create(f.owner, std::move(arena)));
    attached.count_down();
    destroyed.wait();
  });
  attached.wait();
  resource.reset();
  CHECK(channel->wait() == notification_channel::wait_status::NOTIFIED);
  destroyed.count_down();
  worker.join();
  REQUIRE(success);
}

TEST_CASE("Reset and origin frees commute under concurrent execution",
          "[allocation_origin][cpu][threading]")
{
  fixture f{scope::PER_STREAM};
  for (int round = 0; round < 200; ++round) {
    REQUIRE(f.attach(f.origin, 1024));
    auto* pointer = f.resource.allocate(f.origin, 1536, 256);
    std::barrier start{2};
    std::jthread worker([&] {
      start.arrive_and_wait();
      f.resource.deallocate(f.other, pointer, 1536, 256);
    });
    start.arrive_and_wait();
    f.resource.reset_stream_reservation(f.origin);
    worker.join();
    CHECK(f.resource.get_total_allocated_bytes() == 0);
    CHECK(f.resource.get_total_reserved_bytes() == 0);
    CHECK(f.resource.get_active_reservation_count() == 0);
  }
}

TEST_CASE("Retry to zero retains upstream free metadata", "[allocation_origin][cpu]")
{
  auto mode = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  fixture f{mode};
  REQUIRE(f.attach(f.origin, 1024, nullptr, std::make_unique<changed_retry>(f.other, 0)));
  auto context                = f.resource.allocation_context_for(f.origin);
  f.upstream.shared->failures = 1;
  auto* pointer               = f.resource.allocate(f.lane, 512, 256);
  CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
  f.resource.deallocate(f.origin, pointer, 512, 256);
  CHECK(f.upstream.shared->freed_bytes == 0);
  CHECK(f.upstream.shared->outstanding == 0);
}

TEST_CASE("Adaptor destruction may finish while detached notifier destruction is blocked",
          "[allocation_origin][cpu][threading]")
{
  fixture f{scope::PER_THREAD};
  auto resource =
    std::make_unique<fixture::resource_type>(memory_space_id{Tier::GPU, 0},
                                             rmm::device_async_resource_ref{f.upstream},
                                             8192,
                                             nullptr,
                                             nullptr,
                                             scope::PER_THREAD);
  std::latch attached{1};
  std::latch permit_exit{1};
  std::latch notifier_destroying{1};
  std::latch permit_notifier_destroy{1};
  auto channel = std::shared_ptr<notification_channel>(new notification_channel,
                                                       [&](notification_channel* pointer) {
                                                         notifier_destroying.count_down();
                                                         permit_notifier_destroy.wait();
                                                         delete pointer;
                                                       });
  bool success = false;
  std::jthread worker([&] {
    auto arena = resource->reserve(1024, channel->get_notifier());
    success    = resource->attach_reservation_to_tracker(
      f.origin, reservation::create(f.owner, std::move(arena)));
    attached.count_down();
    permit_exit.wait();
  });
  attached.wait();
  channel.reset();
  permit_exit.count_down();
  notifier_destroying.wait();
  // TLS close already released counters and disabled the arena callback before this notifier phase.
  resource.reset();
  permit_notifier_destroy.count_down();
  worker.join();
  REQUIRE(success);
}

TEST_CASE("Per-thread contexts restore explicit unreserved origins and isolate adaptor instances",
          "[allocation_origin][cpu]")
{
  fixture first{scope::PER_THREAD};
  fixture second{scope::PER_THREAD};
  auto unreserved = first.resource.allocation_context_for(first.origin);
  REQUIRE(first.attach(first.origin, 1024));
  REQUIRE(second.attach(second.origin, 2048));
  {
    auto reserved = first.resource.allocation_context_for(first.origin);
    auto* one     = first.resource.allocate(first.lane, 512, 256);
    auto* two     = second.resource.allocate(second.lane, 256, 256);
    CHECK(first.resource.get_allocated_bytes(first.origin) == 512);
    CHECK(second.resource.get_allocated_bytes(second.origin) == 256);
    first.resource.deallocate(first.other, one, 512, 256);
    second.resource.deallocate(second.other, two, 256, 256);
  }
  auto* pointer = first.resource.allocate(first.lane, 512, 256);
  CHECK(first.resource.get_allocated_bytes(first.origin) == 0);
  CHECK(first.resource.get_total_allocated_bytes() == 1536);
  first.resource.deallocate(first.origin, pointer, 512, 256);
  CHECK(first.resource.get_total_allocated_bytes() == 1024);
}

TEST_CASE("Failed upstream allocation retains growth but rolls back all live charge",
          "[allocation_origin][cpu]")
{
  auto mode = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  fixture f{mode};
  REQUIRE(f.attach(f.origin, 512, std::make_unique<increase_reservation_limit_policy>(1.0)));
  auto context                = f.resource.allocation_context_for(f.origin);
  f.upstream.shared->failures = 1;
  CHECK_THROWS_AS(f.resource.allocate(f.lane, 1024, 256), cucascade_out_of_memory);
  CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
  CHECK(f.resource.get_total_reserved_bytes() == 1024);
  CHECK(f.resource.get_total_allocated_bytes() == 1024);
  f.resource.reset_stream_reservation(f.origin);
  CHECK(f.resource.get_total_allocated_bytes() == 0);
}

TEST_CASE("Thread-local resource destruction does not revisit destroyed tracker TLS",
          "[allocation_origin][cpu][threading]")
{
  fixture f{scope::PER_THREAD};
  auto channel  = std::make_shared<notification_channel>();
  bool attached = false;
  std::jthread worker([&] {
    // This resource is initialized before its first use creates the nontrivial tracker TLS map.
    thread_local fixture::resource_type resource{memory_space_id{Tier::GPU, 0},
                                                 rmm::device_async_resource_ref{f.upstream},
                                                 8192,
                                                 nullptr,
                                                 nullptr,
                                                 scope::PER_THREAD};
    auto arena = resource.reserve(1024, channel->get_notifier());
    attached   = resource.attach_reservation_to_tracker(
      f.origin, reservation::create(f.owner, std::move(arena)));
    auto context  = resource.allocation_context_for(f.origin);
    auto* pointer = resource.allocate(f.lane, 512, 256);
    resource.deallocate(f.other, pointer, 512, 256);
  });
  worker.join();
  REQUIRE(attached);
  CHECK(channel->wait() == notification_channel::wait_status::NOTIFIED);
  CHECK(f.upstream.shared->outstanding == 0);
}

TEST_CASE("Concurrent unreserved admission respects global capacity in both scopes",
          "[allocation_origin][cpu][threading]")
{
  auto mode = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  fixture f{mode, 1024};
  for (int round = 0; round < 20; ++round) {
    std::barrier start{9};
    std::latch attempted{8};
    std::latch release{1};
    std::atomic<std::size_t> admitted{0};
    std::vector<std::jthread> threads;
    for (int worker = 0; worker < 8; ++worker) {
      threads.emplace_back([&] {
        auto context = f.resource.allocation_context_for(f.origin);
        start.arrive_and_wait();
        void* pointer = nullptr;
        try {
          pointer = f.resource.allocate(f.lane, 256, 256);
          admitted.fetch_add(256);
        } catch (rmm::out_of_memory const&) {
        }
        attempted.count_down();
        release.wait();
        if (pointer) { f.resource.deallocate(f.other, pointer, 256, 256); }
      });
    }
    start.arrive_and_wait();
    attempted.wait();
    CHECK(admitted == 1024);
    CHECK(f.resource.get_total_allocated_bytes() == 1024);
    release.count_down();
    threads.clear();
    CHECK(f.resource.get_total_allocated_bytes() == 0);
    CHECK(f.upstream.shared->outstanding == 0);
  }
}
