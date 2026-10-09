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

#include <catch2/catch_all.hpp>

#include <new>
#include <utility>

namespace {
thread_local bool fail_next_host_allocation = false;
}

// This separate test executable owns the replacement allocator. Production has no fault-injection
// knob.
void* operator new(std::size_t bytes)
{
  if (std::exchange(fail_next_host_allocation, false)) { throw std::bad_alloc{}; }
  if (auto* pointer = std::malloc(std::max<std::size_t>(bytes, 1))) { return pointer; }
  throw std::bad_alloc{};
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

TEST_CASE("Ledger publication failure frees upstream and rolls back admission",
          "[allocation_origin][cpu][metadata]")
{
  using fixture = cucascade::test::allocation_origin_fixture;
  using scope   = fixture::scope_type;
  auto mode     = GENERATE(scope::PER_THREAD, scope::PER_STREAM);
  auto detach   = GENERATE(false, true);
  fixture f{mode};
  REQUIRE(f.attach(f.origin, 1024));
  auto context                      = f.resource.allocation_context_for(f.origin);
  f.upstream.shared->after_allocate = [&] {
    if (detach) { f.resource.reset_stream_reservation(f.origin); }
    // The next host allocation is the ledger node or bucket publication.
    fail_next_host_allocation = true;
  };
  CHECK_THROWS_AS(f.resource.allocate(f.lane, 513, 256), std::bad_alloc);
  CHECK_FALSE(fail_next_host_allocation);
  CHECK(f.upstream.shared->outstanding == 0);
  CHECK(f.upstream.shared->freed_bytes == 513);
  CHECK(f.resource.get_allocated_bytes(f.origin) == 0);
  CHECK(f.resource.get_total_allocated_bytes() == (detach ? 0 : 1024));
  CHECK(f.resource.get_total_reserved_bytes() == (detach ? 0 : 1024));
  f.resource.reset_stream_reservation(f.origin);
  CHECK(f.resource.get_total_allocated_bytes() == 0);
}
