/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace cucascade {
namespace cuda {

/**
 * @brief Cached driver version as reported by cudaDriverGetVersion (e.g. 12070 for 12.7).
 *
 * Returns 0 when the query itself fails. A CUDA 12.x toolkit newer than the installed driver
 * runs under minor-version compatibility: the runtime loads, but every entry point introduced
 * after the driver's release returns cudaErrorCallRequiresNewerDriver. Callers use this to pick
 * the older code path at runtime instead of relying on CUDART_VERSION alone.
 */
inline int driver_version() noexcept
{
  static int const version = [] {
    int value = 0;
    if (::cudaDriverGetVersion(&value) != cudaSuccess) {
      (void)::cudaGetLastError();
      return 0;
    }
    return value;
  }();
  return version;
}

/// True when the driver implements the CUDA 12.8 runtime additions (cudaMemcpyBatchAsync,
/// cudaStreamGetDevice, ...).
inline bool driver_supports_cuda_12_8() noexcept { return driver_version() >= 12080; }

/// cudaMemcpyBatchAsync is usable only when both the toolkit and the driver are >= 12.8.
inline bool supports_batched_memcpy() noexcept
{
#if CUDART_VERSION >= 12080
  return driver_supports_cuda_12_8();
#else
  return false;
#endif
}

/**
 * @brief cudaStreamGetDevice with a fallback for drivers older than 12.8.
 *
 * Default-stream handles resolve to the calling thread's current device, matching
 * cudaStreamGetDevice. For other streams on an old driver, the stream's context (its device's
 * primary context for runtime-created streams) is pushed via driver entry points obtained with
 * cudaGetDriverEntryPoint, the runtime's current device is read, and the context is popped.
 */
inline cudaError_t stream_get_device(cudaStream_t stream, int* device) noexcept
{
#if CUDART_VERSION >= 12080
  if (driver_supports_cuda_12_8()) { return ::cudaStreamGetDevice(stream, device); }
#endif
  if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
    return ::cudaGetDevice(device);
  }
#if CUDART_VERSION < 13000
  using stream_get_ctx_fn = CUresult (*)(CUstream, CUcontext*);
  using ctx_push_fn       = CUresult (*)(CUcontext);
  using ctx_pop_fn        = CUresult (*)(CUcontext*);
  struct entry_points {
    stream_get_ctx_fn stream_get_ctx = nullptr;
    ctx_push_fn ctx_push             = nullptr;
    ctx_pop_fn ctx_pop               = nullptr;
    bool ok                          = false;
  };
  static entry_points const eps = [] {
    entry_points result{};
    // Request the CUDA 12.0 ABI explicitly: the default query returns the newest symbol version
    // the driver knows, and cuStreamGetCtx_v2 (12.5+) takes a third CUgreenCtx* argument.
    auto resolve = [](char const* symbol, void** target) {
      cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
      auto const err =
        ::cudaGetDriverEntryPointByVersion(symbol, target, 12000, cudaEnableDefault, &status);
#else
      auto const err = ::cudaGetDriverEntryPoint(symbol, target, cudaEnableDefault, &status);
#endif
      if (err != cudaSuccess) { (void)::cudaGetLastError(); }
      return err == cudaSuccess && status == cudaDriverEntryPointSuccess && *target != nullptr;
    };
    result.ok = resolve("cuStreamGetCtx", reinterpret_cast<void**>(&result.stream_get_ctx)) &&
                resolve("cuCtxPushCurrent", reinterpret_cast<void**>(&result.ctx_push)) &&
                resolve("cuCtxPopCurrent", reinterpret_cast<void**>(&result.ctx_pop));
    return result;
  }();
  if (!eps.ok) { return cudaErrorCallRequiresNewerDriver; }
  CUcontext context = nullptr;
  if (eps.stream_get_ctx(static_cast<CUstream>(stream), &context) != CUDA_SUCCESS) {
    return cudaErrorInvalidResourceHandle;
  }
  if (eps.ctx_push(context) != CUDA_SUCCESS) { return cudaErrorInvalidValue; }
  auto const status = ::cudaGetDevice(device);
  CUcontext popped  = nullptr;
  (void)eps.ctx_pop(&popped);
  return status;
#else
  return cudaErrorCallRequiresNewerDriver;
#endif
}

}  // namespace cuda
}  // namespace cucascade
