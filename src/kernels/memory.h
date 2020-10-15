#pragma once

#include "pcg_rng.h"

struct PRD
{};

static __forceinline__ __device__ void*
UnpackPointer(std::uint32_t i0, std::uint32_t i1)
{
  const std::uint64_t uptr = static_cast<std::uint64_t>(i0) << 32 | i1;
  void* ptr = reinterpret_cast<void*>(uptr);
  return ptr;
}

static __forceinline__ __device__ void
PackPointer(void* ptr, std::uint32_t& i0, std::uint32_t& i1)
{
  const std::uint64_t uptr = reinterpret_cast<std::uint64_t>(ptr);
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T*
GetPRD()
{
  const std::uint32_t u0 = optixGetPayload_0();
  const std::uint32_t u1 = optixGetPayload_1();
  return reinterpret_cast<T*>(UnpackPointer(u0, u1));
}
