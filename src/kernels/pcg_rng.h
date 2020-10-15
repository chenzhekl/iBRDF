#pragma once

#include <cstdint>

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
class PCG32
{
public:
  __device__ std::uint32_t Random();
  __device__ float RandomF();
  __device__ void Seed(std::uint64_t initState, std::uint64_t initSeq);

private:
  std::uint64_t mState = 0x853c49e6748fea9bULL;
  std::uint64_t mInc = 0xda3e39cb94b95bdbULL;
};

__device__ void
PCG32::Seed(std::uint64_t initState, std::uint64_t initSeq)
{
  mState = 0U;
  mInc = (initSeq << 1u) | 1u;
  Random();
  mState += initState;
  Random();
}

__device__ std::uint32_t
PCG32::Random()
{
  std::uint64_t oldstate = mState;
  // Advance internal state
  mState = oldstate * 6364136223846793005ULL + (mInc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  std::uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  std::uint32_t rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ float
PCG32::RandomF()
{
  return ldexpf(Random(), -32);
}

__device__ float3
CosineSampleHemisphere(const float2& sample, float& pdf)
{
  float cosPhi = cosf(2.0f * kPI * sample.x);
  float sinPhi = sinf(2.0f * kPI * sample.x);
  float cosTheta = sqrtf(1.0f - sample.y);
  float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
  float pu = sinTheta * cosPhi;
  float pv = sinTheta * sinPhi;
  float pw = cosTheta;

  pdf = cosTheta * kInvPI;

  return make_float3(pu, pv, pw);
}
