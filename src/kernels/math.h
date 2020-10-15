#pragma once

#include <cuda_runtime.h>

#include "../math.h"

__device__ inline float
Clamp(float x, float a, float b)
{
  return fmaxf(a, fminf(b, x));
}

__device__ inline int
ClampI(int x, int a, int b)
{
  return (x < a) ? a : (b < x) ? b : x;
}

// NOTICE: v must be a normalized vector
__device__ inline float
SphericalTheta(const float3& v)
{
  return acosf(Clamp(v.z, -1.0f, 1.0f));
}

// NOTICE: v must be a normalized vector
__device__ inline float
SphericalPhi(const float3& v)
{
  float p;
  p = atan2f(v.y, v.x);
  if (p < 0.0f) {
    p += 2.0 * kPI;
  }
  return p;
}

__device__ __host__ inline float3
NormalizeF(const float3& a)
{
  return a * rnorm3df(a.x, a.y, a.z);
}

__device__ inline void
BuildCoordSystem(const float3& up, float3& u, float3& v, float3& w)
{
  w = NormalizeF(up);
  v = NormalizeF(Cross(make_float3(0.0034f, 1.0f, 0.0071f), w));
  u = Cross(v, w);
}

__device__ inline float3
MapToLocalCoordSystem(const float3& vec,
                      const float3& u,
                      const float3& v,
                      const float3& w)
{
  float3 local = make_float3(Dot(vec, u), Dot(vec, v), Dot(vec, w));

  return NormalizeF(local);
}

__device__ inline bool
HasInf(const float3& a)
{
  return isinf(a.x) || isinf(a.y) || isinf(a.z);
}

__device__ inline bool
HasNaN(const float3& a)
{
  return isnan(a.x) || isnan(a.y) || isnan(a.z);
}

__device__ inline float3
OffsetRay(const float3& point, const float3& geometricNormal)
{
  constexpr float origin = 1.0f / 32.0f;
  constexpr float floatScale = 1.0f / 65536.0f;
  constexpr float intScale = 256.0f;

  int3 ofI = make_int3(intScale * geometricNormal.x,
                       intScale * geometricNormal.y,
                       intScale * geometricNormal.z);

  float3 pI = make_float3(__int_as_float(__float_as_int(point.x) +
                                         ((point.x < 0.0f) ? -ofI.x : ofI.x)),
                          __int_as_float(__float_as_int(point.y) +
                                         ((point.y < 0.0f) ? -ofI.y : ofI.y)),
                          __int_as_float(__float_as_int(point.z) +
                                         ((point.z < 0.0f) ? -ofI.z : ofI.z)));

  return make_float3(
    fabsf(point.x) < origin ? point.x + floatScale * geometricNormal.x : pI.x,
    fabsf(point.y) < origin ? point.y + floatScale * geometricNormal.y : pI.y,
    fabsf(point.z) < origin ? point.z + floatScale * geometricNormal.z : pI.z);
}
