#pragma once

#include <cmath>

#include <cuda_runtime.h>

constexpr float kPI = 3.14159265359f;
constexpr float kInvPI = 1.0f / kPI;
constexpr unsigned int kMaxUInt32 = 4294967295;

__device__ __host__ inline float3
operator+(const float3& a, const float3& b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline float3
operator+(const float3& a, float b)
{
  return make_float3(a.x + b, a.y + b, a.z + b);
}

__device__ __host__ inline float3
operator+(float a, const float3& b)
{
  return b + a;
}

__device__ __host__ inline float3
operator-(const float3& a, const float3& b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline float3
operator-(const float3& a)
{
  return make_float3(-a.x, -a.y, -a.z);
}

__device__ __host__ inline float3
operator/(const float3& a, const float3& b)
{
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __host__ inline float2
operator/(const float2& a, const float2& b)
{
  return make_float2(a.x / b.x, a.y / b.y);
}

__device__ __host__ inline float3
operator/(const float3& a, float b)
{
  return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ __host__ inline float3 operator*(const float3& a, const float3& b)
{
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __host__ inline float3 operator*(const float3& a, float b)
{
  return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ __host__ inline float3 operator*(float a, const float3& b)
{
  return b * a;
}

__device__ __host__ inline float3
Cross(const float3& a, const float3& b)
{
  return make_float3(
    a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ __host__ inline float
Dot(const float3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ inline float3
Normalize(const float3& a)
{
  return a / std::sqrt(Dot(a, a));
}

__device__ __host__ inline float
NormDot(const float3& a, const float3& b)
{
  return Dot(Normalize(a), Normalize(b));
}

__device__ __host__ inline float3
Reflect(const float3& wo, const float3& n)
{
  return 2.0f * n * Dot(n, wo) - wo;
}

__device__ __host__ inline bool
AllZero(const float3& a)
{
  return a.x == 0.0f && a.y == 0.0f && a.z == 0.0f;
}
