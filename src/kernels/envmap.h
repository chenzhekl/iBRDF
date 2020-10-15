#pragma once

#include "dist.h"
#include "math.h"

__device__ unsigned int
GetEnvMapIndex(const float3& wi, unsigned int width, unsigned int height)
{
  auto col = static_cast<unsigned int>(width * SphericalPhi(wi) * kInvPI * 0.5);
  auto row = static_cast<unsigned int>(height * SphericalTheta(wi) * kInvPI);

  col = Clamp(col, 0u, width - 1);
  row = Clamp(row, 0u, height - 1);

  return row * width + col;
}

__device__ unsigned int
GetEnvMapIndexNM(const float3& wi, unsigned int width, unsigned int height)
{
  float theta = acosf(wi.y);
  float phi = atan2f(wi.x, wi.z);
  phi += 0.5f * kPI;
  if (phi < 0.0f) {
    phi += 2.0 * kPI;
  }

  auto col = static_cast<unsigned int>(width * phi * kInvPI * 0.5);
  auto row = static_cast<unsigned int>(height * theta * kInvPI);

  col = Clamp(col, 0u, width - 1);
  row = Clamp(row, 0u, height - 1);

  return row * width + col;
}

class EnvMap
{
public:
  __device__ EnvMap(float* data,
                    unsigned int width,
                    unsigned int height,
                    const float* f,
                    const float* condCdf,
                    const float* condFuncInt,
                    const float* margCdf,
                    float margFuncInt);

  __device__ float3 Sample(const float2& sample, float3& wi, float& pdf) const;

  __device__ float Pdf(const float3& wi) const;

private:
  float* mData;
  unsigned int mWidth;
  unsigned int mHeight;

  PiecewiseConst2D mSamplingDist;
};

__device__ inline EnvMap::EnvMap(float* data,
                                 unsigned int width,
                                 unsigned int height,
                                 const float* f,
                                 const float* condCdf,
                                 const float* condFuncInt,
                                 const float* margCdf,
                                 float margFuncInt)
  : mData(data)
  , mWidth(width)
  , mHeight(height)
  , mSamplingDist(f, condCdf, condFuncInt, margCdf, margFuncInt, height, width)
{}

__device__ inline float3
EnvMap::Sample(const float2& sample, float3& wi, float& pdf) const
{
  float mapPdf;
  float2 uv = mSamplingDist.SampleContinuous(sample, mapPdf);
  if (mapPdf < 1e-7f) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }
  float theta = uv.y * kPI;
  float phi = uv.x * 2.0f * kPI;

  float sinTheta = sinf(theta);
  float cosTheta = cosf(theta);
  float sinPhi = sinf(phi);
  float cosPhi = cosf(phi);
  wi = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);

  pdf = mapPdf / (2.0f * kPI * kPI * sinTheta);
  if (sinTheta < 1e-7f) {
    pdf = 0.0f;
  }

  auto col = static_cast<unsigned int>(mWidth * uv.x);
  auto row = static_cast<unsigned int>(mHeight * uv.y);

  col = Clamp(col, 0u, mWidth - 1);
  row = Clamp(row, 0u, mHeight - 1);
  unsigned int ind = row * mWidth + col;

  float3 li =
    make_float3(mData[ind * 3 + 0], mData[ind * 3 + 1], mData[ind * 3 + 2]);

  return li;
}

__device__ inline float
EnvMap::Pdf(const float3& wi) const
{
  float theta = SphericalTheta(wi);
  float phi = SphericalPhi(wi);
  float sinTheta = sinf(theta);
  if (sinTheta > 1e-7f) {
    return mSamplingDist.Pdf(make_float2(phi * kInvPI * 0.5, theta * kInvPI)) /
           (2.0f * kPI * kPI * sinTheta);
  } else {
    return 0.0f;
  }
}
