#pragma once

#include "math.h"

class Phong
{
public:
  __device__ Phong(const float3& kd, const float3& ks, float kg)
    : mKd(kd)
    , mKs(ks)
    , mKg(kg){};

  __device__ inline float3 Eval(const float3& wo,
                                const float3& wi,
                                const float3& shadingNormal)
  {
    return mKd + mKs * powf(Dot(Reflect(wi, shadingNormal), wo), mKg);
  }

private:
  float3 mKd, mKs;
  float mKg;
};
