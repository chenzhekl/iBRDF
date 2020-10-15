#pragma once

#include <cmath>
#include <cstdlib>

#include "math.h"
#include "merl_constant.h"
#include "pcg_rng.h"

// rotate vector along one axis
__device__ float3
RotateVector(float3 vector, float3 axis, float angle)
{
  float cosAngle = cosf(angle);
  float sinAngle = sinf(angle);
  float3 out = vector * cosAngle;

  float temp = Dot(axis, vector) * (1.0f - cosAngle);
  out = out + axis * temp;

  float3 cross = Cross(axis, vector);
  out = out + cross * sinAngle;

  return out;
}

// convert standard coordinates to half vector/difference vector coordinates
__device__ void
StdCoordsToHalfDiffCoords(float thetaIn,
                          float phiIn,
                          float thetaOut,
                          float phiOut,
                          float& thetaHalf,
                          float& phiHalf,
                          float& thetaDiff,
                          float& phiDiff)
{

  // compute in vector
  float inVecZ = cosf(thetaIn);
  float projInVec = sinf(thetaIn);
  float inVecX = projInVec * cosf(phiIn);
  float inVecY = projInVec * sinf(phiIn);
  float3 in = make_float3(inVecX, inVecY, inVecZ);
  in = Normalize(in);

  // compute out vector
  float outVecZ = cosf(thetaOut);
  float projOutVec = sinf(thetaOut);
  float outVecX = projOutVec * cosf(phiOut);
  float outVecY = projOutVec * sinf(phiOut);
  float3 out = make_float3(outVecX, outVecY, outVecZ);
  out = Normalize(out);

  // compute halfway vector
  float3 half = (in + out) * 0.5f;
  half = Normalize(half);

  // compute  theta_half, fi_half
  thetaHalf = acosf(half.z);
  phiHalf = atan2f(half.y, half.x);

  float3 biNormal = make_float3(0.0, 1.0, 0.0);
  float3 normal = make_float3(0.0, 0.0, 1.0);

  // compute diff vector
  float3 temp = RotateVector(in, normal, -phiHalf);
  float3 diff = RotateVector(temp, biNormal, -thetaHalf);

  // compute  theta_diff, fi_diff
  thetaDiff = acosf(diff.z);
  phiDiff = atan2f(diff.y, diff.x);
}

// Lookup theta_half index
// This is a non-linear mapping!
// In:  [0 .. pi/2]
// Out: [0 .. 89]
__device__ inline unsigned int
ThetaHalfIndex(float thetaHalf)
{
  if (thetaHalf <= 0.0f)
    return 0;
  float thetaHalfDeg = ((thetaHalf / (kPI / 2.0f)) * kBRDFSamplingResThetaH);
  float temp = thetaHalfDeg * kBRDFSamplingResThetaH;
  temp = sqrtf(temp);
  int retVal = (int)temp;
  if (retVal < 0)
    retVal = 0;
  if (retVal >= kBRDFSamplingResThetaH)
    retVal = kBRDFSamplingResThetaH - 1;
  return retVal;
}

// Lookup theta_diff index
// In:  [0 .. pi/2]
// Out: [0 .. 89]
__device__ inline unsigned int
ThetaDiffIndex(float thetaDiff)
{
  int tmp = int(thetaDiff / (kPI * 0.5f) * kBRDFSamplingResThetaD);
  if (tmp < 0)
    return 0;
  else if (tmp < kBRDFSamplingResThetaD - 1)
    return tmp;
  else
    return kBRDFSamplingResThetaD - 1;
}

// Lookup phi_diff index
__device__ inline unsigned int
PhiDiffIndex(float phiDiff)
{
  // Because of reciprocity, the BRDF is unchanged under
  // phi_diff -> phi_diff + M_PI
  if (phiDiff < 0.0)
    phiDiff += M_PI;

  // In: phi_diff in [0 .. pi]
  // Out: tmp in [0 .. 179]
  int tmp = int(phiDiff / M_PI * kBRDFSamplingResPhiD / 2);
  if (tmp < 0)
    return 0;
  else if (tmp < kBRDFSamplingResPhiD / 2 - 1)
    return tmp;
  else
    return kBRDFSamplingResPhiD / 2 - 1;
}

class MERL
{
public:
  __device__ explicit MERL(const float* brdf,
                           float pd = 0.7f,
                           float ps = 0.3f,
                           int n = 30);

  __device__ float3 Sample(const float3& wo,
                           const float2& sample,
                           const float3& shadingNormal,
                           float3& wi,
                           float& pdf,
                           unsigned int* id = nullptr) const;

  __device__ float Pdf(const float3& wo,
                       const float3& wi,
                       const float3& shadingNormal) const;

  __device__ float3 Eval(const float3& wo,
                         const float3& wi,
                         const float3& shadingNormal,
                         unsigned int* id = nullptr) const;

private:
  __device__ float3 Lookup(float thetaIn,
                           float phiIn,
                           float thetaOut,
                           float phiOut,
                           unsigned int* id = nullptr) const;

  __device__ float3 Lookup(const float3& wi,
                           const float3& wo,
                           unsigned int* id = nullptr) const;

  const float* mData;
  float mPd;
  float mPs;
  int mN;
  float mSpecularSamplingWeight;
};

__device__ inline MERL::MERL(const float* brdf, float pd, float ps, int n)
  : mData(brdf)
  , mPd(pd)
  , mPs(ps)
  , mN(n)
  , mSpecularSamplingWeight(ps / (ps + pd))
{}

// Given a pair of incoming/outgoing angles, look up the BRDF.
__device__ inline float3
MERL::Lookup(float thetaIn,
             float phiIn,
             float thetaOut,
             float phiOut,
             unsigned int* id) const
{
  // Convert to halfangle / difference angle coordinates
  float thetaHalf, phiHalf, thetaDiff, phiDiff;

  StdCoordsToHalfDiffCoords(
    thetaIn, phiIn, thetaOut, phiOut, thetaHalf, phiHalf, thetaDiff, phiDiff);

  // Find index.
  // Note that phi_half is ignored, since isotropic BRDFs are assumed
  unsigned int ind = PhiDiffIndex(phiDiff) +
                     ThetaDiffIndex(thetaDiff) * kBRDFSamplingResPhiD / 2 +
                     ThetaHalfIndex(thetaHalf) * kBRDFSamplingResPhiD / 2 *
                       kBRDFSamplingResThetaD;

  if (id) {
    *id = ind;
  }

  float redVal = mData[ind] * kRedScale;
  float greenVal = mData[ind + kBRDFSamplingResThetaH * kBRDFSamplingResThetaD *
                                 kBRDFSamplingResPhiD / 2] *
                   kGreenScale;
  float blueVal = mData[ind + kBRDFSamplingResThetaH * kBRDFSamplingResThetaD *
                                kBRDFSamplingResPhiD] *
                  kBlueScale;

  // if (redVal < 0.0 || greenVal < 0.0 || blueVal < 0.0)
  //   printf("Below horizon.\n");

  return make_float3(redVal, greenVal, blueVal);
}

__device__ inline float3
MERL::Lookup(const float3& wi, const float3& wo, unsigned int* id) const
{
  if (wi.z < 1e-7f || wo.z < 1e-7f) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  float thetaIn = SphericalTheta(wi);
  float phiIn = SphericalPhi(wi);
  float thetaOut = SphericalTheta(wo);
  float phiOut = SphericalPhi(wo);

  return Lookup(thetaIn, phiIn, thetaOut, phiOut, id);
}

__device__ inline float3
MERL::Sample(const float3& wo,
             const float2& sample,
             const float3& shadingNormal,
             float3& wi,
             float& pdf,
             unsigned int* id) const
{
  float3 u, v, w;
  BuildCoordSystem(shadingNormal, u, v, w);

  bool sampleSpecular = true;
  float2 scaledSample = sample;

  if (sample.x < mSpecularSamplingWeight) {
    scaledSample.x /= mSpecularSamplingWeight;
  } else {
    scaledSample.x =
      (sample.x - mSpecularSamplingWeight) / (1.0f - mSpecularSamplingWeight);
    sampleSpecular = false;
  }

  if (sampleSpecular) {
    float sinAlpha = sqrtf(1.0f - powf(scaledSample.y, 2.0f / (mN + 1.0f)));
    float cosAlpha = powf(scaledSample.y, 1.0f / (mN + 1.0f));
    float phi = (2.0f * kPI) * scaledSample.x;
    float3 localDir =
      make_float3(sinAlpha * cosf(phi), sinAlpha * sinf(phi), cosAlpha);

    float3 r = Reflect(wo, shadingNormal);
    float3 u, v, w;
    BuildCoordSystem(r, u, v, w);
    wi = NormalizeF(localDir.x * u + localDir.y * v + localDir.z * w);

    if (Dot(wi, shadingNormal) <= 1e-7f) {
      pdf = 0.0f;
      return make_float3(0.0f, 0.0f, 0.0f);
    }
  } else {
    float diffusePdf;
    wi = CosineSampleHemisphere(scaledSample, diffusePdf);
    wi = NormalizeF(wi.x * u + wi.y * v + wi.z * w);
  }

  pdf = Pdf(wo, wi, shadingNormal);

  if (pdf == 0.0f) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  float3 localWi = MapToLocalCoordSystem(wi, u, v, w);
  float3 localWo = MapToLocalCoordSystem(wo, u, v, w);

  return Lookup(localWi, localWo, id);
}

__device__ inline float
MERL::Pdf(const float3& wo, const float3& wi, const float3& shadingNormal) const
{
  if (Dot(wo, shadingNormal) <= 0.0f || Dot(wi, shadingNormal) <= 0.0f) {
    return 0.0f;
  }

  float diffuseProb = Dot(wi, shadingNormal) * kInvPI;

  float cosAlpha = Dot(wi, Reflect(wo, shadingNormal));
  float specularProb = 0.0f;
  if (cosAlpha > 0.0f) {
    specularProb = powf(cosAlpha, mN) * (mN + 1.0f) * kInvPI * 0.5f;
  }

  float pdf = mSpecularSamplingWeight * specularProb +
              (1.0f - mSpecularSamplingWeight) * diffuseProb;

  return pdf;
}

__device__ inline float3
MERL::Eval(const float3& wo,
           const float3& wi,
           const float3& shadingNormal,
           unsigned int* id) const
{
  float3 u, v, w;
  BuildCoordSystem(shadingNormal, u, v, w);

  float3 localWi = MapToLocalCoordSystem(wi, u, v, w);
  float3 localWo = MapToLocalCoordSystem(wo, u, v, w);

  return Lookup(localWi, localWo, id);
}
