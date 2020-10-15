#pragma once

#include <cuda_runtime.h>

#include "math.h"

template<typename Predicate>
__device__ int
FindInterval(int size, const Predicate& pred)
{
  int first = 0, len = size;

  while (len > 0) {
    int half = len >> 1, middle = first + half;
    if (pred(middle)) {
      first = middle + 1;
      len -= half + 1;
    } else {
      len = half;
    }
  }

  int ret = ClampI(first - 1, 0, size - 2);

  return ret;
}

// ===============================

class PiecewiseConst1D
{
public:
  __device__ PiecewiseConst1D(const float* f,
                              int nf,
                              const float* cdf,
                              float funcInt);

  __device__ float SampleContinuous(float p,
                                    float& pdf,
                                    int* offset = nullptr) const;
  __device__ float Pdf(float p) const;

  __device__ float Cdf(int index) const { return mCdf[index]; }
  __device__ float Func(int index) const { return mFunc[index]; }
  __device__ float FuncInt() const { return mFuncInt; }
  __device__ int N() const { return mNx; }

private:
  const float* mCdf;
  const float* mFunc;
  float mFuncInt = 0;
  int mNx = 0;
};

__device__ inline PiecewiseConst1D::PiecewiseConst1D(const float* f,
                                                     int nf,
                                                     const float* cdf,
                                                     float funcInt)
  : mCdf(cdf)
  , mFunc(f)
  , mFuncInt(funcInt)
  , mNx(nf)
{}

__device__ inline float
PiecewiseConst1D::SampleContinuous(float p, float& pdf, int* offset) const
{
  int off = FindInterval(mNx + 1, [&](int index) { return mCdf[index] <= p; });

  if (offset) {
    *offset = off;
  }

  float du = p - mCdf[off];
  if ((mCdf[off + 1] - mCdf[off]) > 0) {
    du /= mCdf[off + 1] - mCdf[off];
  }

  pdf = mFunc[off] / mFuncInt;
  float x = (off + du) / mNx;

  return x;
}

__device__ inline float
PiecewiseConst1D::Pdf(float p) const
{
  int index = static_cast<int>(p * mNx);
  index = ClampI(index, 0, mNx - 1);
  float pdf = mFunc[index] / mFuncInt;

  return pdf;
}

// ===============================

class PiecewiseConst2D
{
public:
  __device__ PiecewiseConst2D(const float* f,
                              const float* condCdf,
                              const float* condFuncInt,
                              const float* margCdf,
                              float margFuncInt,
                              int nRows,
                              int nCols);

  __device__ float2 SampleContinuous(const float2& p,
                                     float& pdf,
                                     int2* offset = nullptr) const;
  __device__ float Pdf(const float2& p) const;

private:
  const float* mFunc;
  const float* mCondCdf;
  const float* mCondFuncInt;
  const float* mMargCdf;
  const float mMargFuncInt;
  int mNRows = 0;
  int mNCols = 0;
};

__device__ inline PiecewiseConst2D::PiecewiseConst2D(const float* f,
                                                     const float* condCdf,
                                                     const float* condFuncInt,
                                                     const float* margCdf,
                                                     float margFuncInt,
                                                     int nRows,
                                                     int nCols)
  : mFunc(f)
  , mCondCdf(condCdf)
  , mCondFuncInt(condFuncInt)
  , mMargCdf(margCdf)
  , mMargFuncInt(margFuncInt)
  , mNRows(nRows)
  , mNCols(nCols)
{}

__device__ inline float2
PiecewiseConst2D::SampleContinuous(const float2& p,
                                   float& pdf,
                                   int2* offset) const
{
  PiecewiseConst1D pMarg(mCondFuncInt, mNRows, mMargCdf, mMargFuncInt);
  float yPdf;
  int yOff;
  float yP = pMarg.SampleContinuous(p.y, yPdf, &yOff);

  PiecewiseConst1D pCond(mFunc + yOff * mNCols,
                         mNCols,
                         mCondCdf + yOff * (mNCols + 1),
                         mCondFuncInt[yOff]);
  float xPdf;
  int xOff;
  float xP = pCond.SampleContinuous(p.x, xPdf, &xOff);

  if (offset) {
    *offset = make_int2(xOff, yOff);
  }

  pdf = xPdf * yPdf;

  return make_float2(xP, yP);
}

__device__ inline float
PiecewiseConst2D::Pdf(const float2& p) const
{
  int row = static_cast<int>(p.y * mNRows);
  row = ClampI(row, 0, mNRows - 1);

  int col = static_cast<int>(p.x * mNCols);
  col = ClampI(col, 0, mNCols - 1);

  float pdf = mFunc[row * mNCols + col] / mMargFuncInt;

  return pdf;
}
