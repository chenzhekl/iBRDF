#include "piecewise.h"

namespace ibrdf {
// =================================
PiecewiseConst1D::PiecewiseConst1D(const torch::Tensor& f)
{
  torch::NoGradGuard noGradGuard;

  mFunc = f.clone();
  mFuncAccessor = mFunc.accessor<float, 1>();
  mCdf = torch::cat({ torch::tensor({ 0.0f }), (f / f.size(0)).cumsum(0) });

  torch::TensorAccessor mCdfAccessor = mCdf.accessor<float, 1>();

  mFuncInt = mCdfAccessor[mCdf.size(0) - 1];
  if (mFuncInt == 0.0f) {
    mCdf /= f.size(0);
  } else {
    mCdf /= mFuncInt;
  }
  mNx = f.size(0);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PiecewiseConst1D::sampleContinuous(const torch::Tensor& p) const
{
  torch::NoGradGuard noGradGuard;

  torch::Tensor offset = torch::empty_like(p, torch::kInt64);
  torch::Tensor x = torch::empty_like(p);
  torch::Tensor pdf = torch::empty_like(p);

  torch::TensorAccessor offsetAccessor = offset.accessor<std::int64_t, 1>();
  torch::TensorAccessor xAccessor = x.accessor<float, 1>();
  torch::TensorAccessor pdfAccessor = pdf.accessor<float, 1>();
  torch::TensorAccessor pAccessor = p.accessor<float, 1>();
  torch::TensorAccessor mCdfAccessor = mCdf.accessor<float, 1>();
  //  torch::TensorAccessor mFuncAccessor = mFunc.accessor<float, 1>();

  for (std::int64_t i = 0; i < pAccessor.size(0); ++i) {
    offsetAccessor[i] = findInterval(mCdfAccessor.size(0), [&](long index) {
      return mCdfAccessor[index] <= pAccessor[i];
    });

    float du = pAccessor[i] - mCdfAccessor[offsetAccessor[i]];
    if ((mCdfAccessor[offsetAccessor[i] + 1] -
         mCdfAccessor[offsetAccessor[i]]) > 0.0f) {
      du /=
        (mCdfAccessor[offsetAccessor[i] + 1] - mCdfAccessor[offsetAccessor[i]]);
    }

    pdfAccessor[i] = mFuncAccessor[offsetAccessor[i]] / mFuncInt;
    xAccessor[i] = (offsetAccessor[i] + du) / mNx;
  }

  return std::make_tuple(x, pdf, offset);
}

std::tuple<float, float, std::int64_t>
PiecewiseConst1D::sampleContinuousP(float p) const
{
  torch::NoGradGuard noGradGuard;

  float x, pdf;
  std::int64_t offset;

  torch::TensorAccessor mCdfAccessor = mCdf.accessor<float, 1>();
  //  torch::TensorAccessor mFuncAccessor = mFunc.accessor<float, 1>();

  offset = findInterval(mCdfAccessor.size(0),
                        [&](long index) { return mCdfAccessor[index] <= p; });
  float du = p - mCdfAccessor[offset];
  if ((mCdfAccessor[offset + 1] - mCdfAccessor[offset]) > 0.0f) {
    du /= (mCdfAccessor[offset + 1] - mCdfAccessor[offset]);
  }

  pdf = mFuncAccessor[offset] / mFuncInt;
  x = (offset + du) / mNx;

  return std::make_tuple(x, pdf, offset);
}

torch::Tensor
PiecewiseConst1D::pdf(const torch::Tensor& p) const
{
  torch::NoGradGuard noGradGuard;

  torch::Tensor indices = (p * mNx).to(torch::kInt64).clamp(0, mNx - 1);
  torch::Tensor pdf = mFunc.index_select(0, indices) / mFuncInt;

  return pdf;
}

float
PiecewiseConst1D::pdfP(float p) const
{
  torch::NoGradGuard noGradGuard;

  auto index = static_cast<std::int64_t>(p * mNx);
  index = std::clamp<std::int64_t>(index, 0, mNx - 1);
  float pdf = mFuncAccessor[index] / mFuncInt;

  return pdf;
}

// ============================

PiecewiseConst2D::PiecewiseConst2D(const torch::Tensor& f)
{
  torch::NoGradGuard noGradGuard;

  for (std::int64_t row = 0; row < f.size(0); ++row) {
    mPyOnx.emplace_back(f[row]);
  }

  torch::Tensor marginalFunc = torch::empty({ f.size(0) });
  torch::TensorAccessor marginalFuncAccessor =
    marginalFunc.accessor<float, 1>();
  for (std::size_t i = 0; i < mPyOnx.size(); ++i) {
    marginalFuncAccessor[i] = mPyOnx[i].funcInt();
  }

  mPx = PiecewiseConst1D(marginalFunc);
  mNx = f.size(0);
  mNy = f.size(1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PiecewiseConst2D::sampleContinuous(const torch::Tensor& p) const
{
  torch::NoGradGuard noGradGuard;

  torch::TensorAccessor pAccessor = p.accessor<float, 2>();

  torch::Tensor x = torch::empty({ p.size(0), 2 });
  torch::Tensor pdf = torch::empty({ p.size(0) });
  torch::Tensor offset = torch::empty({ p.size(0), 2 }, torch::kInt64);

  torch::TensorAccessor offsetAccessor = offset.accessor<std::int64_t, 2>();
  torch::TensorAccessor xAccessor = x.accessor<float, 2>();
  torch::TensorAccessor pdfAccessor = pdf.accessor<float, 1>();

  torch::Tensor xOffset;
  torch::Tensor xP;
  torch::Tensor xPdf;
  std::tie(xP, xPdf, xOffset) = mPx.sampleContinuous(p.select(1, 0));

  torch::TensorAccessor xOffsetAccessor = xOffset.accessor<std::int64_t, 1>();
  torch::TensorAccessor xPAccessor = xP.accessor<float, 1>();
  torch::TensorAccessor xPdfAccessor = xPdf.accessor<float, 1>();

  for (std::int64_t row = 0; row < p.size(0); ++row) {
    float yP, yPdf;
    std::int64_t yOffset;
    std::tie(yP, yPdf, yOffset) =
      mPyOnx[xOffsetAccessor[row]].sampleContinuousP(pAccessor[row][1]);

    xAccessor[row][0] = xPAccessor[row];
    xAccessor[row][1] = yP;

    pdfAccessor[row] = xPdfAccessor[row] * yPdf;

    offsetAccessor[row][0] = xOffsetAccessor[row];
    offsetAccessor[row][1] = yOffset;
  }

  return std::make_tuple(x, pdf, offset);
}

std::tuple<torch::Tensor, float, torch::Tensor>
PiecewiseConst2D::sampleContinuousP(const torch::Tensor& p) const
{
  torch::NoGradGuard noGradGuard;

  torch::TensorAccessor pAccessor = p.accessor<float, 1>();

  float pdf;
  torch::Tensor x = torch::empty({ 2 });
  torch::Tensor offset = torch::empty({ 2 }, torch::kInt64);

  torch::TensorAccessor offsetAccessor = offset.accessor<std::int64_t, 1>();
  torch::TensorAccessor xAccessor = x.accessor<float, 1>();

  float xP, xPdf;
  std::int64_t xOffset;
  std::tie(xP, xPdf, xOffset) = mPx.sampleContinuousP(pAccessor[0]);

  float yP, yPdf;
  std::int64_t yOffset;
  std::tie(yP, yPdf, yOffset) = mPyOnx[xOffset].sampleContinuousP(pAccessor[1]);

  xAccessor[0] = xP;
  xAccessor[1] = yP;

  pdf = xPdf * yPdf;

  offsetAccessor[0] = xOffset;
  offsetAccessor[1] = yOffset;

  return std::make_tuple(x, pdf, offset);
}

torch::Tensor
PiecewiseConst2D::pdf(const torch::Tensor& p) const
{
  torch::NoGradGuard noGradGuard;

  torch::Tensor pdf = torch::empty({ p.size(0) });
  torch::Tensor ix = (p.select(1, 0) * mNx).to(torch::kInt64).clamp(0, mNx - 1);
  torch::Tensor iy = (p.select(1, 1) * mNy).to(torch::kInt64).clamp(0, mNy - 1);

  torch::TensorAccessor pdfAccessor = pdf.accessor<float, 1>();
  torch::TensorAccessor ixAccessor = ix.accessor<float, 1>();
  torch::TensorAccessor iyAccessor = iy.accessor<float, 1>();

  for (std::int64_t row = 0; row < pdf.size(0); ++row) {
    pdfAccessor[row] =
      mPyOnx[ixAccessor[row]].func(iyAccessor[row]) / mPx.funcInt();
  }

  return pdf;
}

float
PiecewiseConst2D::pdfP(const torch::Tensor& p) const
{
  torch::NoGradGuard noGradGuard;

  torch::TensorAccessor pAccessor = p.accessor<float, 1>();

  auto ix = static_cast<std::int64_t>(pAccessor[0] * mNx);
  ix = std::clamp<std::int64_t>(ix, 0, mNx - 1);
  auto iy = static_cast<std::int64_t>(pAccessor[1] * mNy);
  iy = std::clamp<std::int64_t>(iy, 0, mNy - 1);

  float pdf = mPyOnx[ix].func(iy) / mPx.funcInt();

  return pdf;
}

// ======================================

// f: nx x ny x nz
PiecewiseConst3D::PiecewiseConst3D(const torch::Tensor& f)
{
  torch::NoGradGuard noGradGuard;

  for (std::int64_t ix = 0; ix < f.size(0); ++ix) {
    mPzOnxy.emplace_back(std::vector<PiecewiseConst1D>());
    for (std::int64_t iy = 0; iy < f.size(1); ++iy) {
      mPzOnxy[mPzOnxy.size() - 1].emplace_back(PiecewiseConst1D(f[ix][iy]));
    }
  }

  torch::Tensor xys = torch::empty({ f.size(0), f.size(1) });
  torch::TensorAccessor xysAccessor = xys.accessor<float, 2>();
  for (std::int64_t ix = 0; ix < f.size(0); ++ix) {
    for (std::int64_t iy = 0; iy < f.size(1); ++iy) {
      xysAccessor[ix][iy] = mPzOnxy[ix][iy].funcInt();
    }
  }
  mPxy = PiecewiseConst2D(xys);

  mNx = f.size(0);
  mNy = f.size(1);
  mNz = f.size(2);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PiecewiseConst3D::sampleContinuous(const torch::Tensor& p) const
{
  torch::NoGradGuard noGradGuard;

  torch::TensorAccessor pAccessor = p.accessor<float, 2>();

  torch::Tensor xyOffset;
  torch::Tensor xyP;
  torch::Tensor xyPdf;
  std::tie(xyP, xyPdf, xyOffset) = mPxy.sampleContinuous(p.slice(1, 0, 2));

  torch::TensorAccessor xyOffsetAccessor = xyOffset.accessor<std::int64_t, 2>();
  torch::TensorAccessor xyPAccessor = xyP.accessor<float, 2>();
  torch::TensorAccessor xyPdfAccessor = xyPdf.accessor<float, 1>();

  torch::Tensor offset = torch::empty({ p.size(0), 3 }, torch::kInt64);
  torch::Tensor x = torch::empty({ p.size(0), 3 });
  torch::Tensor pdf = torch::empty({ p.size(0) });

  torch::TensorAccessor offsetAccessor = offset.accessor<std::int64_t, 2>();
  torch::TensorAccessor xAccessor = x.accessor<float, 2>();
  torch::TensorAccessor pdfAccessor = pdf.accessor<float, 1>();

  for (std::int64_t row = 0; row < p.size(0); ++row) {
    std::int64_t zOffset;
    float zP, zPdf;
    std::tie(zP, zPdf, zOffset) =
      mPzOnxy[xyOffsetAccessor[row][0]][xyOffsetAccessor[row][1]]
        .sampleContinuousP(pAccessor[row][2]);

    xAccessor[row][0] = xyPAccessor[row][0];
    xAccessor[row][1] = xyPAccessor[row][1];
    xAccessor[row][2] = zP;

    pdfAccessor[row] = xyPdfAccessor[row] * zPdf;

    offsetAccessor[row][0] = xyOffsetAccessor[row][0];
    offsetAccessor[row][1] = xyOffsetAccessor[row][1];
    offsetAccessor[row][2] = zOffset;
  }

  return std::make_tuple(x, pdf, offset);
}

torch::Tensor
PiecewiseConst3D::pdf(const torch::Tensor& p) const
{
  torch::NoGradGuard noGradGuard;

  torch::TensorAccessor pAccessor = p.accessor<float, 2>();

  torch::Tensor pdf = torch::empty({ p.size(0) });
  torch::Tensor ix = (p.select(1, 0) * mNx).to(torch::kInt64).clamp(0, mNx - 1);
  torch::Tensor iy = (p.select(1, 1) * mNy).to(torch::kInt64).clamp(0, mNy - 1);

  torch::TensorAccessor pdfAccessor = pdf.accessor<float, 1>();
  torch::TensorAccessor ixAccessor = ix.accessor<float, 1>();
  torch::TensorAccessor iyAccessor = iy.accessor<float, 1>();

  for (std::int64_t row = 0; row < p.size(0); ++row) {
    pdfAccessor[row] =
      mPzOnxy[ixAccessor[row]][iyAccessor[row]].pdfP(pAccessor[row][2]) *
      mPxy.pdfP(p[row].slice(0, 0, 2));
  }

  return pdf;
}
} // namespace ibrdf
