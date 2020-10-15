//
// Created by Zhe Chen on 2019/08/27.
//

#include "piecewise_quadratic.h"

namespace ibrdf {
// inline long
// LowerBound(float x, const torch::TensorAccessor<float, 3>& bins, long i, long
// j)
// {
//   long count = bins.size(2);
//   long first = 0;

//   while (count > 0) {
//     long it = first;
//     long step = count / 2;
//     it += step;
//     if (x > bins[i][j][it]) {
//       first = ++it;
//       count -= step + 1;
//     } else {
//       count = step;
//     }
//   }

//   return first;
// }

// inline long
// upperBound(float x, const torch::TensorAccessor<float, 3>& bins, long i, long
// j)
// {
//   long count = bins.size(2);
//   long first = 0;

//   while (count > 0) {
//     long it = first;
//     long step = count / 2;
//     it += step;
//     if (x >= bins[i][j][it]) {
//       first = ++it;
//       count -= step + 1;
//     } else {
//       count = step;
//     }
//   }

//   return first;
// }

// TODO: further improve its performance (bottleneck)
// torch::Tensor
// searchSorted3d(const torch::Tensor& x, const torch::Tensor& indices)
// {
//   torch::Device device = x.device();
//   torch::Tensor ret = torch::empty({ x.size(0), x.size(1) },
//                                    torch::TensorOptions().dtype(torch::kLong));
//   torch::Tensor xCPU = x.to(torch::kCPU);
//   torch::Tensor indicesCPU = indices.to(torch::kCPU);

//   torch::TensorAccessor retAccessor = ret.accessor<long, 2>();
//   torch::TensorAccessor indicesCPUAccessor = indicesCPU.accessor<float, 2>();
//   torch::TensorAccessor xCPUAccessor = xCPU.accessor<float, 3>();

// #pragma omp parallel for collapse(2) default(none)                             \
//   shared(indicesCPUAccessor, xCPUAccessor, retAccessor, x)
//   for (long i = 0; i < x.size(0); ++i) {
//     for (long j = 0; j < x.size(1); ++j) {
//       retAccessor[i][j] =
//         upperBound(indicesCPUAccessor[i][j], xCPUAccessor, i, j) - 1;
//     }
//   }

//   return ret.to(device);
// }

std::tuple<torch::Tensor, torch::Tensor>
PiecewiseQuadraticCouplingImpl::coupleForward(
  const torch::Tensor& xa,
  const torch::Tensor& xb,
  const std::optional<torch::Tensor>& embedCode)
{
  auto [v, w] = getParams(xa, xb, embedCode);

  torch::Tensor wCumSum = w.roll(1, -1);
  wCumSum.select(-1, 0) = 0.0f;
  wCumSum = wCumSum.to(torch::kFloat64).cumsum(-1).to(w.dtype());

  //  std::clog << "Before ss3d, ";

  // torch::Tensor b = searchSorted3d(wCumSum, xb).unsqueeze(-1);
  torch::Tensor b =
    torch::searchsorted(wCumSum.detach().squeeze(1), xb.detach(), false, true)
      .unsqueeze(-1) -
    1;

  //  std::clog << "After ss3d, ";

  torch::Tensor a = xb - wCumSum.gather(-1, b).squeeze(-1);
  torch::Tensor aNormalized = a / w.gather(-1, b).squeeze(-1);

  torch::Tensor vw = ((v + v.roll(1, -1)).slice(-1, 1) / 2.0f * w).roll(1, -1);
  vw.select(-1, 0) = 0.0f;
  torch::Tensor vwCumSum = vw.to(torch::kFloat64).cumsum(-1).to(vw.dtype());

  torch::Tensor vIb1 = v.gather(-1, b + 1).squeeze(-1);
  torch::Tensor vIb = v.gather(-1, b).squeeze(-1);

  torch::Tensor yb =
    a * a / (2.0f * w.gather(-1, b).squeeze(-1)) * (vIb1 - vIb) + a * vIb +
    vwCumSum.gather(-1, b).squeeze(-1);

  // Workaround rounding errors
  yb.masked_scatter_(yb >= 1.0f, yb - 1e-6f);

  torch::Tensor det = vIb + aNormalized * (vIb1 - vIb);
  torch::Tensor logDet = det.log().sum(-1);

  return { yb, logDet };
}

std::tuple<torch::Tensor, torch::Tensor>
PiecewiseQuadraticCouplingImpl::coupleInverse(
  const torch::Tensor& ya,
  const torch::Tensor& yb,
  const std::optional<torch::Tensor>& embedCode)
{
  auto [v, w] = getParams(ya, yb, embedCode);

  torch::Tensor vw = ((v + v.roll(1, -1)).slice(-1, 1) / 2.0f * w).roll(1, -1);
  vw.select(-1, 0) = 0.0f;
  torch::Tensor vwCumSum = vw.to(torch::kFloat64).cumsum(-1).to(vw.dtype());

  //  std::clog << "Before ss3d, ";

  // torch::Tensor b = searchSorted3d(vwCumSum, yb).unsqueeze(-1);
  torch::Tensor b =
    torch::searchsorted(vwCumSum.detach().squeeze(1), yb.detach(), false, true)
      .unsqueeze(-1) -
    1;

  //  std::clog << "After ss3d, ";

  torch::Tensor vIb1 = v.gather(-1, b + 1).squeeze(-1);
  torch::Tensor vIb = v.gather(-1, b).squeeze(-1);

  torch::Tensor eqA = (vIb1 - vIb) / (2.0f * w.gather(-1, b).squeeze(-1));
  torch::Tensor& eqB = vIb;
  torch::Tensor eqC = vwCumSum.gather(-1, b).squeeze(-1) - yb;

  torch::Tensor disc = (eqB * eqB - 4.0f * eqA * eqC).sqrt();
  // Ref: https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
  torch::Tensor eqSol1 = (-eqB - disc) / (2.0f * eqA);
  torch::Tensor eqSol2 = (2.0f * eqC) / (-eqB - disc);

  // * is used here instead of &. Because bitwise_and is still unsupported in
  // ATen
  torch::Tensor a = torch::where(
    (eqSol1 > 0.0f) * (eqSol1.abs() < eqSol2.abs()), eqSol1, eqSol2);
  torch::Tensor aNormalized = a / w.gather(-1, b).squeeze(-1);

  torch::Tensor wCumSum = w.roll(1, -1);
  wCumSum.select(-1, 0) = 0.0f;
  wCumSum = wCumSum.to(torch::kFloat64).cumsum(-1).to(w.dtype());

  torch::Tensor xb = a + wCumSum.gather(-1, b).squeeze(-1);

  // Workaround rounding errors
  xb.masked_scatter_(xb >= 1.0f, xb - 1e-6f);

  torch::Tensor det = vIb + aNormalized * (vIb1 - vIb);
  torch::Tensor logDet = -det.log().sum(-1);

  return { xb, logDet };
}

std::tuple<torch::Tensor, torch::Tensor>
PiecewiseQuadraticCouplingImpl::getParams(
  const torch::Tensor& xa,
  const torch::Tensor& xb,
  const std::optional<torch::Tensor>& embedCode)
{
  torch::Tensor netIn;

  if (embedCode) {
    netIn = torch::cat({ xa, *embedCode }, -1);
  } else {
    netIn = xa;
  }

  torch::Tensor params = mNetM->forward(netIn);
  params =
    params.view({ params.size(0), xb.size(1), static_cast<long>(2 * mK + 1) });

  std::vector<torch::Tensor> vw = params.split_with_sizes(
    { static_cast<long>(mK + 1), static_cast<long>(mK) }, -1);
  torch::Tensor w = torch::softmax(vw[1], -1);
  torch::Tensor vExp = vw[0].exp();
  torch::Tensor vNormalizer =
    ((vExp.roll(1, -1) + vExp).slice(-1, 1) / 2.0f * w).sum(-1, true);
  torch::Tensor v = vExp / vNormalizer;

  return { v, w };
}
} // namespace ibrdf
