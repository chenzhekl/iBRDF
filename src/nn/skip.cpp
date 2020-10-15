//
// Created by Zhe Chen on 2019/09/09.
//

#include "skip.h"

#include "concat.h"

SkipImpl::SkipImpl(std::int64_t inFeatures,
                   std::int64_t outFeatures,
                   const std::vector<std::int64_t>& numChannelsDown,
                   const std::vector<std::int64_t>& numChannelsUp,
                   const std::vector<std::int64_t>& numChannelsSkip,
                   std::int64_t filterSizeDown,
                   std::int64_t filterSizeUp,
                   std::int64_t filterSizeSkip,
                   bool needBias,
                   bool need1x1Up)
{
  long padDown = static_cast<long>((filterSizeDown - 1) / 2);
  long padUp = static_cast<long>((filterSizeUp - 1) / 2);
  long padSkip = static_cast<long>((filterSizeSkip - 1) / 2);
  std::vector<long> padSizeDown = { padDown, padDown, padDown, padDown };
  std::vector<long> padSizeUp = { padUp, padUp, padUp, padUp };
  std::vector<long> padSizeSkip = { padSkip, padSkip, padSkip, padSkip };

  std::int64_t numScales = numChannelsDown.size();
  std::int64_t lastScale = numScales - 1;
  StackSequentail model;
  StackSequentail modelTmp = model;

  torch::nn::LeakyReLU actFn(
    torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true));

  std::int64_t inputDepth = inFeatures;
  for (std::int64_t i = 0; i < numScales; ++i) {
    StackSequentail deeper;
    StackSequentail skip;

    if (numChannelsSkip[i] != 0) {
      modelTmp->push_back(Concat(1, skip, deeper));
    } else {
      modelTmp->push_back(deeper);
    }

    modelTmp->push_back(torch::nn::BatchNorm2d(
      numChannelsSkip[i] +
      (i < lastScale ? numChannelsUp[i + 1] : numChannelsDown[i])));

    if (numChannelsSkip[i] != 0) {
      skip->push_back(
        torch::nn::Functional(torch::reflection_pad2d, padSizeSkip));
      skip->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(inputDepth, numChannelsSkip[i], filterSizeSkip)
          .bias(needBias)));
      skip->push_back(torch::nn::BatchNorm2d(numChannelsSkip[i]));
      skip->push_back(actFn);
    }

    deeper->push_back(
      torch::nn::Functional(torch::reflection_pad2d, padSizeDown));
    deeper->push_back(torch::nn::Conv2d(
      torch::nn::Conv2dOptions(inputDepth, numChannelsDown[i], filterSizeDown)
        .stride(2)
        .bias(needBias)));
    deeper->push_back(torch::nn::BatchNorm2d(numChannelsDown[i]));
    deeper->push_back(actFn);

    deeper->push_back(
      torch::nn::Functional(torch::reflection_pad2d, padSizeDown));
    deeper->push_back(torch::nn::Conv2d(
      torch::nn::Conv2dOptions(
        numChannelsDown[i], numChannelsDown[i], filterSizeDown)
        .bias(needBias)));
    deeper->push_back(torch::nn::BatchNorm2d(numChannelsDown[i]));
    deeper->push_back(actFn);

    StackSequentail deeperMain;

    std::int64_t k;
    if (i == lastScale) {
      k = numChannelsDown[i];
    } else {
      deeper->push_back(deeperMain);
      k = numChannelsUp[i + 1];
    }

    deeper->push_back(
      torch::nn::Upsample(torch::nn::UpsampleOptions()
                            .scale_factor(std::vector{ 2.0, 2.0 })
                            .mode(torch::kBilinear)
                            .align_corners(false)));

    modelTmp->push_back(
      torch::nn::Functional(torch::reflection_pad2d, padSizeUp));
    modelTmp->push_back(torch::nn::Conv2d(
      torch::nn::Conv2dOptions(
        numChannelsSkip[i] + k, numChannelsUp[i], filterSizeUp)
        .bias(needBias)));
    modelTmp->push_back(torch::nn::BatchNorm2d(numChannelsUp[i]));
    modelTmp->push_back(actFn);

    if (need1x1Up) {
      modelTmp->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(numChannelsUp[i], numChannelsUp[i], 1)
          .bias(needBias)));
      modelTmp->push_back(torch::nn::BatchNorm2d(numChannelsUp[i]));
      modelTmp->push_back(actFn);
    }

    inputDepth = numChannelsDown[i];
    modelTmp = deeperMain;
  }

  model->push_back(torch::nn::Conv2d(
    torch::nn::Conv2dOptions(numChannelsUp[0], outFeatures, 1).bias(needBias)));

  // model->push_back(torch::nn::Functional(torch::relu));

  mModel = model;

  register_module("Module", mModel);
  register_parameter("Scale", mScale);
}

torch::Tensor
SkipImpl::forward(const torch::Tensor& x)
{
  return mModel->forward(x).exp();
}
