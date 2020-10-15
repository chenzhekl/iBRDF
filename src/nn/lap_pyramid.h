#include <torch/torch.h>

class LapPyramidLossImpl : public torch::nn::Module
{
public:
  explicit LapPyramidLossImpl(std::size_t numLevels);

private:
};

TORCH_MODULE(LapPyramidLoss);
