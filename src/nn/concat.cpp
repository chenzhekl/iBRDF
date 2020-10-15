//
// Created by Zhe Chen on 2019/10/30.
//

#include "concat.h"

torch::Tensor
ConcatImpl::forward(const torch::Tensor& tensor)
{
  torch::Tensor m1Output = mModule1->forward(tensor);
  torch::Tensor m2Output = mModule2->forward(tensor);

  std::int64_t minShape2 = std::min(m1Output.size(2), m2Output.size(2));
  std::int64_t minShape3 = std::min(m1Output.size(3), m2Output.size(3));

  std::vector<torch::Tensor> outputs;
  if (m1Output.size(2) == minShape2 && m2Output.size(2) == minShape2 &&
      m1Output.size(3) == minShape3 && m2Output.size(3) == minShape3) {
    outputs.push_back(m1Output);
    outputs.push_back(m2Output);
  } else {
    std::int64_t diff2 = (m1Output.size(2) - minShape2) / 2;
    std::int64_t diff3 = (m1Output.size(3) - minShape3) / 2;

    outputs.push_back(m1Output.slice(2, diff2, diff2 + minShape2)
                        .slice(3, diff3, diff3 + minShape3));

    diff2 = (m2Output.size(2) - minShape2) / 2;
    diff3 = (m2Output.size(3) - minShape3) / 2;

    outputs.push_back(m2Output.slice(2, diff2, diff2 + minShape2)
                        .slice(3, diff3, diff3 + minShape3));
  }

  return torch::cat(outputs, mDim);
}
