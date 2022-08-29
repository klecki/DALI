// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "dali/pipeline/operator/builtin/split.h"
#include "dali/core/util.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"

namespace dali {

template <typename Backend>
bool Split<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                               const workspace_t<Backend> &ws) {
  const auto &input = ws.template Input<Backend>(0);
  const auto &predicate = ws.ArgumentInput("predicate");
  DALI_ENFORCE(
      input.num_samples() == predicate.num_samples(),
      make_string("Split description must cover whole input, got ", input.num_samples(),
                  " input samples and ", predicate.num_samples(), " elements denoting the split."));
  for (int i = 0; i < predicate.num_samples(); i++) {
    DALI_ENFORCE(predicate[i].shape() == TensorShape<0>(), "Only scalar indexing is supported.");
    // int output_category = *predicate.template tensor<bool>(i);
  }

  category_counts_.fill(0);

  for (int i = 0; i < predicate.num_samples(); i++) {
    int output_category = *predicate.template tensor<bool>(i);
    category_counts_[output_category]++;
  }


  // TODO(klecki): we can construct the output_desc, it won't be useful now
  output_desc.resize(kMaxCategories);  // we only support two for now, so there is no dynamic split
  // for (int i = 0; i < predicate.num_samples(); i++) {

  // }
  // for (auto &desc : output_desc) {
  //   desc.shape = input.shape();
  //   desc.type = input.type();
  // }
  return false;
}

template <typename Backend>
void Split<Backend>::RunImpl(workspace_t<Backend> &ws) {
  const auto &input = ws.template Input<Backend>(0);
  const auto &predicate = ws.ArgumentInput("predicate");
  auto category_output_idx = uniform_array<kMaxCategories>(0);

  for (int output_category = 0; output_category < kMaxCategories; output_category++) {
    auto &output = ws.template Output<Backend>(output_category);

    // We can (and need to) do it only once, for each new output instance, when it doesn't have
    // data yet. It should be consistent across iterations.
    if (!output.has_data()) {
      output.SetupLike(input);
    }
    output.SetSize(category_counts_[output_category]);
  }

  for (int input_idx = 0; input_idx < predicate.num_samples(); input_idx++) {
    int output_category = *predicate.template tensor<bool>(input_idx);
    auto &output = ws.template Output<Backend>(output_category);

    // get the output index and increment for the next case.
    int output_idx = category_output_idx[output_category];
    category_output_idx[output_category]++;

    // share the sample to the output
    output.UnsafeSetSample(output_idx, input, input_idx);
  }
}


DALI_SCHEMA(Split)
    .DocStr(R"code(Split batch based on a predicate.)code")
    .NumInput(1)
    .NumOutput(2)
    .PassThrough({{0, 0}})  //todo add special pass through
    .AddArg("predicate", "Boolean categorization of the input batch", DALI_BOOL, true)
    .MakeInternal();

DALI_REGISTER_OPERATOR(Split, Split<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Split, Split<GPUBackend>, GPU);

}  // namespace dali
