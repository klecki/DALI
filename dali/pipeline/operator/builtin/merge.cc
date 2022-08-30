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

#include "dali/pipeline/operator/builtin/merge.h"
#include "dali/core/util.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"

namespace dali {

template <typename Backend>
bool Merge<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                               const workspace_t<Backend> &ws) {
  input_sample_count_ = 0;
  int nonzero_input_idx = -1;
  for (int input_category = 0; input_category < kMaxCategories; input_category++) {
    const auto &input = ws.template Input<Backend>(input_category);
    input_sample_count_ += input.num_samples();
    // TODO(klecki): do not compare against empty inputs unless we ensure consistent run behaviour
    // for empty samples.
    if (input.num_samples() > 0) {
      nonzero_input_idx = input_category;
    }
    if (nonzero_input_idx >= 0) {
      const auto &base_input = ws.template Input<Backend>(nonzero_input_idx);
      // TODO(klecki): Error messages. BTW, we can just let it explode in Run, TV already makes
      // sure that those are ok
      DALI_ENFORCE(base_input.shape().sample_dim() == input.shape().sample_dim());
      DALI_ENFORCE(base_input.type() == input.type());
      DALI_ENFORCE(base_input.GetLayout() == input.GetLayout());
      DALI_ENFORCE(base_input.is_pinned() == input.is_pinned());
      DALI_ENFORCE(base_input.order() == input.order());
      DALI_ENFORCE(base_input.device_id() == input.device_id());
    }
  }
  const auto &predicate = ws.ArgumentInput("predicate");
  DALI_ENFORCE(
      input_sample_count_ == predicate.num_samples(),
      make_string("Merge description must cover whole input, got ", input_sample_count_,
                  " input samples and ", predicate.num_samples(), " elements denoting the merge."));
  for (int i = 0; i < predicate.num_samples(); i++) {
    DALI_ENFORCE(predicate[i].shape() == TensorShape<0>(), "Only scalar indexing is supported.");
  }


  // TODO(klecki): we can construct the output_desc, it won't be useful now
  return false;
}

template <typename Backend>
void Merge<Backend>::RunImpl(workspace_t<Backend> &ws) {
  auto &output = ws.template Output<Backend>(0);
  const auto &predicate = ws.ArgumentInput("predicate");
  auto category_input_idx = uniform_array<kMaxCategories>(0);

  for (int input_category = 0; input_category < kMaxCategories; input_category++) {
    const auto &input = ws.template Input<Backend>(input_category);

    // We can (and need to) do it only once, for each new output instance, when it doesn't have
    // data yet. It should be consistent across iterations.
    if (input.num_samples() > 0 && !output.has_data()) {
      output.SetupLike(input);
    }
  }
  output.SetSize(input_sample_count_);

  for (int output_idx = 0; output_idx < predicate.num_samples(); output_idx++) {
    int input_category = *predicate.template tensor<bool>(output_idx);
    auto &input = ws.template Input<Backend>(input_category);

    // get the index within input category and increment for the next occurrence.
    int input_idx = category_input_idx[input_category];
    category_input_idx[input_category]++;

    // share the sample to the output
    output.UnsafeSetSample(output_idx, input, input_idx);
  }
}

DALI_SCHEMA(Merge)
    .DocStr(R"code(Merge batch based on a predicate.)code")
    .NumInput(2)
    .NumOutput(1)
    .PassThrough({{0, 0}})  //todo add special pass through
    .AddArg("predicate", "Boolean categorization of the inputs", DALI_BOOL, true)
    .MakeInternal();

DALI_REGISTER_OPERATOR(Merge, Merge<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Merge, Merge<GPUBackend>, GPU);

}  // namespace dali
