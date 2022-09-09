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

#include "dali/core/common.h"
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
    if (nonzero_input_idx < 0 && input.num_samples() > 0) {
      nonzero_input_idx = input_category;
    }
    if (nonzero_input_idx >= 0) {
      const auto &base_input = ws.template Input<Backend>(nonzero_input_idx);
      // TODO(klecki): Error messages. BTW, we can just let it explode in Run, TV already makes
      // sure that those are ok
      DALI_ENFORCE(base_input.shape().sample_dim() == input.shape().sample_dim());
      DALI_ENFORCE(base_input.type() == input.type());
      DALI_ENFORCE(base_input.GetLayout() == input.GetLayout());
      // We allow to mix pinned and not pinned memory, defaulting to the non-pinned in that case
      // DALI_ENFORCE(base_input.is_pinned() == input.is_pinned(),
      //   make_string("Pinned ", base_input.is_pinned(), " vs ", input.is_pinned()));
      DALI_ENFORCE(
          base_input.order() == input.order(),
          make_string("Order ", base_input.order().device_id(), " ", base_input.order().stream(),
                      " vs ", input.order().device_id(), " ", input.order().stream()));
      if (base_input.is_pinned() == input.is_pinned()) {
        DALI_ENFORCE(base_input.device_id() == input.device_id(),
                     make_string("Device id: ", base_input.device_id(), " vs ", input.device_id()));
      }
    }
  }

  // We can have inputs of different pinnedness coming in from DALI
  // In theory we can ensure that if one is pinned, than all should be, and propagate that
  // information in graph building stage (as we prepin outputs and argument inputs).
  // For that we need nicer graph analysis, as pinnedness is also not propagated back through
  // pass through operators correctly (if we want argument input pinned, but it's produced by pass
  // through, the origin for the buffer won't be pinned).
  // TODO(klecki): Remove the pinned madness, and let executor unify this.
  pinned_ = true;
  for (int input_category = 0; input_category < kMaxCategories; input_category++) {
    const auto &input = ws.template Input<Backend>(input_category);
    if (input.num_samples() > 0)
      pinned_ = pinned_ && input.is_pinned();
    if (!pinned_) {
      break;
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

  // We propagate views only, so just don't care about what is here and reset, to have
  // some simpler pinned handling
  output.Reset();
  for (int input_category = 0; input_category < kMaxCategories; input_category++) {
    const auto &input = ws.template Input<Backend>(input_category);

    // We can (and need to) do it only once, for each new output instance, when it doesn't have
    // data yet. It should be consistent across iterations.
    if (input.num_samples() > 0) {
      output.SetupLike(input);
    }
  }
  if (pinned_ != output.is_pinned()) {
    output.set_pinned(false);
    if (std::is_same_v<CPUBackend, Backend>)
      output.set_device_id(CPU_ONLY_DEVICE_ID);
  }


  output.SetSize(input_sample_count_);

  for (int output_idx = 0; output_idx < predicate.num_samples(); output_idx++) {
    int input_category = *predicate.template tensor<bool>(output_idx);
    auto &input = ws.template Input<Backend>(input_category);

    // get the index within input category and increment for the next occurrence.
    int input_idx = category_input_idx[input_category];
    category_input_idx[input_category]++;

    // share the sample to the output
    if (input.is_pinned() == output.is_pinned()) {
      output.SetSample(output_idx, input, input_idx);
    } else {
      assert(!output.is_pinned() && "We only allow to downgrade to non-pinned");
      // TODO(klecki): This branch is super-ugly WAR for the fact that we don't have
      // nice way of making Tensor forget that it is pinned.
      // Degrading that attribute should be possible in theory.
      Tensor<Backend> tmp_sample;
      tmp_sample.set_backing_allocation(
          unsafe_sample_owner(const_cast<TensorList<Backend> &>(input), input_idx),
          volume(input.shape().tensor_shape_span(input_idx)) * output.type_info().size(),
          output.is_pinned(), input.type(), volume(input.shape().tensor_shape_span(input_idx)),
          output.device_id(), input.order());
      tmp_sample.Resize(input.shape()[input_idx]);
    }
  }
}

DALI_SCHEMA(Merge)
    .DocStr(R"code(Merge batch based on a predicate.)code")
    .NumInput(2)
    .NumOutput(1)
    .SamplewisePassThrough()
    .AddArg("predicate", "Boolean categorization of the inputs", DALI_BOOL, true)
    .MakeInternal();

DALI_REGISTER_OPERATOR(Merge, Merge<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Merge, Merge<GPUBackend>, GPU);

}  // namespace dali
