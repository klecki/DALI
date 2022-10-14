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
#include "dali/core/util.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/builtin/merge.h"
#include "dali/pipeline/operator/builtin/split_merge.h"

namespace dali {


template <typename Backend>
bool Merge<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                               const workspace_t<Backend> &ws) {
  input_sample_count_ = 0;
  int nonzero_input_sample_idx = -1;
  for (int input_category_idx = 0; input_category_idx < kMaxCategories; input_category_idx++) {
    const auto &input = ws.template Input<Backend>(input_category_idx);
    input_sample_count_ += input.num_samples();
    std::cout << "[Merge]: category: " << input_category_idx << ", size: " << input.num_samples()
              << std::endl;
    // TODO(klecki): do not compare against empty inputs unless we ensure consistent run behaviour
    // for empty samples.
    if (nonzero_input_sample_idx < 0 && input.num_samples() > 0) {
      nonzero_input_sample_idx = input_category_idx;
      continue;  // no point in comparing with ourselves
    }
    if (nonzero_input_sample_idx >= 0 && input.num_samples() > 0) {
      const auto &base_input = ws.template Input<Backend>(nonzero_input_sample_idx);
      // TODO(klecki): Error messages. BTW, we can just let it explode in Run, TV already makes
      // sure that those are ok
      DALI_ENFORCE(base_input.shape().sample_dim() == input.shape().sample_dim(),
                   make_string("Sample dim wrong ", base_input.shape().sample_dim(), " ",
                               input.shape().sample_dim()));
      DALI_ENFORCE(base_input.type() == input.type());
      DALI_ENFORCE(base_input.GetLayout() == input.GetLayout());
      // When not pinned, we can have device_id = CPU_ONLY_DEVICE_ID, and for pinned it is the id
      // of an actual device.
      if (base_input.is_pinned() == input.is_pinned()) {
        DALI_ENFORCE(base_input.device_id() == input.device_id(),
                     make_string("Device id: ", base_input.device_id(), " vs ", input.device_id()));
      }
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
  return false;
}


template <typename Backend>
void Merge<Backend>::RunImpl(workspace_t<Backend> &ws) {
  auto &output = ws.template Output<Backend>(0);
  const auto &predicate = ws.ArgumentInput("predicate");
  auto sample_idx_in_input = uniform_array<kMaxCategories>(0);

  if (!pinned_) {
    // We produce pinned data if the executor said so, or we got any pinned input.
    pinned_ = output.is_pinned();
    //  || [&ws]() {
    //   for (int input_category = 0; input_category < kMaxCategories; input_category++) {
    //     const auto &input = ws.template Input<Backend>(input_category);
    //     if (input.is_pinned()) {
    //       return true;
    //     }
    //   }
    //   return false;
    // }();
  }

  // TODO(klecki): no longer necessary, as we allowed it to be synchronized in SetSample
  if (!order_) {
    order_ = output.order();
    if (ws.has_stream()) {
      assert(order_->get() == ws.stream() && "We use the order of current stage");
    } else {
      assert(order_ == AccessOrder::host() && "We use host order in CPU stage");
    }
  }

  // We propagate views only, so just don't care about what is here and reset, to have
  // some simpler pinned handling
  output.Reset();
  for (int input_category = 0; input_category < kMaxCategories; input_category++) {
    const auto &input = ws.template Input<Backend>(input_category);
    if (input.num_samples() > 0) {
      output.set_type(input.type());
      output.set_sample_dim(input.shape().sample_dim());
      output.SetLayout(input.GetLayout());
      // The pinned and order can differ depending on the pipeline graph. Let the executor
      // set the desired one, and we will copy if we don't match.
      // TODO(klecki): device_id when pinned or not is broken
      if (input.order() == *order_ && input.is_pinned() == *pinned_) {
        output.set_device_id(input.device_id());
        output.set_pinned(*pinned_);
      }
    }
  }

  output.SetSize(input_sample_count_);

  for (int output_sample_idx = 0; output_sample_idx < predicate.num_samples();
       output_sample_idx++) {
    int input_category_idx = get_category_index(predicate, output_sample_idx);
    auto &input = ws.template Input<Backend>(input_category_idx);

    // get the index within input category and increment for the next occurrence.
    int input_sample_idx = sample_idx_in_input[input_category_idx];
    sample_idx_in_input[input_category_idx]++;

    if (input.order() == *order_ && input.is_pinned() == *pinned_) {
      // share the sample to the output
      output.SetSample(output_sample_idx, input, input_sample_idx);
      // The commented out code might be unsafe - if we downgrade the pinned memory, and do a async
      // copy from it, it will have different behaviour than non-pinned memory.
      // } else if (std::is_same_v<CPUBackend, Backend> && *pinned_ == false &&
      //            input.is_pinned() == true) {
      //   std::cout << "Downgrade to non-pinned" << std::endl;
      //   auto sample = unsafe_sample_owner(input, input_sample_idx);
      //   const auto &sample_shape = input.shape()[input_sample_idx];
      //   assert(output.device_id() == CPU_ONLY_DEVICE_ID && input.device_id() !=
      //   CPU_ONLY_DEVICE_ID &&
      //          "Looks like when we pin we get a valid device id for cpu memory");
      //   output.SetSample(output_sample_idx, sample, volume(sample_shape) *
      //   input.type_info().size(), false,
      //                    sample_shape, input.type(), output.device_id(), output.order(),
      //                    output.layout());
    } else {
      // Pessimistic variant, we need to copy.
      // Unless we get to know the queue indexing, we cannot easily make internal copy and share
      // it as it would break at the pipeline outputs - we would need to subscribe to the buffering
      // done by the executor, and ensure the same lifetime of internal tmp buffer.
      // TODO(klecki): Do one allocation, where samples that we share are 0-volumed - this might
      // be perf optimization reducing the number of allocations to 1.
      CopySampleToOutput(output, output_sample_idx, input, input_sample_idx, ws);
    }
  }
  FinalizeCopy(ws);
}


template <>
void Merge<CPUBackend>::CopySampleToOutput(TensorList<CPUBackend> &output, int output_sample_idx,
                                           const TensorList<CPUBackend> &input,
                                           int input_sample_idx, workspace_t<CPUBackend> &ws) {
  auto &tp = ws.GetThreadPool();
  tp.AddWork(
      [&output, &input, output_sample_idx, input_sample_idx](int thread_idx) {
        output.ResizeSample(output_sample_idx, input.shape()[input_sample_idx]);
        output.CopySample(output_sample_idx, input, input_sample_idx, output.order());
      },
      volume(input.tensor_shape_span(input_sample_idx)));
}


template <>
void Merge<GPUBackend>::CopySampleToOutput(TensorList<GPUBackend> &output, int output_sample_idx,
                                           const TensorList<GPUBackend> &input,
                                           int input_sample_idx, workspace_t<GPUBackend> &ws) {
  output.ResizeSample(output_sample_idx, input.shape()[input_sample_idx]);
  output.CopySample(output_sample_idx, input, input_sample_idx, ws.stream());
}


template <>
void Merge<CPUBackend>::FinalizeCopy(workspace_t<CPUBackend> &ws) {
  ws.GetThreadPool().RunAll();
}


template <>
void Merge<GPUBackend>::FinalizeCopy(workspace_t<GPUBackend> &ws) {}

DALI_SCHEMA(Merge)
    .DocStr(R"code(Merge batch based on a predicate.)code")
    .NumInput(2)
    .NumOutput(1)
    .SamplewisePassThrough()
    .AddArg("predicate", "Boolean categorization of the inputs", DALI_BOOL, true)
    .MakeInternal();

DALI_REGISTER_OPERATOR(Merge, Merge<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Merge, Merge<GPUBackend>, GPU);


DALI_SCHEMA(experimental___Merge)
    .DocStr(R"code(Merge batch based on a predicate.)code")
    .NumInput(2)
    .NumOutput(1)
    .SamplewisePassThrough()
    .AddArg("predicate", "Boolean categorization of the inputs", DALI_BOOL, true);

DALI_REGISTER_OPERATOR(experimental___Merge, Merge<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(experimental___Merge, Merge<GPUBackend>, GPU);

}  // namespace dali
