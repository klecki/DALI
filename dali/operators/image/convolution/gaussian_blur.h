// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_H_
#define DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_H_


// #include "dali/core/common.h"
// #include "dali/core/error_handling.h"
// #include "dali/core/tensor_shape.h"
// #include "dali/kernels/scratch.h"

#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/imgproc/convolution/convolution_cpu.h"
#include "dali/kernels/imgproc/convolution/gaussian_blur.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

namespace dali {

// using Kernel = dali::kernels::GaussianBlur<uint8_t, uint8_t, float, 2, true>;

template <typename Backend>
class GaussianBlur : public Operator<Backend> {
 public:
  inline explicit GaussianBlur(const OpSpec& spec) : Operator<Backend>(spec) {
    // if (spec.HasArgument("new_order")) {
    //   new_order_ = spec.GetRepeatedArgument<int>("new_order");
    //   DALI_ENFORCE(!new_order_.empty(), "Empty result sequences are not allowed.");
    //   single_order_ = true;
    // }
  }

  DISABLE_COPY_MOVE_ASSIGN(GaussianBlur);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<Backend>& ws) override {
    const auto& input = ws.template InputRef<Backend>(0);
    auto layout = input.GetLayout();
    auto dim_desc = ParseAndValidateDim(input.shape().sample_dim(), layout);
    // kmgrs_.resize(dim_desc.usable_dim_count);

    int nsamples = input.size();
    auto nthreads = ws.GetThreadPool().size();

    // for (auto &kmgr : kmgrs_) {
    //   // kmgr.template Resize<kernels::SeparableConvolution<float, uint8_t, float, 3, true>>(nthreads, nsamples);
    // }
    using Kernel = kernels::GaussianBlur<uint8_t, uint8_t, float, 2, true>;
    kmgr_.template Initialize<Kernel>();
    kmgr_.template Resize<Kernel>(nthreads, nsamples);

    // auto req = kmgr_.Setup<Kernel>(thread_id, i, ctx, out_view, in_view, args_[i]);


    output_desc.resize(1);
    output_desc[0].type = TypeTable::GetTypeInfo(TypeTable::GetTypeID<uint8_t>());
    output_desc[0].shape.resize(nsamples, 3);


    auto sigma = spec_.GetArgument<float>("sigma");
    for (int i = 0; i < nsamples; i++) {
      const auto in_view = view<const uint8_t, 3>(input[i]);
      auto &req = kmgr_.Setup<Kernel>(i, ctx_, in_view, sigma, sigma);
      output_desc[0].shape.set_tensor_shape(i, req.output_shapes[0][0].shape);
    }



    // TYPE_SWITCH(input.type().id(), type2id, T, ERASE_SUPPORTED_TYPES, (
    //   VALUE_SWITCH(in_shape.sample_dim(), Dims, ERASE_SUPPORTED_NDIMS, (
    //     impl_ = std::make_unique<EraseImplCpu<T, Dims>>(spec_);
    //   ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
    // ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT

    // layout.
    // layout.
    // const auto& in_shape = input.shape();  // temporary in some cases
    // DALI_ENFORCE(in_shape.sample_dim() > 1, "Sequence elements must have at least 1 dim");
    // output_desc.resize(1);
    // output_desc[0].type = input.type();
    // output_desc[0].shape = TensorListShape<>(in_shape.num_samples(), in_shape.sample_dim());
    // if (single_order_) {
    //   TensorView<StorageCPU, const int, 1> new_order(new_order_.data(),
    //                                                  TensorShape<1>(new_order_.size()));
    //   for (int i = 0; i < batch_size_; i++) {
    //     ValidateSeqRearrange(in_shape[i], new_order, i);
    //     output_desc[0].shape.set_tensor_shape(i, GetSeqRearrangedShape(in_shape[i], new_order));
    //   }
    // } else {
    //   const auto& new_orders = ws.ArgumentInput("new_order");
    //   for (int i = 0; i < batch_size_; i++) {
    //     auto new_order = view<const int, 1>(new_orders[i]);
    //     ValidateSeqRearrange(in_shape[i], new_order, i);
    //     output_desc[0].shape.set_tensor_shape(i, GetSeqRearrangedShape(in_shape[i], new_order));
    //   }
    // }

    // auto layout = input.GetLayout();
    // DALI_ENFORCE(layout.empty() || layout.find('F') == 0,
    //              make_string("Expected sequence as the input, where outermost dimension represents "
    //                          "frames dimension `F`, got data with layout = \"",
    //                          layout, "\"."));

    return true;
  }

  void RunImpl(workspace_t<Backend>& ws) override {

    const auto &input = ws.template InputRef<CPUBackend>(0);
    auto &output = ws.template OutputRef<CPUBackend>(0);
    auto in_shape = input.shape();
    auto& thread_pool = ws.GetThreadPool();
    using Kernel = kernels::GaussianBlur<uint8_t, uint8_t, float, 2, true>;
    auto sigma = spec_.GetArgument<float>("sigma");

    // TYPE_SWITCH(input.type().id(), type2id, T, MEL_FBANK_SUPPORTED_TYPES, (
    //   VALUE_SWITCH(in_shape.sample_dim(), Dims, MEL_FBANK_SUPPORTED_NDIMS, (
    //     using MelFilterBankKernel = kernels::audio::MelFilterBankCpu<T, Dims>;
        for (int i = 0; i < input.shape().num_samples(); i++) {
          thread_pool.DoWorkWithID(
            [this, &input, &output, i, sigma](int thread_id) {
              auto in_view = view<const uint8_t, 3>(input[i]);
              auto out_view = view<uint8_t, 3>(output[i]);
              kmgr_.Run<Kernel>(thread_id, i, ctx_, out_view, in_view, sigma, sigma);
            });
        }
    //   ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
    // ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT

    thread_pool.WaitForWork();
  }



 private:

  struct DimDesc {
    int usable_dim_start;
    int usable_dim_count;
    bool has_channels;
    bool is_sequence;
  };

  DimDesc ParseAndValidateDim(int ndim, TensorLayout layout) {
  static constexpr int kMaxDim = 3;
    if (layout.empty()) {
      // assuming plain data with no channels
      DALI_ENFORCE(ndim <= kMaxDim, make_string("Input data with empty layout cannot have more than ", kMaxDim, " dimensions, got input with ", ndim, " dimensions."));
      return {0, ndim, false, false};
    }
    // not-empty layout
    int dim_start = 0;
    int dim_count = ndim;
    bool has_channels = ImageLayoutInfo::HasChannel(layout);
    if (has_channels) {
      dim_count--;
      DALI_ENFORCE(ImageLayoutInfo::IsChannelLast(layout), "Only input data with no channels or channel-last is supported.");
    }
    bool is_sequence = layout.find('F') >= 0;
    if (is_sequence) {
      dim_start++;
      dim_count--;
      DALI_ENFORCE(layout.find('F') == 0, make_string("For sequence inputs frames 'F' should be the first dimension, got layout: \"", layout.str(), "\"."));
    }
    DALI_ENFORCE(dim_count <= kMaxDim, "Too many dimensions");
    return {dim_start, dim_count, has_channels, is_sequence};
  }

  USE_OPERATOR_MEMBERS();


  std::unique_ptr<OpImplBase<Backend>> impl_;

  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;
  // std::vector<kernels::KernelManager> kmgrs_;
  // bool single_order_ = false;
  // std::vector<int> new_order_;
  // static constexpr int kMaxDim = 3; /// wth linker, pls?
};


}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_H_