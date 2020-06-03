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

#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/convolution/gaussian_blur_cpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

// using Kernel = dali::kernels::GaussianBlur<uint8_t, uint8_t, float, 2, true>;

struct DimDesc {
  int usable_dim_start;
  int usable_dim_count;
  bool has_channels;
  bool is_sequence;
};

struct GaussianSampleParams {
  GaussianSampleParams() = default;
  GaussianSampleParams(int axes) {
    window_sizes.resize(axes);
    sigmas.resize(axes);
  }

  template <int N>
  std::array<float, N> GetSigmas() {
    std::array<float, N> result;
    for (int i = 0; i < N; i++) {
      result[i] = sigmas[i];
    }
    return result;
  }

  template <int N>
  std::array<int, N> GetWindowSizes() {
    std::array<int, N> result;
    for (int i = 0; i < N; i++) {
      result[i] = window_sizes[i];
    }
    return result;
  }

  SmallVector<int, 3> window_sizes;
  SmallVector<float, 3> sigmas;
};

DimDesc ParseAndValidateDim(int ndim, TensorLayout layout) {
  static constexpr int kMaxDim = 3;
  if (layout.empty()) {
    // assuming plain data with no channels
    DALI_ENFORCE(ndim <= kMaxDim,
                 make_string("Input data with empty layout cannot have more than ", kMaxDim,
                             " dimensions, got input with ", ndim, " dimensions."));
    return {0, ndim, false, false};
  }
  // not-empty layout
  int dim_start = 0;
  int dim_count = ndim;
  bool has_channels = ImageLayoutInfo::HasChannel(layout);
  if (has_channels) {
    dim_count--;
    DALI_ENFORCE(ImageLayoutInfo::IsChannelLast(layout),
                 "Only input data with no channels or channel-last is supported.");
  }
  bool is_sequence = layout.find('F') >= 0;
  if (is_sequence) {
    dim_start++;
    dim_count--;
    DALI_ENFORCE(
        layout.find('F') == 0,
        make_string("For sequence inputs frames 'F' should be the first dimension, got layout: \"",
                    layout.str(), "\"."));
  }
  DALI_ENFORCE(dim_count <= kMaxDim, "Too many dimensions");
  return {dim_start, dim_count, has_channels, is_sequence};
}

#define GAUSSIAN_BLUR_SUPPORTED_TYPES \
  (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, float16)

#define GAUSSIAN_BLUR_SUPPORTED_AXES (1, 2, 3)

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
    dim_desc_ = ParseAndValidateDim(input.shape().sample_dim(), layout);
    int nsamples = input.size();
    auto nthreads = ws.GetThreadPool().size();
    output_desc.resize(1);
    output_desc[0].type = TypeTable::GetTypeInfo(TypeTable::GetTypeID<uint8_t>());
    output_desc[0].shape.resize(nsamples, input.shape().sample_dim());
    params_.resize(nsamples);

    // clang-format off
    TYPE_SWITCH(input.type().id(), type2id, T, GAUSSIAN_BLUR_SUPPORTED_TYPES, (
      VALUE_SWITCH(dim_desc_.usable_dim_count, AXES, GAUSSIAN_BLUR_SUPPORTED_AXES, (
        // To WAR the switch over bool
        VALUE_SWITCH(static_cast<int>(dim_desc_.has_channels), HAS_CHANNELS, (0, 1), (
          constexpr bool has_channels = HAS_CHANNELS;
          constexpr int ndim = AXES + HAS_CHANNELS;
          using Kernel = kernels::GaussianBlurCpu<T, T, float, ndim, has_channels>;
          kmgr_.template Initialize<Kernel>();
          kmgr_.template Resize<Kernel>(nthreads, nsamples);

          for (int i = 0; i < nsamples; i++) {
            params_[i] = GetSampleParams(AXES, i, spec_, ws);
            auto elem_shape = input[i].shape().template last<ndim>();
            // For sequence it will hold the first element, the rest will have the same shapes
            const auto in_view = TensorView<StorageCPU, const T, ndim>{input[i].template data<T>(), elem_shape};
            auto& req = kmgr_.Setup<Kernel>(i, ctx_, in_view, params_[i].template GetWindowSizes<AXES>());
            // The shape of data stays untouched
            output_desc[0].shape.set_tensor_shape(i, input[i].shape());
          }

        ), (DALI_FAIL("Got value different than {0, 1} when converting bool to int.")));
      ), DALI_FAIL(""));
    ),
      DALI_FAIL(make_string("Unsupported data type: ", input.type().id()))
    );
    // clang-format on

    return true;
  }

  void RunImpl(workspace_t<Backend>& ws) override {
    const auto& input = ws.template InputRef<CPUBackend>(0);
    auto& output = ws.template OutputRef<CPUBackend>(0);
    auto in_shape = input.shape();
    auto& thread_pool = ws.GetThreadPool();

    // clang-format off
    TYPE_SWITCH(input.type().id(), type2id, T, GAUSSIAN_BLUR_SUPPORTED_TYPES, (
      VALUE_SWITCH(dim_desc_.usable_dim_count, AXES, GAUSSIAN_BLUR_SUPPORTED_AXES, (
        // To WAR the switch over bool
        VALUE_SWITCH(static_cast<int>(dim_desc_.has_channels), HAS_CHANNELS, (0, 1), (
          constexpr bool has_channels = HAS_CHANNELS;
          constexpr int ndim = AXES + HAS_CHANNELS;
          using Kernel = kernels::GaussianBlurCpu<T, T, float, ndim, has_channels>;

          for (int i = 0; i < input.shape().num_samples(); i++) {
            thread_pool.DoWorkWithID([this, &input, &output, i](int thread_id) {
              auto elem_shape = input[i].shape().template last<ndim>();
              auto in_view = TensorView<StorageCPU, const T, ndim>{input[i].template data<T>(), elem_shape};
              auto out_view = TensorView<StorageCPU, T, ndim>{output[i].template mutable_data<T>(), elem_shape};
              int64_t stride = 0;
              int seq_elements = 1;
              if (dim_desc_.is_sequence) {
                seq_elements = input[i].shape()[0];
                stride = volume(elem_shape);
              }
              auto window_sizes = params_[i].template GetWindowSizes<AXES>();
              auto sigmas = params_[i].template GetSigmas<AXES>();
              for (int elem_idx = 0; elem_idx < seq_elements; elem_idx++) {
                kmgr_.Run<Kernel>(thread_id, i, ctx_, out_view, in_view, window_sizes, sigmas);
                in_view.data += stride;
                out_view.data += stride;
              }
            });
          }

        ), (DALI_FAIL("Got value different than {0, 1} when converting bool to int.")));
      ), DALI_FAIL(""));
    ),
      DALI_FAIL(make_string("Unsupported data type: ", input.type().id()))
    );
    // clang-format on


    thread_pool.WaitForWork();
  }

 private:
  float GetSigma(int dim, int sample, const OpSpec& spec, const ArgumentWorkspace& ws) {
    if (spec.ArgumentDefined(kSigmaPerAxisArgNames[dim])) {
      return spec.GetArgument<float>(kSigmaPerAxisArgNames[dim], &ws, sample);
    } else {
      return spec.GetArgument<float>(kSigmaArgName, &ws, sample);
    }
  }

  int GetWindowSize(int dim, int sample, const OpSpec& spec, const ArgumentWorkspace& ws) {
    if (spec.ArgumentDefined(kWindowSizePerAxisArgNames[dim])) {
      return spec.GetArgument<float>(kWindowSizePerAxisArgNames[dim], &ws, sample);
    } else {
      return spec.GetArgument<float>(kWindowSizeArgName, &ws, sample);
    }
  }

  GaussianSampleParams GetSampleParams(int axes, int sample, const OpSpec& spec,
                                       const ArgumentWorkspace& ws) {
    GaussianSampleParams params(axes);
    for (int i = 0; i < axes; i++) {
      params.sigmas[i] = GetSigma(i, sample, spec, ws);
      params.window_sizes[i] = GetWindowSize(i, sample, spec, ws);
      DALI_ENFORCE(
          !(params.sigmas[i] == 0 && params.window_sizes[i] == 0),
          make_string("`sigma` and `window_size` shouldn't be 0 at the same time for sample ",
                      sample, "."));
      DALI_ENFORCE(params.sigmas[i] >= 0,
                   make_string("`sigma` must have non-negative values, got ", params.sigmas[i],
                               " for sample: ", sample, ", axis: ", i, "."));
      DALI_ENFORCE(params.window_sizes[i] >= 0,
                   make_string("`window_size` must have non-negative values, got ",
                               params.sigmas[i], " for sample: ", sample, ", axis : ", i, "."));
      if (params.window_sizes[i] == 0 && params.sigmas[i] > 0.f) {
        params.window_sizes[i] = 2 * ceilf(params.sigmas[i] * 3) + 1;
      }
    }
    return params;
  }

  USE_OPERATOR_MEMBERS();

  std::unique_ptr<OpImplBase<Backend>> impl_;

  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;
  DimDesc dim_desc_;
  constexpr static const char* kSigmaArgName = "sigma";
  constexpr static const char* kSigmaPerAxisArgNames[] = {"sigma_0", "sigma_1", "sigma_2"};

  constexpr static const char* kWindowSizeArgName = "window_size";
  constexpr static const char* kWindowSizePerAxisArgNames[] = {"window_size_0", "window_size_1",
                                                               "window_size_2"};

  std::vector<GaussianSampleParams> params_;

  // std::vector<kernels::KernelManager> kmgrs_;
  // bool single_order_ = false;
  // std::vector<int> new_order_;
  // static constexpr int kMaxDim = 3; /// wth linker, pls?
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_H_