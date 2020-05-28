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
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_cpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {
struct DimDesc {
  int usable_dim_start;
  int usable_dim_count;
  bool has_channels;
  bool is_sequence;
};

template <int axes>
struct gaussian_sample_params {
  std::array<int, axes> window_sizes;
  std::array<float, axes> sigmas;
};

void FillGaussian(const TensorView<StorageCPU, float, 1>& window, float sigma) {
  int r = (window.num_elements() - 1) / 2;
  double exp_scale = 0.5 / (sigma * sigma);
  double mul_scale = 1.0 / sqrt(2 * M_PI * sigma * sigma); // TODO mul_scale actually disappears, can be removed
  double sum = 0.;
  // Calculate first half
  for (int x = -r; x < 0; x++) {
    *window(x + r) = mul_scale * exp(-(x * x * exp_scale));
    sum += *window(x + r);
  }
  // Total sum, it's symmetric with `mul_scale * 1` in the center.
  sum *= 2.;
  sum += mul_scale;
  double scale = 1. / sum;
  // place center, scaled element
  *window(r) = mul_scale * scale;
  // scale all elements so they sum up to 1, duplicate the second half
  for (int x = 0; x < r; x++) {
    *window(x) *= scale;
    *window(2 * r - x) = *window(x);
  }
}

template <int axes>
class GaussianWindows {
 public:
  GaussianWindows() {
    previous.sigmas = uniform_array<axes>(-1.f);
    previous.window_sizes = uniform_array<axes>(0);
  }

  // todo:clean this up
  std::array<TensorView<StorageCPU, const float, 1>, axes> GetWindows(const gaussian_sample_params<axes> &params) {
    std::array<TensorView<StorageCPU, const float, 1>, axes> result;
    size_t new_elements = 0;
    bool prev_sizes_equal = true, prev_sigmas_equal = true;
    for (int i = 0; i < axes; i++) {
      new_elements += params.window_sizes[i];
      prev_sizes_equal = prev_sizes_equal && (previous.window_sizes[i] == params.window_sizes[i]);
      prev_sigmas_equal = prev_sigmas_equal && (previous.sigmas[i] == params.sigmas[i]);
    }
    bool all_elements_equal = true;
    for (int i = 1; i < axes; i++) {
      if (params.sigmas[i - 1] != params.sigmas[i] || params.window_sizes[i - 1] != params.window_sizes[i]) {
        all_elements_equal = false;
      }
    }

    if (prev_sizes_equal && prev_sigmas_equal) {
      // fill and return the old result todo: do not use the same loop twice
      int offset = 0;
      for (int i = 0; i < axes; i++) {
        result[i] = {&memory[offset], {params.window_sizes[i]}};
        if (!all_elements_equal)
          offset += params.window_sizes[i];
      }
      return result;
    }
    // there were changes, todo: if all_sizes_equal we need one allocation
    if (new_elements > memory.size()) {
      memory.resize(new_elements);
    }
    // fill and return the new results
    int offset = 0;
    for (int i = 0; i < axes; i++) {
      TensorView<StorageCPU, float, 1> tmp_view;
      tmp_view = {&memory[offset], {params.window_sizes[i]}};
      if (!all_elements_equal || i == 0)
        FillGaussian(tmp_view, params.sigmas[i]);
      if (!all_elements_equal)
        offset += params.window_sizes[i];
      result[i] = {tmp_view.data, tmp_view.shape};
    }
    return result;
  }

 private:
  gaussian_sample_params<axes> previous;
  std::vector<float> memory;
};

constexpr static const char* kSigmaArgName = "sigma";
constexpr static const char* kSigmaPerAxisArgNames[] = {"sigma_0", "sigma_1", "sigma_2"};

constexpr static const char* kWindowSizeArgName = "window_size";
constexpr static const char* kWindowSizePerAxisArgNames[] = {"window_size_0", "window_size_1",
                                                              "window_size_2"};

// todo deduplicate
float GetSigma(int dim, int sample, const OpSpec& spec, const ArgumentWorkspace& ws) {
  if (spec.ArgumentDefined(kSigmaPerAxisArgNames[dim])) {
    return spec.GetArgument<float>(kSigmaPerAxisArgNames[dim], &ws, sample);
  } else {
    return spec.GetArgument<float>(kSigmaArgName, &ws, sample);
  }
}

int GetWindowSize(int dim, int sample, const OpSpec& spec, const ArgumentWorkspace& ws) {
  if (spec.ArgumentDefined(kWindowSizePerAxisArgNames[dim])) {
    return spec.GetArgument<int>(kWindowSizePerAxisArgNames[dim], &ws, sample);
  } else {
    return spec.GetArgument<int>(kWindowSizeArgName, &ws, sample);
  }
}

template <int axes>
gaussian_sample_params<axes> GetSampleParams(int sample, const OpSpec& spec, const ArgumentWorkspace& ws) {
  gaussian_sample_params<axes> params;
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
    } else if (params.sigmas[i] == 0.f && params.window_sizes[i] > 0) {
      // Based on OpenCV
      int r = (params.window_sizes[i] - 1) / 2;
      params.sigmas[i] = (r - 1) * 0.3 + 0.8;
    }
  }
  return params;
}


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

// ndim here is dimension of element processed by kernel - in case of sequence it's 1 less than the
// actual dim
template <typename T, int ndim, bool has_channels>
class GaussianBlurOpCpu : public OpImplBase<CPUBackend> {
 public:
  using Kernel = kernels::SeparableConvolutionCpu<T, T, float, ndim, has_channels>;
  static constexpr int axes = ndim - has_channels;

  explicit GaussianBlurOpCpu(const OpSpec& spec, const DimDesc &dim_desc)
      : spec_(spec), batch_size_(spec.GetArgument<int>("batch_size")), dim_desc_(dim_desc) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override {
    const auto& input = ws.template InputRef<CPUBackend>(0);
    int nsamples = input.size();
    auto nthreads = ws.GetThreadPool().size();

    output_desc.resize(1);
    output_desc[0].type = input.type();
    output_desc[0].shape.resize(nsamples, input.shape().sample_dim());

    params_.resize(nsamples);
    windows_.resize(nsamples);

    kmgr_.template Initialize<Kernel>();
    kmgr_.template Resize<Kernel>(nthreads, nsamples);

    for (int i = 0; i < nsamples; i++) {
      params_[i] = GetSampleParams<axes>(i, spec_, ws);
      // We take only last `ndim` siginificant dimensions to handle sequences as well
      auto elem_shape = input[i].shape().template last<ndim>();
      auto& req = kmgr_.Setup<Kernel>(i, ctx_, elem_shape, params_[i].window_sizes);
      // The shape of data stays untouched
      output_desc[0].shape.set_tensor_shape(i, input[i].shape());
    }
    return true;
  }

  void RunImpl(workspace_t<CPUBackend> &ws) override {

    const auto& input = ws.template InputRef<CPUBackend>(0);
    auto& output = ws.template OutputRef<CPUBackend>(0);
    auto in_shape = input.shape();
    auto& thread_pool = ws.GetThreadPool();

    for (int i = 0; i < input.shape().num_samples(); i++) {
      thread_pool.DoWorkWithID([this, &input, &output, i](int thread_id) {
        auto gaussian_windows = windows_[i].GetWindows(params_[i]);
        auto elem_shape = input[i].shape().template last<ndim>();
        auto in_view = TensorView<StorageCPU, const T, ndim>{input[i].template data<T>(), elem_shape};
        auto out_view = TensorView<StorageCPU, T, ndim>{output[i].template mutable_data<T>(), elem_shape};
        int64_t stride = 0;
        int seq_elements = 1;
        if (dim_desc_.is_sequence) {
          seq_elements = input[i].shape()[0];
          stride = volume(elem_shape);
        }

        for (int elem_idx = 0; elem_idx < seq_elements; elem_idx++) {
          kmgr_.Run<Kernel>(thread_id, i, ctx_, out_view, in_view, gaussian_windows);
          in_view.data += stride;
          out_view.data += stride;
        }
      });
    }
    thread_pool.WaitForWork();
  }

 private:
  OpSpec spec_;
  int batch_size_ = 0;

  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;

  DimDesc dim_desc_;
  std::vector<gaussian_sample_params<axes>> params_;
  std::vector<GaussianWindows<axes>> windows_;
};

template <typename Backend>
class GaussianBlur : public Operator<Backend> {
 public:
  inline explicit GaussianBlur(const OpSpec& spec) : Operator<Backend>(spec) {}

  DISABLE_COPY_MOVE_ASSIGN(GaussianBlur);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<Backend>& ws) override {
    const auto& input = ws.template InputRef<Backend>(0);
    auto layout = input.GetLayout();
    auto dim_desc = ParseAndValidateDim(input.shape().sample_dim(), layout);

    // clang-format off
    TYPE_SWITCH(input.type().id(), type2id, T, GAUSSIAN_BLUR_SUPPORTED_TYPES, (
      VALUE_SWITCH(dim_desc.usable_dim_count, AXES, GAUSSIAN_BLUR_SUPPORTED_AXES, (
        // To WAR the switch over bool
        VALUE_SWITCH(static_cast<int>(dim_desc.has_channels), HAS_CHANNELS, (0, 1), (
          constexpr bool has_channels = HAS_CHANNELS;
          constexpr int ndim = AXES + HAS_CHANNELS;
          impl_ = std::make_unique<GaussianBlurOpCpu<T, ndim, has_channels>>(spec_, dim_desc);
        ), (DALI_FAIL("Got value different than {0, 1} when converting bool to int.")));
      ), DALI_FAIL(""));
    ),
      DALI_FAIL(make_string("Unsupported data type: ", input.type().id()))
    );
    // clang-format on

    return impl_->SetupImpl(output_desc, ws);
  }

  void RunImpl(workspace_t<Backend>& ws) override {
    impl_->RunImpl(ws);
  }

 private:
  USE_OPERATOR_MEMBERS();
  std::unique_ptr<OpImplBase<Backend>> impl_;

};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_H_