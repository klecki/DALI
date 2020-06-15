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

#include "dali/operators/image/convolution/gaussian_blur.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_cpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/convolution/gaussian_blur_params.h"
#include "dali/pipeline/data/views.h"

namespace dali {

constexpr static const char* kSigmaArgName = "sigma";
constexpr static const char* kSigmaPerAxisArgNames[] = {"sigma_0", "sigma_1", "sigma_2"};

constexpr static const char* kWindowSizeArgName = "window_size";
constexpr static const char* kWindowSizePerAxisArgNames[] = {"window_size_0", "window_size_1",
                                                             "window_size_2"};

DALI_SCHEMA(GaussianBlur)
    .DocStr(R"code(Apply Gaussian Blur to the input.
Separable convolution with Gaussian Kernel is used to calculate the output.

User can specify the sigma or kernel window size.
If only the sigma is provided, the radius is of kernel is calculated ``ceil(3 * sigma)``,
thus the kernel window size is ``2 * ceil(3 * sigma) + 1``.

If only the kernel window size is provided, the sigma is calculated using the following formula::

  radius = (window_size - 1) / 2
  sigma = (radius - 1) * 0.3 + 0.8

Both sigma and kernel window size can be specified as single value for all data axes
or per data axis.

When specifying the sigma or window size per axis, index 0 represents the outermost data axis.
The channel ``C`` and frame ``F`` dimensions are not considered data axes.

For example, with ``HWC`` input, user can provide ``sigma=1.0`` or ``sigma=(1.0, 2.0)`` as there
are two data axes H and W.

The same input can be provided as per-sample tensors.
)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .AddOptionalArg(kWindowSizeArgName, "The diameter of kernel.", std::vector<int>{0}, true)
    // .AddOptionalArg<int>(kWindowSizePerAxisArgNames[0],
    //                      "The diameter of kernel window in first, outermost axis 0.", nullptr,
    //                      true)
    // .AddOptionalArg<int>(kWindowSizePerAxisArgNames[1], "The diameter of kernel window for axis 1.",
    //                      nullptr, true)
    // .AddOptionalArg<int>(kWindowSizePerAxisArgNames[2], "The diameter of kernel window for axis 2.",
                        //  nullptr, true)
    .AddOptionalArg<float>(kSigmaArgName, R"code(Sigma value for Gaussian Kernel.)code", std::vector<float>{0.f}, true);
    // .AddOptionalArg<float>(kSigmaPerAxisArgNames[0],
    //                        R"code(Sigma value for Gaussian Kernel, for axis 0.)code", nullptr, true)
    // .AddOptionalArg<float>(kSigmaPerAxisArgNames[1],
    //                        R"code(Sigma value for Gaussian Kernel, for axis 1.)code", nullptr, true)
    // .AddOptionalArg<float>(kSigmaPerAxisArgNames[2],
    //                        R"code(Sigma value for Gaussian Kernel, for axis 2.)code", nullptr,
    //                        true);

float GetSigma(int dim, int dims, int sample, const OpSpec& spec, const ArgumentWorkspace& ws) {
  if (spec.HasTensorArgument(kSigmaArgName)) {

  }
  // auto spec.GetRepeatedArgument<float>(kSigmaArgName);
  if (spec.ArgumentDefined(kSigmaPerAxisArgNames[dim])) {
    return spec.GetArgument<float>(kSigmaPerAxisArgNames[dim], &ws, sample);
  } else {
    return spec.GetArgument<float>(kSigmaArgName, &ws, sample);
  }
}

int GetWindowSize(int dim, int dims, int sample, const OpSpec& spec, const ArgumentWorkspace& ws) {
  if (spec.ArgumentDefined(kWindowSizePerAxisArgNames[dim])) {
    return spec.GetArgument<int>(kWindowSizePerAxisArgNames[dim], &ws, sample);
  } else {
    return spec.GetArgument<int>(kWindowSizeArgName, &ws, sample);
  }
}

template <int axes>
GaussianSampleParams<axes> GetSampleParams(int sample, const OpSpec& spec,
                                           const ArgumentWorkspace& ws) {
  GaussianSampleParams<axes> params;
  for (int i = 0; i < axes; i++) {
    params.sigmas[i] = GetSigma(i, axes, sample, spec, ws);
    params.window_sizes[i] = GetWindowSize(i, axes, sample, spec, ws);
    DALI_ENFORCE(
        !(params.sigmas[i] == 0 && params.window_sizes[i] == 0),
        make_string("`sigma` and `window_size` shouldn't be 0 at the same time for sample ", sample,
                    "."));
    DALI_ENFORCE(params.sigmas[i] >= 0,
                 make_string("`sigma` must have non-negative values, got ", params.sigmas[i],
                             " for sample: ", sample, ", axis: ", i, "."));
    DALI_ENFORCE(params.window_sizes[i] >= 0,
                 make_string("`window_size` must have non-negative values, got ", params.sigmas[i],
                             " for sample: ", sample, ", axis : ", i, "."));
    if (params.window_sizes[i] == 0 && params.sigmas[i] > 0.f) {
      params.window_sizes[i] = GaussianSigmaToDiameter(params.sigmas[i]);
    } else if (params.sigmas[i] == 0.f && params.window_sizes[i] > 0) {
      params.sigmas[i] = GaussianDiameterToSigma(params.window_sizes[i]);
    }
  }
  return params;
}

GaussianDimDesc ParseAndValidateDim(int ndim, TensorLayout layout) {
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

// ndim here is dimension of element processed by kernel - in case of sequence it's 1 less than the
// actual dim
template <typename T, int ndim, bool has_channels>
class GaussianBlurOpCpu : public OpImplBase<CPUBackend> {
 public:
  using Kernel = kernels::SeparableConvolutionCpu<T, T, float, ndim, has_channels>;
  static constexpr int axes = ndim - has_channels;

  explicit GaussianBlurOpCpu(const OpSpec& spec, const GaussianDimDesc& dim_desc)
      : spec_(spec), batch_size_(spec.GetArgument<int>("batch_size")), dim_desc_(dim_desc) {}

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<CPUBackend>& ws) override {
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

  void RunImpl(workspace_t<CPUBackend>& ws) override {
    const auto& input = ws.template InputRef<CPUBackend>(0);
    auto& output = ws.template OutputRef<CPUBackend>(0);
    auto in_shape = input.shape();
    auto& thread_pool = ws.GetThreadPool();

    for (int i = 0; i < input.shape().num_samples(); i++) {
      thread_pool.DoWorkWithID([this, &input, &output, i](int thread_id) {
        auto gaussian_windows = windows_[i].GetWindows(params_[i]);
        auto elem_shape = input[i].shape().template last<ndim>();
        auto in_view =
            TensorView<StorageCPU, const T, ndim>{input[i].template data<T>(), elem_shape};
        auto out_view =
            TensorView<StorageCPU, T, ndim>{output[i].template mutable_data<T>(), elem_shape};
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

  GaussianDimDesc dim_desc_;
  std::vector<GaussianSampleParams<axes>> params_;
  std::vector<GaussianWindows<axes>> windows_;
};

template <>
bool GaussianBlur<CPUBackend>::SetupImpl(std::vector<OutputDesc>& output_desc,
                                         const workspace_t<CPUBackend>& ws) {
  const auto& input = ws.template InputRef<CPUBackend>(0);
  auto layout = input.GetLayout();
  auto dim_desc = ParseAndValidateDim(input.shape().sample_dim(), layout);

  // clang-format off
  TYPE_SWITCH(input.type().id(), type2id, T, GAUSSIAN_BLUR_SUPPORTED_TYPES, (
    VALUE_SWITCH(dim_desc.usable_dim_count, AXES, GAUSSIAN_BLUR_SUPPORTED_AXES, (
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

template <>
void GaussianBlur<CPUBackend>::RunImpl(workspace_t<CPUBackend>& ws) {
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(GaussianBlur, GaussianBlur<CPUBackend>, CPU);

}  // namespace dali
