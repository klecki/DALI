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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_GAUSSIAN_BLUR_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_GAUSSIAN_BLUR_H_

#include "dali/core/convert.h"
#include "dali/core/format.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/util/operator_impl_utils.h"
#include "dali/kernels/imgproc/convolution/convolution.h"

namespace dali {
namespace kernels {


// template <typename Out, typename In, typename W, bool has_channels = true, int... axes>
// struct GaussianBlur {

//   std::tuple<ConvolutionCpu<
// };


template <typename Out, typename In, typename W, int axes, bool has_channels = true>
struct GaussianBlur {
  // KernelRequirements Setup(KernelContext& ctx, const InTensorCPU<In, ndim>& in,
  //                          const TensorView<StorageCPU, const W, 1>& window) {
  //   KernelRequirements req;
  //   ScratchpadEstimator se;
  //   DALI_ENFORCE(
  //       window.num_elements() % 2 == 1,
  //       make_string("Kernel window should have odd length, got: ", window.num_elements(), "."));
  //   se.add<In>(AllocType::Host, GetInputWindowBufSize(in, window));
  //   se.add<In>(AllocType::Host, GetPixelSize(in));  // fill value
  //   se.add<W>(AllocType::Host, GetPixelSize(in));   // tmp result
  //   req.scratch_sizes = se.sizes;
  //   req.output_shapes.push_back(uniform_list_shape<ndim>(1, in.shape));
  //   return req;
  // }

  // void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> out,
  //          const TensorView<StorageCPU, const In, ndim>& in,
  //          const TensorView<StorageCPU, const W, 1>& window, int axis) {
  //   ValidateAxis(axis);
  //   int num_channels = GetPixelSize(in);
  //   int input_window_buf_size = GetInputWindowBufSize(in, window);
  //   auto* input_window_buffer =
  //       ctx.scratchpad->Allocate<In>(AllocType::Host, input_window_buf_size);
  //   auto* border_fill_buf = ctx.scratchpad->Allocate<In>(AllocType::Host, num_channels);
  //   auto* pixel_tmp_buf = ctx.scratchpad->Allocate<W>(AllocType::Host, num_channels);
  //   auto strides = GetStrides(in.shape);
  //   auto diameter = window.num_elements();

  //   auto border_fill = make_span(border_fill_buf, num_channels);
  //   for (int c = 0; c < num_channels; c++) {
  //     border_fill[c] = 0;
  //   }
  //   auto pixel_tmp = make_span(pixel_tmp_buf, num_channels);

  //   ConvolutionCpuCpuImpl<0, true, Out, In, W, ndim>(
  //       out.data, in.data, window.data, axis, in.shape, strides, diameter, 0, border_fill,
  //       input_window_buffer, pixel_tmp);
  // }

//  private:


  // void ValidateAxis(int axis) {
  //   int axis_max = has_channels ? ndim - 1 : ndim;
  //   DALI_ENFORCE(
  //       0 <= axis && axis < axis_max,
  //       make_string("Selected axis is not in range of possible axes of input data, got axis = ",
  //                   axis, ", expected axis in [0, ", axis_max,
  //                   "). Channels cannot be used as convolution axis."));
  // }

  // int GetInputWindowBufSize(const TensorView<StorageCPU, const In, ndim>& in,
  //                           const TensorView<StorageCPU, const W, 1>& window) {
  //   return GetPixelSize(in) * window.num_elements();
  // }
  // int GetPixelSize(const TensorView<StorageCPU, const In, ndim>& in) {
  //   return has_channels ? in.shape[ndim - 1] : 1;
  // }
};


// template <typename Out, typename In, typename W, bool has_channels>
// struct GaussianBlur<Out, In, W, 1, has_channels> {
//   static constexpr int effective_ndim = has_channels ? 2 : 1;

//   ConvolutionCpu<Out, In, W, effective_ndim, has_channels> conv_;
// };


int GetGaussianWindowDiameter(float sigma) {
  // Used by OpenCV: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
  // (sigma - 0.8) / 0.3 = (ksize - 1)*0.5 -1
  // r = (ksize - 1) / 2 = (sigma - 0.8) / 0.3 + 1 = sigma / 0.3 - 1.(6)
  //TODO(klecki): approx:
  int radius = ceilf(sigma * 3);
  int diameter = 1 + 2 * radius;
  return diameter;
}


void FillGaussian(const TensorView<StorageCPU, float, 1> &window, float sigma) {
  int r = (window.num_elements() - 1) / 2;
  for (int x = -r; x <= r; x++) {
    // directly using the formula, todo:optimize and calculate only half
    *window(x + r) = 1.0 / sqrt(2 * M_PI * sigma * sigma) * exp(-((x * x) / (2 * sigma * sigma)));
  }
}

template <typename Out, typename In, typename W, bool has_channels>
struct GaussianBlur<Out, In, W, 2, has_channels> {
  static constexpr int ndim = has_channels ? 3 : 2;

  KernelRequirements Setup(KernelContext& ctx, const InTensorCPU<In, ndim>& in, float sigma_inner, float sigma_outer) {
    KernelRequirements req;

    int diam_inner = GetGaussianWindowDiameter(sigma_inner);
    int diam_outer = GetGaussianWindowDiameter(sigma_outer);
    ScratchpadEstimator se;
    se.add<W>(AllocType::Host, diam_inner);
    se.add<W>(AllocType::Host, diam_outer);
    // If the output type is different than intermediate type we need some helper memory
    if (!std::is_same<Out, W>::value) {
      se.add<W>(AllocType::Host, volume(in.shape));
    }
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in.shape));

    TensorView<StorageCPU, W, 1> window_innermost = {nullptr, TensorShape<1>{diam_inner}};
    TensorView<StorageCPU, W, 1> window_outermost = {nullptr, TensorShape<1>{diam_outer}};

    auto req_inner = conv_innermost_.Setup(ctx, in, window_innermost);
    auto req_outer = conv_innermost_.Setup(ctx, in, window_outermost);

    req.AddInputSet(req_inner, false);
    req.AddInputSet(req_outer, false);

    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> out,
           const TensorView<StorageCPU, const In, ndim>& in, float sigma_inner, float sigma_outer) {

    int diam_inner = GetGaussianWindowDiameter(sigma_inner);
    int diam_outer = GetGaussianWindowDiameter(sigma_outer);

    auto *win_inner = ctx.scratchpad->Allocate<W>(AllocType::Host, diam_inner);
    auto *win_outer = ctx.scratchpad->Allocate<W>(AllocType::Host, diam_outer);
    TensorView<StorageCPU, W, 1> window_innermost = {win_inner, TensorShape<1>{diam_inner}};
    TensorView<StorageCPU, W, 1> window_outermost = {win_outer, TensorShape<1>{diam_outer}};
    TensorView<StorageCPU, W, ndim> intermediate;
    // if constexpr (std::is_same<Out, W>::value) {//todo sfiane
    //   intermediate = view(out);
    // } else {
      auto *tmp = ctx.scratchpad->Allocate<W>(AllocType::Host, volume(in.shape));
      intermediate = {tmp, in.shape};
    // }

    FillGaussian(window_innermost, sigma_inner);
    FillGaussian(window_outermost, sigma_outer);
    conv_innermost_.Run(ctx, intermediate, in, window_innermost, 1);
    conv_outermost_.Run(ctx, out, intermediate, window_outermost, 0);
  }

  ConvolutionCpu<W, In, W, ndim, has_channels> conv_innermost_;
  ConvolutionCpu<Out, W, W, ndim, has_channels> conv_outermost_;
  static_assert(std::is_same<W, float>::value, "Only floats as intermediate values are currently supported.");
};

template <typename Out, typename In, typename W, bool has_channels>
struct GaussianBlur<Out, In, W, 3, has_channels> {
  static constexpr int effective_ndim = has_channels ? 4 : 3;

  ConvolutionCpu<W, In, W, effective_ndim, has_channels> conv_innermost_;
  ConvolutionCpu<W, In, W, effective_ndim, has_channels> conv_middle_;
  ConvolutionCpu<Out, W, W, effective_ndim, has_channels> conv_outermost_;
};


}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_GAUSSIAN_BLUR_H_