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
#include "dali/kernels/imgproc/convolution/separable_convolution_cpu.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {
namespace kernels {

int GetGaussianWindowDiameter(float sigma) {
  // Used by OpenCV: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
  // (sigma - 0.8) / 0.3 = (ksize - 1)*0.5 -1
  // r = (ksize - 1) / 2 = (sigma - 0.8) / 0.3 + 1 = sigma / 0.3 - 1.(6)
  // TODO(klecki): approx:
  int radius = ceilf(sigma * 3);
  int diameter = 1 + 2 * radius;
  return diameter;
}

void FillGaussian(const TensorView<StorageCPU, float, 1>& window, float sigma) {
  int r = (window.num_elements() - 1) / 2;
  for (int x = -r; x <= r; x++) {
    // directly using the formula, todo:optimize and calculate only half
    *window(x + r) = 1.0 / sqrt(2 * M_PI * sigma * sigma) * exp(-((x * x) / (2 * sigma * sigma)));
  }
}

template <typename Out, typename In, typename W, int ndim, bool has_channels = true>
struct GaussianBlurCpu {
  static constexpr int axes = ndim + (has_channels ? -1 : 0);

  KernelRequirements Setup(KernelContext& ctx, const InTensorCPU<In, ndim>& in,
                           const std::array<float, axes>& sigmas) {
    KernelRequirements req;
    ScratchpadEstimator se;

    std::array<int, axes> diams;
    std::array<TensorView<StorageCPU, W, 1>, axes> windows;
    for (int i = 0; i < axes; i++) {
      diams[i] = GetGaussianWindowDiameter(sigmas[i]);
      se.add<W>(AllocType::Host, diams[i]);
      windows[i] = {nullptr, TensorShape<1>{diams[i]}};
    }

    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in.shape));
    auto req_conv = conv_.Setup(ctx, in, windows);
    req.AddInputSet(req_conv, false);

    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<float, axes>& sigmas) {
    std::array<int, axes> diams;
    std::array<TensorView<StorageCPU, W, 1>, axes> windows;
    std::array<float, axes> scales;  // todo remove

    for (int i = 0; i < axes; i++) {
      diams[i] = GetGaussianWindowDiameter(sigmas[i]);
      auto* win_ptr = ctx.scratchpad->Allocate<W>(AllocType::Host, diams[i]);
      windows[i] = {win_ptr, TensorShape<1>{diams[i]}};
      FillGaussian(windows[i], sigmas[i]);
      scales[i] = 0;
    }

    conv_.Run(ctx, out, in, windows, scales);
  }
  SeparableConvolutionCpu<Out, In, W, axes, has_channels> conv_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_GAUSSIAN_BLUR_CPU__H_