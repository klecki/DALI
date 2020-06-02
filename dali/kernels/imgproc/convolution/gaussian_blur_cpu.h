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
#include "include/dali/core/util.h"

namespace dali {
namespace kernels {

void FillGaussian(const TensorView<StorageCPU, float, 1>& window, float sigma) {
  int r = (window.num_elements() - 1) / 2;
  // Based on OpenCV
  sigma = sigma > 0 ? sigma : (r - 1) * 0.3 + 0.8;
  double exp_scale = 0.5 / (sigma * sigma);
  double sum = 0.;
  // Calculate first half
  for (int x = -r; x < 0; x++) {
    *window(x + r) = exp(-(x * x * exp_scale));
    sum += *window(x + r);
  }
  // Total sum, it's symmetric with 1 in the center.
  sum *= 2.;
  sum += 1.;
  double scale = 1. / sum;
  // place center, scaled element
  *window(r) = scale;
  // scale all elements so they sum up to 1, duplicate the second half
  for (int x = 0; x < r; x++) {
    *window(x) *= scale;
    *window(2 * r - x) = *window(x);
  }
}

template <typename Out, typename In, typename W, int ndim, bool has_channels = true>
struct GaussianBlurCpu {
  static constexpr int axes = ndim + (has_channels ? -1 : 0);

  KernelRequirements Setup(KernelContext& ctx, const InTensorCPU<In, ndim>& in,
                           const std::array<int, axes>& window_sizes) {
    KernelRequirements req;
    ScratchpadEstimator se;

    for (int i = 0; i < axes; i++) {
      se.add<W>(AllocType::Host, window_sizes[i]);
    }

    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in.shape));
    auto req_conv = conv_.Setup(ctx, in, window_sizes);
    req.AddInputSet(req_conv, false);

    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<int, axes>& window_sizes,
           const std::array<float, axes>& sigmas = uniform_array<float, axes>(0.f)) {
    std::array<TensorView<StorageCPU, W, 1>, axes> windows_tmp;
    std::array<TensorView<StorageCPU, const W, 1>, axes> windows;

    for (int i = 0; i < axes; i++) {
      auto* win_ptr = ctx.scratchpad->Allocate<W>(AllocType::Host, window_sizes[i]);
      windows_tmp[i] = {win_ptr, TensorShape<1>{window_sizes[i]}};
      FillGaussian(windows_tmp[i], sigmas[i]);
      windows[i] = windows_tmp[i];
    }

    conv_.Run(ctx, out, in, windows);
  }
  SeparableConvolutionCpu<Out, In, W, ndim, has_channels> conv_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_GAUSSIAN_BLUR_CPU__H_