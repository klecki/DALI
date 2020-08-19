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

#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <tuple>
#include <vector>

#include "dali/core/boundary.h"
#include "dali/core/convert.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/convolution_cpu.h"
#include "dali/kernels/imgproc/convolution/convolution_gpu.h"
#include "dali/kernels/scratch.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/imgproc/convolution/baseline_convolution.h"

namespace dali {
namespace kernels {

TEST(CONV, CONV) {
  ConvolutionGpu<float, float, float, 3, 1> kernel;


}

template <int ndim_, bool has_channels_, int axis_, int window_size_, typename InType_>
struct convolution_params {
  static constexpr int ndim = ndim_;
  static constexpr bool has_channels = has_channels_;
  static constexpr int axis = axis_;
  static constexpr int window_size = window_size_;
  using InType = InType_;
};

template <typename T>
struct ConvolutionGpuKernelTest : public ::testing::Test {
  using KernelCpu =
      ConvolutionCpu<float, typename T::InType, float, T::ndim, T::axis, T::has_channels>;
  using KernelGpu =
      ConvolutionGpu<float, typename T::InType, float, T::ndim, T::axis, T::has_channels>;

  TensorListShape<T::ndim> GetShape() {
    if (T::has_channels) {
      return shape_ch_.template last<T::ndim>();
    } else {
      return shape_noch_.template last<T::ndim>();
    }
  }

  void SetUp() override {
    kernel_window_.reshape(shape_window);
    k_win_ = kernel_window_.cpu();

    // almost box filter, with raised center
    for (int sample = 0; sample < shape_window.num_samples(); sample++) {
      int window_size = shape_window[sample][0];
      for (int i = 0; i < window_size; i++) {
        if (i < window_size / 2) {
          k_win_[sample].data[i] = 1;
        } else if (i == window_size / 2) {
          k_win_[sample].data[i] = 2;
        } else {
          k_win_[sample].data[i] = 1;
        }
      }
    }

    input_.reshape(GetShape());
    baseline_in_ = input_.cpu();

    // ConstantFill(in_, 0);

    std::mt19937 rng;
    UniformRandomFill(baseline_in_, rng, 0, 255);
    in_ = input_.gpu();

    output_.reshape(GetShape());
    out_ = output_.gpu();
    baseline_output_.reshape(GetShape());
    baseline_out_ = baseline_output_.cpu();
  }

  void RunTest() {
    KernelContext ctx_cpu, ctx_gpu;
    KernelCpu kernel_cpu;
    KernelGpu kernel_gpu;

    auto data_shape = GetShape();
    int num_samples = data_shape.size();

    for (int sample = 0; sample < num_samples; sample++) {
      int window_size = shape_window[sample][0];
      auto req = kernel_cpu.Setup(ctx_cpu, data_shape[sample], window_size);

      ScratchpadAllocator scratch_alloc;
      scratch_alloc.Reserve(req.scratch_sizes);
      auto scratchpad = scratch_alloc.GetScratchpad();
      ctx_cpu.scratchpad = &scratchpad;

      kernel_cpu.Run(ctx_cpu, baseline_out_[sample], baseline_in_[sample], k_win_[sample]);
    }

    auto req = kernel_gpu.Setup(ctx_gpu, in_.shape, shape_window);

    ScratchpadAllocator scratch_alloc;
    scratch_alloc.Reserve(req.scratch_sizes);
    auto scratchpad = scratch_alloc.GetScratchpad();
    ctx_gpu.scratchpad = &scratchpad;
    kernel_gpu.Run(ctx_gpu, out_, in_, k_win_);

    auto out_cpu_ = output_.cpu();
    // Check(out_cpu_, baseline_out_);

    double eps = 1e-2;
    Check(out_cpu_, baseline_out_, EqualEps(eps));
  }

  TestTensorList<float, 1> kernel_window_;
  TestTensorList<typename T::InType, T::ndim> input_;
  TestTensorList<float, T::ndim> output_;
  TestTensorList<float, T::ndim> baseline_output_;

  TensorListView<StorageCPU, float, 1> k_win_;
  TensorListView<StorageGPU, typename T::InType, T::ndim> in_;
  TensorListView<StorageGPU, float, T::ndim> out_;
  TensorListView<StorageCPU, typename T::InType, T::ndim> baseline_in_;
  TensorListView<StorageCPU, float, T::ndim> baseline_out_;

  // const TensorListShape<> shape_ch_ = {{64, 64, 64, 3}};
  // const TensorListShape<> shape_noch_ = {{64, 64, 64}};
  const TensorListShape<> shape_ch_ = {{64, 64, 64, 3}, {164, 164, 164, 3}, {512, 512, 512, 3}, {64, 128, 512, 3}, {64, 512, 128, 3}};
  const TensorListShape<> shape_noch_ = {{64, 64, 64}, {164, 164, 164}, {512, 512, 512}, {64, 128, 512}, {64, 512, 128}};
  const TensorListShape<1> shape_window = uniform_list_shape(shape_ch_.num_samples(), TensorShape<1>{T::window_size});
};

TYPED_TEST_SUITE_P(ConvolutionGpuKernelTest);

// template <int ndim_, bool has_channels_, int axis_, int window_size_, typename InType_>
using ConvolutionTestValues = ::testing::Types<
    convolution_params<2, false, 0, 3, float>,
    convolution_params<2, false, 1, 3, float>,
    // convolution_params<3, true, 0, 3, float>,
    // convolution_params<3, true, 1, 3, float>,
    convolution_params<2, false, 0, 15, float>,
    convolution_params<2, false, 1, 15, float>
    // convolution_params<3, true, 0, 15, float>,
    // convolution_params<3, true, 1, 15, float>
                                                    >;
// convolution_params<1, false, 0, 1, uint8_t>,
                                              //  convolution_params<1, false, 0, 3, uint8_t>,
                                              //  convolution_params<1, false, 0, 21, uint8_t>,
                                              //  convolution_params<1, false, 0, 51, uint8_t>,
                                              //  convolution_params<2, true, 0, 1, uint8_t>,
                                              //  convolution_params<2, true, 0, 3, uint8_t>,
                                              //  convolution_params<2, true, 0, 21, uint8_t>,
                                              //  convolution_params<2, true, 0, 51, uint8_t>,

                                              //  convolution_params<1, false, 0, 1, float>,
                                              //  convolution_params<1, false, 0, 3, float>,
                                              //  convolution_params<1, false, 0, 21, float>,
                                              //  convolution_params<1, false, 0, 51, float>,
                                              //  convolution_params<2, true, 0, 1, float>,
                                              //  convolution_params<2, true, 0, 3, float>,
                                              //  convolution_params<2, true, 0, 21, float>,
                                              //  convolution_params<2, true, 0, 51, float>,

                                              //  convolution_params<2, false, 0, 1, uint8_t>,
                                              //  convolution_params<2, false, 0, 3, uint8_t>,
                                              //  convolution_params<2, false, 1, 1, uint8_t>,
                                              //  convolution_params<2, false, 1, 3, uint8_t>,
                                              //  convolution_params<3, true, 0, 3, uint8_t>,
                                              //  convolution_params<3, true, 1, 3, uint8_t>,

                                              //  convolution_params<3, false, 1, 1, uint8_t>,
                                              //  convolution_params<3, false, 1, 3, uint8_t>,
                                              //  convolution_params<3, false, 1, 7, uint8_t>,
                                              //  convolution_params<3, false, 1, 11, uint8_t>,
                                              //  convolution_params<3, false, 1, 21, uint8_t>,
                                              //  convolution_params<3, false, 1, 101, uint8_t>,

                                              //  convolution_params<3, false, 1, 1, float>,
                                              //  convolution_params<3, false, 1, 3, float>,
                                              //  convolution_params<3, false, 1, 21, float>,
                                              //  convolution_params<3, false, 1, 101, float>>;

TYPED_TEST_P(ConvolutionGpuKernelTest, DoConvolution) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(ConvolutionGpuKernelTest, DoConvolution);
INSTANTIATE_TYPED_TEST_SUITE_P(ConvolutionGpuKernel, ConvolutionGpuKernelTest,
                               ConvolutionTestValues);

}  // namespace kernels
}  // namespace dali
