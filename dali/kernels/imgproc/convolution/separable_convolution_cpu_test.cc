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

#include "dali/kernels/common/utils.h"
#include "dali/kernels/scratch.h"
// #include "dali/pipeline/data/tensor_list.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/imgproc/convolution/separable_convolution.h"


namespace dali {
namespace kernels {


template <typename T>
struct CyclicPixelWrapperTest: public ::testing::Test {
};

TYPED_TEST_SUITE_P(CyclicPixelWrapperTest);


template <int num_channels_, bool has_channels_>
struct cpw_params {
  static constexpr int num_channels = num_channels_;
  static constexpr bool has_channels = has_channels_;
};

using CyclicPixelWrapperValues =
    ::testing::Types<cpw_params<1, true>, cpw_params<3, true>, cpw_params<1, false>>;

TYPED_TEST_P(CyclicPixelWrapperTest, FillAndCycle) {
  constexpr int size = 6;
  constexpr int num_channels = TypeParam::num_channels;
  constexpr bool has_channels = TypeParam::has_channels;
  int tmp_buffer[size * num_channels];
  int input_buffer[size * num_channels];
  for (int i = 0; i < size * num_channels; i++) {
    input_buffer[i] = i;
    tmp_buffer[i] = -1;
  }
  CyclicPixelWrapper<int, has_channels> cpw(tmp_buffer, size, num_channels);
  EXPECT_EQ(0, cpw.Size());
  for (int i = 0; i < size; i++) {
    cpw.PushPixel(input_buffer + i * num_channels);
    EXPECT_EQ(tmp_buffer + i * num_channels, cpw.GetPixelOffset(i));
    for (int c = 0; c < num_channels; c++) {
      EXPECT_EQ(input_buffer[i * num_channels + c], cpw.GetPixelOffset(i)[c]);
    }
  }
  for (int i = 0; i < size; i++) {
    cpw.PopPixel();
    cpw.PushPixel(input_buffer + i * num_channels);
    for (int j = 0; j < size; j++) {
      // we're starting at i + 1 as we did already one Pop & Push operation
      int element = (i + 1 + j) % size;
      EXPECT_EQ(tmp_buffer + element * num_channels, cpw.GetPixelOffset(j));
      for (int c = 0; c < num_channels; c++) {
        EXPECT_EQ(input_buffer[element * num_channels + c], cpw.GetPixelOffset(j)[c]);
      }
    }
  }
}

void baseline_dot(span<int> result, span<const int> input, span<const int> window, int in_offset) {
  int num_channels = result.size();
  int num_elements = window.size();
  ASSERT_EQ(input.size(), num_channels * num_elements);
  for (int c = 0; c < num_channels; c++) {
    result[c] = 0;
    for (int i = 0; i < num_elements; i++) {
      int in_elem = (i + in_offset) % num_elements;
      result[c] += window[i] * input[in_elem * num_channels + c];
    }
  }
}

TYPED_TEST_P(CyclicPixelWrapperTest, DotProduct) {
  constexpr int size = 6;
  constexpr int num_channels = TypeParam::num_channels;
  constexpr bool has_channels = TypeParam::has_channels;
  int tmp_buffer[size * num_channels];
  int input_buffer[size * num_channels];
  int window[size];
  for (int i = 0; i < size * num_channels; i++) {
    input_buffer[i] = i;
    tmp_buffer[i] = -1;
  }
  for (int i = 0; i < size; i++) {
    window[i] = i;
  }
  int baseline[num_channels], result[num_channels];
  for (int c = 0; c < num_channels; c++) {
    baseline[c] = 0;
    for (int i = 0; i < size; i++) {
      baseline[c] += window[i] * input_buffer[i * num_channels + c];
    }
  }

  CyclicPixelWrapper<int, has_channels> cpw(tmp_buffer, size, num_channels);
  for (int i = 0; i < size; i++) {
    cpw.PushPixel(input_buffer + i * num_channels);
  }
  cpw.CalculateDot(result, window);
  baseline_dot(make_span(baseline), make_span(input_buffer), make_span(window), 0);
  for (int c = 0; c < num_channels; c++) {
    EXPECT_EQ(baseline[c], result[c]);
  }
  for (int i = 0; i < size; i++) {
    cpw.PopPixel();
    cpw.PushPixel(input_buffer + i * num_channels);
    cpw.CalculateDot(result, window);
    // again we start here at i + 1 offset
    baseline_dot(make_span(baseline), make_span(input_buffer), make_span(window), i + 1);
    for (int c = 0; c < num_channels; c++) {
      EXPECT_EQ(baseline[c], result[c]);
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(CyclicPixelWrapperTest, FillAndCycle, DotProduct);

INSTANTIATE_TYPED_TEST_SUITE_P(CyclicPixelWrapper, CyclicPixelWrapperTest, CyclicPixelWrapperValues);

template <typename Out, typename In, typename W>
void baseline_convolve_axis(Out *out, const In* in, const W* window, int len, int r, int channel_num, int64_t stride) {
  for (int i = 0; i < len; i++) {
    for (int c = 0; c < channel_num; c++) {
      out[i * stride + c] = 0;
      for (int d = -r; d <= r; d++) {
        if (i + d >= 0 && i + d < len) {
          out[i * stride + c] += in[(i + d) * stride + c] * window[d + r];
        } else {
          //todo: border handling
          out[i * stride + c] += 0 * window[d + r];
        }
      }
    }
  }
}


template <typename Out, typename In, typename W, int ndim>
void baseline_convolve(const TensorView<StorageCPU, Out, ndim> &out,
                       const TensorView<StorageCPU, In, ndim> &in,
                       const TensorView<StorageCPU, W, 1> &window, int axis, int r,
                       int current_axis = 0, int64_t offset = 0) {
  if (current_axis == ndim - 1) {
    auto stride = GetStrides(out.shape)[axis];
    baseline_convolve_axis(out.data + offset, in.data + offset, window.data, out.shape[axis], r, in.shape[ndim-1], stride);
  } else if (current_axis == axis) {
    baseline_convolve(out, in, window, axis, r, current_axis + 1, offset);
  } else {
    for (int i = 0; i < out.shape[current_axis]; i++) {
      auto stride = GetStrides(out.shape)[current_axis];
      baseline_convolve(out, in, window, axis, r, current_axis + 1, offset + i * stride);
    }
  }
}

// struct convolution_test_setup {
//   int window_size;
//   int num_channels;
// };


// class SeparableConvolutionTest : public testing::TestWithParam

TEST(SeparableConvolutionTest, OneAxisTest) {
  constexpr int window_size = 3;
  constexpr int num_channels = 3;
  constexpr int input_len = 11;
  constexpr int r = window_size / 2;
  TestTensorList<float, 1> kernel_window;
  kernel_window.reshape(uniform_list_shape<1>(1, {window_size}));
  TestTensorList<uint8_t, 2> input;
  input.reshape(uniform_list_shape<2>(1, {input_len, num_channels}));
  TestTensorList<float, 2> output, baseline_output;
  output.reshape(uniform_list_shape<2>(1, {input_len, num_channels}));
  baseline_output.reshape(uniform_list_shape<2>(1, {input_len, num_channels}));

  using Kernel = SeparableConvolution<float, uint8_t, float, 2>;

  KernelContext ctx;
  Kernel kernel;

  auto out = output.cpu()[0];
  auto baseline_out = baseline_output.cpu()[0];
  auto in = input.cpu()[0];
  auto k_win = kernel_window.cpu()[0];
  // almost box filter, with raised center
  for (int i = 0; i < window_size; i++) {
    if (i < window_size / 2) {
      k_win.data[i] = 1;
    } else if (i == window_size / 2) {
      k_win.data[i] = 2;
    } else {
      k_win.data[i] = 1;
    }
  }
  for (int i = 0; i < input_len * num_channels; i++) {
    in.data[i] = 0;
  }
  ConstantFill(in, 0);
  for (int c = 0; c < num_channels; c++) {
    in.data[(input_len / 2) * num_channels + c] = 10 + 2 * c;
  }

  auto req = kernel.Setup(ctx, in, k_win, 0);
  // this is painful
  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(req.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  kernel.Run(ctx, out, in, k_win, 0, 1);
  baseline_convolve(baseline_out, in, k_win, 0, r);
  Check(out, baseline_out);

}



TEST(SeparableConvolutionTest, TwoAxesTest) {
  constexpr int window_size = 3;
  constexpr int height = 11;
  constexpr int width = 21;
  constexpr int r = window_size / 2;
  constexpr int ndim = 3;
  constexpr int num_channels = 3;
  TensorShape<ndim> shape = {11, 21, num_channels};

  TestTensorList<float, 1> kernel_window;
  kernel_window.reshape(uniform_list_shape<1>(1, {window_size}));
  TestTensorList<uint8_t, ndim> input;
  input.reshape(uniform_list_shape<ndim>(1, shape));
  TestTensorList<float, ndim> output, baseline_output;
  output.reshape(uniform_list_shape<ndim>(1, shape));
  baseline_output.reshape(uniform_list_shape<ndim>(1, shape));

  using Kernel = SeparableConvolution<float, uint8_t, float, ndim>;

  KernelContext ctx;
  Kernel kernel;

  auto out = output.cpu()[0];
  auto baseline_out = baseline_output.cpu()[0];
  auto in = input.cpu()[0];
  auto k_win = kernel_window.cpu()[0];

  // almost box filter, with raised center
  for (int i = 0; i < window_size; i++) {
    if (i < window_size / 2) {
      k_win.data[i] = 1;
    } else if (i == window_size / 2) {
      k_win.data[i] = 2;
    } else {
      k_win.data[i] = 1;
    }
  }

  ConstantFill(in, 0);

  for (int c = 0; c < num_channels; c++) {
    *in(shape[0] / 2, shape[1] / 2, c) = 20 + 2 * c;
    *in(shape[0] / 2 - 1, shape[1] / 2, c) = 10;
    *in(shape[0] / 2 + 1, shape[1] / 2, c) = 10;
    *in(shape[0] / 2, shape[1] / 2 - 1, c) = 10;
    *in(shape[0] / 2, shape[1] / 2 + 1, c) = 10;
  }

  auto req = kernel.Setup(ctx, in, k_win, 0);
  // this is painful
  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(req.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  // axis 0
  ConstantFill(out, -1);
  kernel.Run(ctx, out, in, k_win, 0, 2);
  baseline_convolve(baseline_out, in, k_win, 0, r);
  Check(out, baseline_out);

  // axis 1
  ConstantFill(out, -1);
  kernel.Run(ctx, out, in, k_win, 1, 2);
  baseline_convolve(baseline_out, in, k_win, 1, r);
  Check(out, baseline_out);
}


}  // namespace kernels
}  // namespace dali

