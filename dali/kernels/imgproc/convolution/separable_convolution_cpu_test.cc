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

using CyclicPixelWrapperValues =
    ::testing::Types<std::integral_constant<int, 1>, std::integral_constant<int, 3>>;

TYPED_TEST_P(CyclicPixelWrapperTest, FillAndCycle) {
  constexpr int size = 6;
  constexpr int num_channels = TypeParam::value;
  int tmp_buffer[size * num_channels];
  int input_buffer[size * num_channels];
  for (int i = 0; i < size * num_channels; i++) {
    input_buffer[i] = i;
    tmp_buffer[i] = -1;
  }
  CyclicPixelWrapper<int> cpw(tmp_buffer, size, num_channels);
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
  constexpr int num_channels = TypeParam::value;;
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

  CyclicPixelWrapper<int> cpw(tmp_buffer, size, num_channels);
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


// template <typename Out, typename In, typename W>
// void

TEST(SeparableConvolutionTest, OneAxisTest) {
  constexpr int window_size = 3;
  constexpr int num_channels = 3;
  constexpr int input_len = 11;
  TestTensorList<float, 1> kernel_window;
  kernel_window.reshape(uniform_list_shape<1>(1, {window_size}));
  TestTensorList<uint8_t, 2> input;
  input.reshape(uniform_list_shape<2>(1, {input_len, num_channels}));
  TestTensorList<uint8_t, 2> padded_input;
  padded_input.reshape(uniform_list_shape<2>(1, {input_len + window_size - 1, num_channels}));
  TestTensorList<float, 2> output;
  output.reshape(uniform_list_shape<2>(1, {input_len, num_channels}));

  using Kernel1d = SeparableConvolution<float, uint8_t, float, 2>;
  using Kernel2d = SeparableConvolution<float, uint8_t, float, 3>;
  using Kernel3d = SeparableConvolution<float, uint8_t, float, 4>;

  KernelContext ctx;
  Kernel1d k1d;

  auto out = output.cpu()[0];
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
  for (int c = 0; c < num_channels; c++) {
    in.data[(input_len / 2) * num_channels + c] = 10;
  }

  auto req = k1d.Setup(ctx, in, k_win, 0);
  // this is painful
  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(req.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  k1d.Run(ctx, out, in, k_win, 0, 1);
  for (int i = 0; i < input_len; i++) {
    std::cout << "{ ";
    for (int c = 0; c< num_channels; c++) {
      std::cout << out.data[i * num_channels + c] << ", ";
    }
    std::cout << " }\n";
  }
  std::cout << std::endl;
  Kernel2d k2d;
  Kernel3d k3d;
}


}  // namespace kernels
}  // namespace dali
