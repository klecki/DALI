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
#include <opencv2/imgproc.hpp>

#include "dali/kernels/common/utils.h"
#include "dali/kernels/scratch.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
// #include "dali/kernels/imgproc/convolution/gaussian_blur_cpu.h"


// namespace dali {
// namespace kernels {



// // TODO, OpenCV have special cases of precomputed values for for kernels of sizes 1-7 and sigma = 0
// TEST(GaussianBlurCpuTest, KernelWindow) {
//   std::vector<std::pair<int, float>> size_sigma_pairs = {{1, 0}, {3, 0}, {5, 0}, {7, 0}, {9, 0}, {21, 0},
//     {1, 0.8f}, {3, 0.8f}, {5, 0.8f}, {7, 0.8f}, {9, 0.8f}, {21, 0.8f},
//     {1, 3}, {3, 3}, {5, 3}, {7, 3}, {9, 3}, {21, 3}, {1, 0.25f}, {3, 0.25f}, {5, 0.25f}, {7, 0.25f}, {9, 0.25f}, {21, 0.25f}};
//   TestTensorList<float, 1> window;
//   for (const auto &size_sigma : size_sigma_pairs) {
//     int size;
//     float sigma;
//     std::tie(size, sigma) = size_sigma;
//     TensorListShape<1> shape({TensorShape<1>{size}});
//     window.reshape(shape);
//     auto window_view = window.cpu()[0];
//     FillGaussian(window_view, sigma);
//     auto mat = cv::getGaussianKernel(size, sigma, CV_32F);
//     for (int i = 0; i < size; i++) {
//       EXPECT_NEAR(window_view.data[i], mat.at<float>(i), 1e-7f) << "size: " << size << ", sigma: " << sigma;
//     }
//   }
// }

// TEST(GaussianBlurCpuTest, Compiles) {
//   GaussianBlurCpu<uint8_t, uint8_t, float, 2, false> test;

// }


// }  // namespace kernels
// }  // namespace dali

