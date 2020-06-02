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

namespace dali {

DALI_SCHEMA(GaussianBlur)
    .DocStr(R"code(Apply Gaussian Blur to the input.
Separable convolution with Gaussian Kernel is used to calculate the output.

Every axis can have distinct sigma value specified, the channels are not considered an axis.

User can specify either ``sigma`` that will be used uniformly for all axes
or ``sigma_0``, ``sigma_1``, ``sigma_2``. ``sigma_i`` will be used for corresponding axis ``i``.

In case of sequences of ``FHWC`` layout, the frame dimension is skipped, ``sigma_0``
is used for ``H`` and ``sigma_1`` for ``W``.
)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .AddOptionalArg("window_size", "The diameter of kernel window", 0, true)
    .AddOptionalArg<float>("sigma", R"code()code", 0.f, true)
    .AddOptionalArg<float>("sigma_0", R"code()code", nullptr, true)
    .AddOptionalArg<float>("sigma_1", R"code()code", nullptr, true)
    .AddOptionalArg<float>("sigma_2", R"code()code", nullptr, true);

// constexpr int GaussianBlur<CPUBackend>::kMaxDim = 3;

DALI_REGISTER_OPERATOR(GaussianBlur, GaussianBlur<CPUBackend>, CPU);

}  // namespace dali
