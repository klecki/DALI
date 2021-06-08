// Copyright (c) 2019, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include <memory>
#include "dali/operators/decoder/nvjpeg/fused/nvjpeg_decoder_random_crop.h"
#include "dali/util/random_crop_generator.h"

namespace dali {

DALI_REGISTER_OPERATOR(decoders__ImageRandomCrop, nvJPEGDecoderRandomCrop, Mixed);
DALI_REGISTER_OPERATOR(ImageDecoderRandomCrop, nvJPEGDecoderRandomCrop, Mixed);

}  // namespace dali
