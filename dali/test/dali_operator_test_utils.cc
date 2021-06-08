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


#include "dali/test/dali_operator_test_utils.h"

namespace dali {
namespace testing {

namespace utils {

const std::vector<Arguments> kDevices = {
        {{"device", std::string{"cpu"}}},
        {{"device", std::string{"gpu"}}},
};

}  // namespace utils

}  // namespace testing
}  // namespace dali

