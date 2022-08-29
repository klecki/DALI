// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/builtin/split.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"

namespace dali {

template <typename Backend>
bool Split<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) {

}

template <typename Backend>
void Split<Backend>::RunImpl(workspace_t<Backend> &ws) {

}


DALI_SCHEMA(Split)
  .DocStr(R"code(Split batch based on a predicate.)code")
  .NumInput(1)
  .NumOutput(2)
  .AddArg("predicate", "Boolean categorization of the input batch", DALI_BOOL, true)
  .MakeInternal();

DALI_REGISTER_OPERATOR(Split, Split<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Split, Split<GPUBackend>, GPU);


}  // namespace dali
