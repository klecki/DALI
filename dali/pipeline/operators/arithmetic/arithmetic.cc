// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/arithmetic/arithmetic.h"
#include "dali/kernels/type_tag.h"

namespace dali {

template <>
void ArithmeticGenericOp<CPUBackend>::RunImpl(HostWorkspace &ws) {
  auto &pool = ws.GetThreadPool();
  for (int task_idx = 0; task_idx < num_tasks_; task_idx++) {
    // TODO(klecki): reduce lambda footprint
    pool.DoWorkWithID([this, &ws, task_idx](int thread_idx) {
      auto range = tile_range_[task_idx];
      // Go over "tiles"
      for (int extent_idx = range.begin; extent_idx < range.end; extent_idx++) {
        // Go over expression tree in some provided order
        for (auto &expr_task : exec_order_) {
          expr_task.impl->Execute(ws, spec_, expr_task.ctx, tile_cover_,
                                  {extent_idx, extent_idx + 1});
        }
      }
    });
  }
}

DALI_SCHEMA(ArithmeticGenericOp)
    .DocStr(R"code(Arithmetic operator capable of executing expression tree of elementwise
 arithmetic operations.)code")
    .AddArg("expression_desc", R"code(Polish notation describing the expression extendend with
 parentheses, see https://en.wikipedia.org/wiki/Polish_notation.
Examples:
  add(&0 mul(&1 $0:int8))
  add(&0 rand())
Inputs are separated by spaces, &<uint> indicates tensor input,
$<uint>:<type_string> indicates constant.
)code",
            DALIDataType::DALI_STRING, false)
    .AddOptionalArg("integer_scalars", "", std::vector<int>{})
    .NumInput(1, 64)  // TODO(klecki): Some arbitrary number that needs to be validated in operator
    .AddOptionalArg("float_scalars", "", std::vector<float>{})
    .NumOutput(1)
    .MakeInternal();

DALI_REGISTER_OPERATOR(ArithmeticGenericOp, ArithmeticGenericOp<CPUBackend>, CPU);

}  // namespace dali
