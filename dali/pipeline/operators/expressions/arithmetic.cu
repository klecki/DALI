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

#include "dali/kernels/type_tag.h"
#include "dali/pipeline/operators/expressions/arithmetic.h"

namespace dali {

template <>
void ArithmeticGenericOp<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  tiles_per_task_.reserve(exec_order_.size());
  for (auto &expr_task : exec_order_) {
    tiles_per_task_.push_back(TransformDescs(tile_cover_, dynamic_cast<const ExprFunc&>(*expr_task.ctx.node),
                                             ws, constant_storage_, spec_));
  }
  ws.OutputRef<GPUBackend>(0).SetLayout(result_layout_);
  assert(tile_range_.size() == 1 && "Expected to cover whole GPU execution by 1 task");
  for (int i = 0; i < exec_order_.size(); i++) {
    // call impl for whole batch
    exec_order_[i].impl->Execute(exec_order_[i].ctx, tiles_per_task_[i], tile_range_[0]);
  }
}

DALI_REGISTER_OPERATOR(ArithmeticGenericOp, ArithmeticGenericOp<GPUBackend>, GPU);


}  // namespace dali
