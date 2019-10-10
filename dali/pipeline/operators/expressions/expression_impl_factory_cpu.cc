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

#include <memory>

#include "dali/core/static_switch.h"
#include "dali/pipeline/operators/expressions/arithmetic_meta.h"
#include "dali/pipeline/operators/expressions/expression_impl_cpu.h"
#include "dali/pipeline/operators/expressions/expression_impl_factory.h"
#include "dali/pipeline/operators/expressions/expression_tree.h"

namespace dali {

std::unique_ptr<ExprImplBase> ExprImplFactory(const HostWorkspace &ws, const ExprNode &expr) {
  std::unique_ptr<ExprImplBase> result;
  DALI_ENFORCE(expr.GetNodeType() == NodeType::Function, "Only function nodes can be executed.");

  switch (expr.GetSubexpressionCount()) {
    case 1:
      return ExprImplFactory1<ExprImplCpuT, CPUBackend>(ws, dynamic_cast<const ExprFunc &>(expr));
    case 2:
      return ExprImplFactory2<ExprImplCpuTT, ExprImplCpuTC, ExprImplCpuCT, CPUBackend>(
          ws, dynamic_cast<const ExprFunc &>(expr));
    default:
      DALI_FAIL("Expressions with " + std::to_string(expr.GetSubexpressionCount()) +
                " subexpressions are not supported. No implemetation found.");
      break;
  }
}

}  // namespace dali
