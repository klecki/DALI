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

#ifndef DALI_PIPELINE_OPERATORS_EXPRESSIONS_EXPRESSION_IMPL_CPU_H_
#define DALI_PIPELINE_OPERATORS_EXPRESSIONS_EXPRESSION_IMPL_CPU_H_

#include <vector>

#include "dali/core/any.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operators/expressions/arithmetic_meta.h"
#include "dali/pipeline/operators/expressions/expression_impl_factory.h"
#include "dali/pipeline/operators/expressions/expression_tree.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

// Assumes that it will get a workspace for given Backend,
// and the backend will store the inputs and outputs at given Backend.
template <ArithmeticOp op, typename Result,
          typename Left, bool LeftIsTensor,
          typename Right, bool RightIsTensor>
class ExprImplBinCPU : public ExprImplBase {
 public:
  ExprImplBinCPU() {}

  void Execute(ExprImplContext &ctx, const std::vector<ExtendedTileDesc> &tiles, TileRange range) override {
    DALI_ENFORCE(range.begin + 1 == range.end,
                 "CPU Expression implementation can handle only one tile at a time");
    const auto &tile = tiles[range.begin];
    Execute(static_cast<Result*>(tile.output), static_cast<const Left*>(tile.args[0]), static_cast<const Right*>(tile.args[1]), tile.desc.extent_size);
  }

 private:
  using meta = arithm_meta<op, CPUBackend>;

  static void Execute(Result *result, const Left *l, const Right *r, int64_t extent) {
    for (int64_t i = 0; i < extent; i++) {
      result[i] = meta::impl(l[i], r[i]);
    }
  }

  static void Execute(Result *result, Left l, const Right *r, int64_t extent) {
    for (int64_t i = 0; i < extent; i++) {
      result[i] = meta::impl(l, r[i]);
    }
  }

  static void Execute(Result *result, const Left *l, Right r, int64_t extent) {
    for (int64_t i = 0; i < extent; i++) {
      result[i] = meta::impl(l[i], r);
    }
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_EXPRESSIONS_EXPRESSION_IMPL_CPU_H_
