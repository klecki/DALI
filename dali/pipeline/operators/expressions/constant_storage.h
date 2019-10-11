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

#ifndef DALI_PIPELINE_OPERATORS_EXPRESSIONS_CONSTANT_STORAGE_H_
#define DALI_PIPELINE_OPERATORS_EXPRESSIONS_CONSTANT_STORAGE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operators/expressions/expression_tree.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
class ConstantStorage {
 public:
  void Initialize(const OpSpec &spec, cudaStream_t stream,
                  const std::vector<ExprConstant *> &constant_nodes) {
    auto integers_vec = spec.HasArgument("integer_scalars")
                            ? spec.GetRepeatedArgument<int>("integer_scalars")
                            : std::vector<int>{};
    auto reals_vec = spec.HasArgument("real_scalars")
                         ? spec.GetRepeatedArgument<float>("real_scalars")
                         : std::vector<float>{};

    std::vector<ExprConstant *> integer_nodes, real_nodes;
    for (auto *node : constant_nodes) {
      if (IsIntegral(node->GetTypeId())) {
        integer_nodes.push_back(node);
      } else {
        real_nodes.push_back(node);
      }
    }

    Tensor<CPUBackend> cpu_integers, cpu_reals;
    Rewrite(integers_vec, cpu_integers, integers_, stream, integer_nodes);
    Rewrite(reals_vec, cpu_reals, reals_, stream, real_nodes);
  }

  const void *GetPointer(int constant_idx, DALIDataType type_id) {
    if (IsIntegral(type_id)) {
      return integers_.template data<char>() + constant_idx * sizeof(int64_t);
    }
    return reals_.template data<char>() + constant_idx * sizeof(int64_t);
  }

 private:
  Tensor<Backend> integers_, reals_;

  template <typename T>
  void Rewrite(const std::vector<T> constants, Tensor<CPUBackend> &temporary,
               Tensor<Backend> &target, cudaStream_t stream,
               const std::vector<ExprConstant *> &constant_nodes) {
    temporary.Resize({static_cast<int64_t>(constants.size() * sizeof(int64_t))});
    char *data = temporary.mutable_data<char>();
    DALI_ENFORCE(constants.size() == constant_nodes.size(),
                 make_string("Expected number of passed contants to match nubmer of contant nodes "
                             "in expression tree. Found",
                             constants.size(), "constant arguments provided and",
                             constant_nodes.size(), "constant nodes of that type in expression."));
    for (auto *node : constant_nodes) {
      TYPE_SWITCH(node->GetTypeId(), type2id, Type,
                  (uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float16, float, double),
                  (auto idx = node->GetConstIndex();
                   auto *ptr = reinterpret_cast<Type *>(data + idx * sizeof(int64_t));
                   *ptr = static_cast<Type>(constants[idx]);),
                  DALI_FAIL("No suitable type found"););  // NOLINT(whitespace/parens)
    }

    target.Copy(temporary, stream);  // TODO(klecki): stream. limit to gpu only
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_EXPRESSIONS_CONSTANT_STORAGE_H_