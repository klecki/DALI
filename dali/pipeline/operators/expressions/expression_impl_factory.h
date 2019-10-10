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

#ifndef DALI_PIPELINE_OPERATORS_EXPRESSIONS_EXPRESSION_IMPL_FACTORY_H_
#define DALI_PIPELINE_OPERATORS_EXPRESSIONS_EXPRESSION_IMPL_FACTORY_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/any.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operators/expressions/arithmetic_meta.h"
#include "dali/pipeline/operators/expressions/constant_storage.h"
#include "dali/pipeline/operators/expressions/expression_tree.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/workspace.h"
#include "include/dali/core/small_vector.h"

namespace dali {

inline OutputSamplePtr GetOutputSamplePointer(HostWorkspace &ws, int output_idx, int sample_idx) {
  return ws.template OutputRef<CPUBackend>(output_idx)[sample_idx].raw_mutable_data();
}

inline OutputSamplePtr GetOutputSamplePointer(DeviceWorkspace &ws, int output_idx, int sample_idx) {
  return ws.template OutputRef<GPUBackend>(output_idx).raw_mutable_tensor(sample_idx);
}

inline InputSamplePtr GetInputSamplePointer(HostWorkspace &ws, int input_idx, int sample_idx) {
  return ws.template InputRef<CPUBackend>(input_idx)[sample_idx].raw_data();
}

inline InputSamplePtr GetInputSamplePointer(DeviceWorkspace &ws, int input_idx, int sample_idx) {
  return ws.template InputRef<GPUBackend>(input_idx).raw_tensor(sample_idx);
}

template <typename Backend>
inline OutputSamplePtr GetOutput(const ExprFunc &func, workspace_t<Backend> &ws, TileDesc tile) {
  return GetOutputSamplePointer(ws, 0, tile.sample_idx);
}

/**
 * @brief Type erased obtaining pointers to inputs
 */
template <typename Backend>
inline ArgPack GetArgPack(const ExprFunc &func, workspace_t<Backend> &ws,
                          ConstantStorage<Backend> &st, const OpSpec &spec, TileDesc tile) {
  ArgPack result;
  result.resize(func.GetSubexpressionCount());
  for (int i = 0; i < func.GetSubexpressionCount(); i++) {
    if (func[i].GetNodeType() == NodeType::Function) {
      DALI_FAIL("Function nodes not supported as subexpressions");
    }
    if (func[i].GetNodeType() == NodeType::Constant) {
      const auto &constant = dynamic_cast<const ExprConstant &>(func[i]);
      result[i] = st.GetPointer(constant.GetConstIndex(), constant.GetTypeId());
    }
    if (func[i].GetNodeType() == NodeType::Tensor) {
      const auto &tensor = dynamic_cast<const ExprTensor &>(func[i]);
      auto input_idx = tensor.GetInputIndex();
      result[i] = GetInputSamplePointer(ws, input_idx, tile.sample_idx);
    }
  }
  return result;
}

template <typename Backend>
inline std::vector<ExtendedTileDesc> TransformDescs(const std::vector<TileDesc> &tiles, const ExprFunc &func, workspace_t<Backend> &ws,
                          ConstantStorage<Backend> &st, const OpSpec &spec) {
  for (auto &tile : tiles) {

  }
}

std::unique_ptr<ExprImplBase> ExprImplFactory(const HostWorkspace &ws, const ExprNode &expr);

std::unique_ptr<ExprImplBase> ExprImplFactory(const DeviceWorkspace &ws, const ExprNode &expr);

struct ExprImplCache {
  template <typename Backend>
  ExprImplBase *GetExprImpl(const ExprNode &expr) {
    auto node_desc = expr.GetNodeDesc();
    auto it = cache_.find(node_desc);
    if (it != cache_.end()) {
      return it->second.get();
    }
    auto new_impl = ExprImplFactory(workspace_t<Backend>{}, expr);
    auto ptr = std::shared_ptr<ExprImplBase>(std::move(new_impl));
    cache_[node_desc] = ptr;
    return ptr.get();
  }

 private:
  std::map<std::string, std::shared_ptr<ExprImplBase>> cache_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_EXPRESSIONS_EXPRESSION_IMPL_FACTORY_H_
