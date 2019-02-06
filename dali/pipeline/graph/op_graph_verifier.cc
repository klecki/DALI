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

#include <algorithm>
#include <string>
#include <sstream>
#include <vector>

#include "dali/error_handling.h"
#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/graph/op_graph_verifier.h"

namespace dali {

namespace {

std::string concatenate_alternatives(const std::set<DALIOpType>& vec) {
  if (vec.empty())
    return "";
  std::stringstream result;
  bool first = true;
  for (auto op_type : vec) {
    if (first) {
      first = false;
    } else {
      result << ", ";
    }
    result << to_string(op_type);
  }
  return result.str();
}


}  // namespace

std::vector<std::set<DALIOpType>> ParentOpTypeConstraints() {
  std::vector<std::set<DALIOpType>> allowed_parents;
  allowed_parents.resize(static_cast<int>(DALIOpType::DALI_OP_TYPE_COUNT));
  allowed_parents[static_cast<int>(DALIOpType::DALI_GPU)    ] = { DALIOpType::DALI_GPU,
                                                                  DALIOpType::DALI_MIXED,
                                                                  DALIOpType::DALI_SUPPORT };
  allowed_parents[static_cast<int>(DALIOpType::DALI_CPU)    ] = { DALIOpType::DALI_CPU,
                                                                  DALIOpType::DALI_SUPPORT };
  allowed_parents[static_cast<int>(DALIOpType::DALI_MIXED)  ] = { DALIOpType::DALI_CPU };
  // TODO(klecki): there are no inputs to Support operators
  allowed_parents[static_cast<int>(DALIOpType::DALI_SUPPORT)] = { /* DALIOpType::DALI_SUPPORT */ };
  return allowed_parents;
}

std::vector<int> ArgumentInputConstraints() {
  std::vector<int> allows_argument_input;
  allows_argument_input.resize(static_cast<int>(DALIOpType::DALI_OP_TYPE_COUNT));
  allows_argument_input[static_cast<int>(DALIOpType::DALI_GPU)    ] = true;
  allows_argument_input[static_cast<int>(DALIOpType::DALI_CPU)    ] = true;
  allows_argument_input[static_cast<int>(DALIOpType::DALI_MIXED)  ] = false;
  allows_argument_input[static_cast<int>(DALIOpType::DALI_SUPPORT)] = false;
  return allows_argument_input;
}


/**
 * @brief Check if parent nodes have compatible DALIOpType
 */
void CheckParentConstraints(const OpGraph& op_graph, const OpNode& op) {
  static const auto allowed_parent_type = ParentOpTypeConstraints();
  for (auto parent_id : op.parents) {
    const auto& parent = op_graph.Node(parent_id);
    const auto& allowed_parents = allowed_parent_type[static_cast<int>(op.op_type)];
    DALI_ENFORCE(allowed_parents.find(parent.op_type) != allowed_parents.end(),
                 "Op " + op.instance_name + " of type " + to_string(op.op_type) +
                     " has parent Op " + parent.instance_name +
                     " of incompatible type: " + to_string(parent.op_type) + ". Expected one of: " +
                     concatenate_alternatives(allowed_parent_type[static_cast<int>(op.op_type)]) +
                     ".");
  }
}

/**
 * @brief Check support for Argument Inputs and if Argument Inputs are produed by Support Ops
 */
void CheckArgumentInputConstraints(const OpGraph& op_graph, const OpNode& op) {
  static const auto allows_argument_input = ArgumentInputConstraints();
  bool arg_in_allowed = allows_argument_input[static_cast<int>(op.op_type)];
  if (!arg_in_allowed) {
    DALI_ENFORCE(op.spec.NumInput() == op.spec.NumRegularInput(),
                 to_string(op.op_type) + " Ops do not support tensor arguments, found in " +
                     op.instance_name + " Op.");
  }
  for (const auto &arg_pair : op.spec.ArgumentInputs()) {
    auto input_idx = arg_pair.second;
    auto in_tensor = op_graph.Tensor(op.parent_tensors[input_idx]);
    // Parent node of this tensor is support op
    DALI_ENFORCE(in_tensor.producer_edge.is_support,
                 "Argument input to " + op.instance_name + " produced by non-support Op.");
  }
}

void CheckConsistentTensorEdges(const OpGraph& op_graph, const TensorNode& tensor) {
  for (auto consumer_edge : tensor.consumer_edges) {
    DALI_ENFORCE(tensor.producer_edge.is_support == consumer_edge.is_support,
                 "Use of tensor " + tensor.name +
                     " as support is mismatched between producer Op and consumer Op.");
    DALI_ENFORCE(tensor.producer_edge.storage_device == consumer_edge.storage_device,
                 "Storage device of tensor " + tensor.name +
                     " is mismatched between producer Op and consumer Op.");
  }
}

void CheckGraphConstraints(const OpGraph& op_graph) {
  for (int i = 0; i < op_graph.NumOp(); i++) {
    CheckParentConstraints(op_graph, op_graph.Node(i));
    CheckArgumentInputConstraints(op_graph, op_graph.Node(i));
  }
  for (int i = 0; i < op_graph.NumTensor(); i++) {
    CheckConsistentTensorEdges(op_graph, op_graph.Tensor(i));
  }
}

}  // namespace dali