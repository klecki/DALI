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

#ifndef DALI_PIPELINE_GRAPH_OP_GRAPH_VERIFIER_H_
#define DALI_PIPELINE_GRAPH_OP_GRAPH_VERIFIER_H_

#include <set>
#include <vector>

#include "dali/pipeline/graph/op_graph.h"

namespace dali {

std::vector<int> ArgumentInputConstraints();
std::vector<std::set<DALIOpType>> ParentOpTypeConstraints();

// NB: we could collect all the errors in graph before reporting them to user
void CheckGraphConstraints(const OpGraph &op_graph);

}  // namespace dali

#endif  // DALI_PIPELINE_GRAPH_OP_GRAPH_VERIFIER_H_