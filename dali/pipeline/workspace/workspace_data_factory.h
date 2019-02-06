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

#ifndef DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_
#define DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_

#include <memory>

#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/host_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"
#include "dali/pipeline/workspace/support_workspace.h"

namespace dali {

template <DALIOpType op_type, DALITensorDevice device>
struct BatchFactory {
  workspace_output_t<op_type, storage_t<device>> CreateOutputBatch(int batch_size) {
    // Output batch from GPU, MIXED and SUPPORT Ops are shared_ptr<Something>
    using BatchType = typename workspace_output_t<op_type, storage_t<device>>::element_type;
    return std::make_shared<BatchType>();
  }
  static_assert(
      op_type == DALIOpType::DALI_GPU || op_type == DALIOpType::DALI_MIXED ||
          op_type == DALIOpType::DALI_SUPPORT,
      "Only GPU, MIXED and SUPPORT handled by default case due to use of outermost shared_ptr");
};


// template <DALITensorDevice device>
// workspace_output_t<DALIOpType::DALI_CPU, storage_t<device>> BatchFactory<DALIOpType::DALI_CPU, device>::CreateOutputBatch(int batch_size) {

// }

}

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_