// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/workspace/mixed_workspace.h"

#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {

int MixedWorkspace::NumInputAtIdx(int idx) const {
  DALI_ENFORCE_VALID_INDEX(idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  if (tensor_meta.storage_device == StorageDevice::CPU) {
    return cpu_inputs_[tensor_meta.index]->size();
  }
  return gpu_inputs_[tensor_meta.index]->size();
}

template <>
const Tensor<CPUBackend>& MixedWorkspace::Input(int idx, int data_idx) const {
  return *(*InputHandle<CPUBackend>(idx))[data_idx];
}

template <>
const Tensor<GPUBackend>& MixedWorkspace::Input(int idx, int data_idx) const {
  return *(*InputHandle<GPUBackend>(idx))[data_idx];
}

template <>
TensorList<CPUBackend>& MixedWorkspace::Output(int idx) {
  return *OutputHandle<CPUBackend>(idx);
}

template <>
TensorList<GPUBackend>& MixedWorkspace::Output(int idx) {
  return  *OutputHandle<GPUBackend>(idx);
}

}  // namespace dali
