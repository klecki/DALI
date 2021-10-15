// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/workspace/host_workspace.h"

#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {

template <>
const Tensor<CPUBackend>& HostWorkspace::Input(int idx, int data_idx) const {
  return InputRef<CPUBackend>(idx)[data_idx];
}

template <>
const Tensor<GPUBackend>& HostWorkspace::Input(int idx, int data_idx) const {
  return InputRef<GPUBackend>(idx)[data_idx];
}

template <>
Tensor<CPUBackend>& HostWorkspace::Output(int idx, int data_idx) {
  return OutputRef<CPUBackend>(idx)[data_idx];
}

template <>
Tensor<GPUBackend>& HostWorkspace::Output(int idx, int data_idx) {
  return OutputRef<GPUBackend>(idx)[data_idx];
}

}  // namespace dali
