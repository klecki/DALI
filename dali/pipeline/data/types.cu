// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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



// #define DALI_TYPENAME_REGISTERER(Type, dtype)                                    \
// {                                                                                \
//   return to_string(dtype);                                                       \
// }

// #define DALI_TYPEID_REGISTERER(Type, dtype)                                      \
// {                                                                                \
//   static DALIDataType type_id = TypeTable::instance().RegisterType<Type>(dtype); \
//   return type_id;                                                                \
// }

// #define DALI_REGISTER_TYPE_IMPL(Type, Id) \
// const auto &_type_info_##Id = TypeTable::GetTypeID<Type>()

#include "dali/pipeline/data/types.h"
// #include "dali/core/float16.h"
#include <cuda_fp16.h>  // for __half & related methods

namespace dali {
namespace detail {

__global__ void CopyKernel(uint8_t *dst, const uint8_t *src, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    dst[i] = src[i];
  }
}

void LaunchCopyKernel(void *dst, const void *src, int64_t nbytes, cudaStream_t stream) {
  unsigned block = std::min<int64_t>(nbytes, 1024);
  unsigned grid = std::min<int64_t>(1024, div_ceil(static_cast<unsigned>(nbytes), block));
  CopyKernel<<<grid, block, 0, stream>>>(reinterpret_cast<uint8_t*>(dst),
                                         reinterpret_cast<const uint8_t*>(src),
                                         nbytes);
  CUDA_CALL(cudaGetLastError());
}

__host__ __half __int2half_rn(int i) {
  return {};
}

}  // namespace detail
}  // namespace dali
