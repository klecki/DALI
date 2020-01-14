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

#ifndef DALI_KERNELS_COMMON_TRANSPOSE_H_
#define DALI_KERNELS_COMMON_TRANSPOSE_H_

#include <utility>
#include <dbg.h>

#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"

namespace dali {
namespace kernels {

// template <typename Dst, typename Src>
// void permute(const OutTensorCPU<Dst,-1> &out, const InTensorCPU<Src,-1> &in, span<const int>
// axis_map);

namespace detail {
template <typename Dst, typename Src>
void permute_recurse(Dst *dst, const Src *src, int level, int max_levels,
                     span<const ptrdiff_t> dst_stride, span<const ptrdiff_t> src_stride,
                     TensorShape<> size, span<const int> perm) {
  auto dst_level_stride = dst_stride[level];
  auto src_level_stride = src_stride[perm[level]];

  if (level == max_levels - 1) {
    for (int i = 0; i < size[level]; i++) {
      *dst = *src;
      dst += dst_level_stride;
      src += src_level_stride;
    }
    return;
  }

  for (int i = 0; i < size[level]; i++) {
    permute_recurse(dst, src, level + 1, max_levels, dst_stride, src_stride, size, perm);
    dst += dst_level_stride;
    src += src_level_stride;
  }
}
}  // namespace detail

constexpr bool kDstMajor = true;

template <typename Dst, typename Src>
void permute(const TensorView<StorageCPU, Dst> &dst, const TensorView<StorageCPU, const Src> &src,
             span<const int> perm) {
  int N = src.shape.sample_dim();
  assert(dst.shape.sample_dim() == N);
  assert(volume(src.shape) == volume(dst.shape));
  auto dst_stride_tmp = GetStrides(dst.shape);
  auto src_stride_tmp = GetStrides(src.shape);
  std::vector<ptrdiff_t> dst_strides(N), src_strides(N);
  if (kDstMajor) {
    // permute dst strides
    for (int i = 0; i < perm.size(); i++) {
      dst_strides[i] = dst_stride_tmp[i];  // hmm, weird, it's already defined
      src_strides[i] = src_stride_tmp[i];
    }
    detail::permute_recurse(dst.data, src.data, 0, N, make_span(dst_strides), make_span(src_strides), dst.shape, perm);
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_TRANSPOSE_H_
