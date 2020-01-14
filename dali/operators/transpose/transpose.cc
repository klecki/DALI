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

#include "dali/kernels/common/transpose.h"
#include "dali/operators/transpose/transpose.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"


namespace dali {

#define TRANSPOSE_ALLOWED_TYPES \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, float16, double)


template<>
void Transpose<CPUBackend>::RunImpl(HostWorkspace &ws) {
  const auto& input = ws.InputRef<CPUBackend>(0);
  auto& output = ws.OutputRef<CPUBackend>(0);

  auto input_type = input.type().id();
  TYPE_SWITCH(input_type, type2id, T, TRANSPOSE_ALLOWED_TYPES, (
    for (int i = 0; i < batch_size_; i++) {
      kernels::permute(view<T>(output[i]), view<const T>(input[i]), make_span(perm_));
    }
  ),
  DALI_FAIL("Input type not supported."));


  // TypeInfo itype = input.type();
  // DALI_ENFORCE((itype.size() == 1 || itype.size() == 2 || itype.size() == 4 || itype.size() == 8),
  //     "cuTT transpose supports only [1-2-4-8] bytes types.");

  // output.set_type(itype);
  // output.SetLayout(output_layout_);

  // auto input_shape = input.tensor_shape(0);
  // DALI_ENFORCE(input_shape.size() == static_cast<int>(perm_.size()),
  //              "Transposed tensors rank should be equal to the permutation index list.");

  // if (input.IsDenseTensor()) {
  //   if (cutt_handle_ != 0) {
  //     if (input_shape != previous_iter_shape_) {
  //       cuttCheck(cuttDestroy(cutt_handle_));
  //       cutt_handle_ = 0;
  //       previous_iter_shape_ = input_shape;
  //     }
  //   } else {
  //     previous_iter_shape_ = input_shape;
  //   }
  //   auto permuted_dims = detail::Permute(input_shape, perm_);
  //   output.Resize(uniform_list_shape(batch_size_, permuted_dims));
  //   if (itype.size() == 1) {
  //     kernel::cuTTKernelBatched<uint8_t>(input, output, perm_, &cutt_handle_, ws.stream());
  //   } else if (itype.size() == 2) {
  //     kernel::cuTTKernelBatched<uint16_t>(input, output, perm_, &cutt_handle_, ws.stream());
  //   } else if (itype.size() == 4) {
  //     kernel::cuTTKernelBatched<int32_t>(input, output, perm_, &cutt_handle_, ws.stream());
  //   } else {  // itype.size() == 8
  //     kernel::cuTTKernelBatched<int64_t>(input, output, perm_, &cutt_handle_, ws.stream());
  //   }
  // } else {
  //   std::vector<TensorShape<>> tl_shape;
  //   for (int i = 0; i < batch_size_; ++i) {
  //     auto in_shape = input.tensor_shape(i);
  //     tl_shape.emplace_back(detail::Permute(in_shape, perm_));
  //   }
  //   output.Resize(tl_shape);
  //   if (itype.size() == 1) {
  //     kernel::cuTTKernel<uint8_t>(input, output, perm_, ws.stream());
  //   } else if (itype.size() == 2) {
  //     kernel::cuTTKernel<uint16_t>(input, output, perm_, ws.stream());
  //   } else if (itype.size() == 4) {
  //     kernel::cuTTKernel<int32_t>(input, output, perm_, ws.stream());
  //   } else {  // itype.size() == 8
  //     kernel::cuTTKernel<int64_t>(input, output, perm_, ws.stream());
  //   }
  // }
}

template <>
Transpose<CPUBackend>::~Transpose() noexcept {
  // meh
  // if (cutt_handle_ > 0) {
  //   auto err = cuttDestroy(cutt_handle_);
  //   if (err != CUTT_SUCCESS) {
  //     // Something terrible happened. Just quit now, before you'll loose your life or worse...
  //     std::terminate();
  //   }
  // }
}

DALI_REGISTER_OPERATOR(Transpose, Transpose<CPUBackend>, CPU);




DALI_SCHEMA(Transpose)
  .DocStr("Transpose tensor dimension to a new permutated dimension specified by `perm`.")
  .NumInput(1)
  .NumOutput(1)
  .AllowSequences()
  .SupportVolumetric()
  .AddArg("perm",
      R"code(Permutation of the dimensions of the input (e.g. [2, 0, 1]).)code",
      DALI_INT_VEC)
  .AddOptionalArg("transpose_layout",
      R"code(When set to true, the output data layout will be transposed according to perm.
Otherwise, the input layout is copied to the output)code",
      true)
  .AddOptionalArg("output_layout",
      R"code(If provided, sets output data layout, overriding any `transpose_layout` setting)code",
      "");

}  // namespace dali
