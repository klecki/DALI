// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <gtest/gtest.h>

#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/types.h"
#include "dali/test/dali_operator_test.h"

#include "dali/pipeline/pipeline.h"


namespace dali {

template <typename T>
class SplitMerge : public ::testing::Test {};

typedef ::testing::Types<CPUBackend, GPUBackend> Backends;

TYPED_TEST_SUITE(SplitMerge, Backends);

TYPED_TEST(SplitMerge, SimplePipe) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;

  auto backend = testing::detail::BackendStringName<TypeParam>();

  auto shape = uniform_list_shape(10, {10, 5, 3});

  Pipeline pipe(shape.num_samples(), 4, 0);


  pipe.AddOperator(OpSpec("ExternalSource")
                       .AddArg("device", "cpu")
                       .AddArg("name", "input")
                       .AddOutput("input", "cpu"),
                   "input");

  pipe.AddOperator(OpSpec("ExternalSource")
                       .AddArg("device", "cpu")
                       .AddArg("name", "pred")
                       .AddOutput("pred", "cpu"),
                   "pred");


  pipe.AddOperator(OpSpec("Split")
                       .AddArg("device", backend)
                       .AddInput("input", backend)
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("split_0", backend)
                       .AddOutput("split_1", backend),
                   "split");


  pipe.AddOperator(OpSpec("Copy")
                       .AddArg("device", backend)
                       .AddInput("split_0", backend)
                       .AddOutput("split_0_copy", backend),
                   "copy_0");


  pipe.AddOperator(OpSpec("Merge")
                       .AddArg("device", backend)
                       .AddInput("split_0_copy", backend)
                       .AddInput("split_1", backend)
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("merge", backend),
                   "merge");

  // TODO(klecki): why did we not add MakeContiguous at the end? We did wrong pass through.
  vector<std::pair<string, string>> outputs = {{"merge", backend}};
  pipe.Build(outputs);

  pipe.SaveGraphToDotFile("split_merge.dot", true, true, true);

  TensorList<CPUBackend> input, predicate;
  input.set_pinned(false);
  predicate.set_pinned(false);
  input.Resize(shape, DALI_INT32);
  for (int i = 0; i < shape.num_samples(); i++) {
    for (int elem = 0; elem < shape[i].num_elements(); elem++) {
      input.mutable_tensor<int32_t>(i)[elem] = i;
    }
  }
  predicate.Resize(uniform_list_shape(10, TensorShape<0>{}), DALI_BOOL);

  for (int i = 0; i < 10; i++) {
    *predicate.mutable_tensor<bool>(i) = i % 2;
  }


  pipe.SetExternalInput("input", input);
  pipe.SetExternalInput("pred", predicate);

  pipe.RunCPU();
  pipe.RunGPU();
  DeviceWorkspace ws;
  pipe.Outputs(&ws);
}

}  // namespace dali
