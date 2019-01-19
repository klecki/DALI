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

#include "dali/test/dali_operator_test.h"
#include "dali/pipeline/operators/util/reshape.h"

namespace dali {
class ReshapeTest : public testing::DaliOperatorTest {
  GraphDescr GenerateOperatorGraph() const noexcept override {
    GraphDescr graph("Reshape");
    return graph;
  }

 public:
  ReshapeTest() : DaliOperatorTest(1, 1) {}
  // I need sequentiall fill of that data
  // Where do I define input data generation?
  TensorListWrapper getInputContigous() = 0; // Return 1-Dim 120 elements Tensor / (TensorList of such Tensors)
  TensorListWrapper getInputShaped() = 0; // Return 2x3x20 Tensors
  TensorListWrapper getInputDesc() = 0; // Return 1D tensors with new shapes as in standard_reshapes below
};

std::vector<Arguments> standard_reshapes = {
      {{"new_shape", std::vector<Index>{120}}},
      {{"new_shape", std::vector<Index>{2, 60}}},
      {{"new_shape", std::vector<Index>{2, 3, 20}}},
      {{"new_shape", std::vector<Index>{2, 3, 4, 5}}},
};

std::vector<Arguments> input_reshape = {
      {{"new_shape", std::vector<Index>{-1}}},
      {{}} // no-arg?
};

std::vector<Arguments> wrong_reshape = {
      {{"new_shape", std::vector<Index>{121}}},
      {{"new_shape", std::vector<Index>{3, 60}}},
      {{"new_shape", std::vector<Index>{-1, 3, 20}}},
      {{"new_shape", std::vector<Index>{2, -3, 4, 5}}},
      {{"new_shape", std::vector<Index>{2, -3, -4, 5}}},
};

// TODO unify Verify
void Verify(TensorListWrapper input, TensorListWrapper output, Arguments args) {
  auto in = ToTensorListView(input);
  auto out = ToTensorListView(output);
  // for each element:
  ASSERT_EQ(Product(in.shape()), Product(out.shape()));
  EXPECT_EQ(out.shape(), args[new_shape]);
  // compare in.data() and out.data() elementwise
}

void Verify_2arg(std::vector<TensorListWrapper> inputs, TensorListWrapper output, Arguments args) {
  auto in = ToTensorListView(inputs[0]);
  auto shape = ToTensorListView(inputs[1]);
  auto out = ToTensorListView(output);
  // for each element:
  ASSERT_EQ(Product(in.shape()), Product(out.shape()));
  EXPECT_EQ(out.shape(), shape);
  // compare in.data() and out.data() elementwise
}

TEST_P(ReshapeTest, ContigousInTest) {
  auto args = GetParam();
  TensorListWrapper tlout; // todo, whats with that out?
  this->RunTest<CPUBackend>(getInputContigous(), tlout, args, Verify);
}


TEST_P(ReshapeTest, ShapedInTest) {
  auto args = GetParam();
  TensorListWrapper tlout; // todo, whats with that out?
  this->RunTest<CPUBackend>(getInputShaped(), tlout, args, Verify);
}

TEST_P(ReshapeTest, AsInputTest) {
  auto args = GetParam();
  TensorListWrapper tlout; // todo, whats with that out?
  this->RunTest<CPUBackend>({getInputShaped(), getInputDesc()}, tlout, args, Verify_2arg);
}

TEST_P(ReshapeTest, WrongTest) {
  auto args = GetParam();
  TensorListWrapper tlout; // todo, whats with that out?
  ASSERT_THROW(this->RunTest<CPUBackend>(getInputContigous(), tlout, args, Verify));
}

INSTANTIATE_TEST_CASE_P(ReshapeTest, ContigousInTest, ::testing::ValuesIn(standard_reshapes));
INSTANTIATE_TEST_CASE_P(ReshapeTest, ShapedInTest, ::testing::ValuesIn(standard_reshapes));
INSTANTIATE_TEST_CASE_P(ReshapeTest, AsInputTest, ::testing::ValuesIn(input_reshape));
INSTANTIATE_TEST_CASE_P(ReshapeTest, WrongTest, ::testing::ValuesIn(wrong_reshape));
}  // namespace dali
