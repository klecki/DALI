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

#include <gtest/gtest.h>
#include <random>
#include "dali/operators/geometric/shapes.h"
#include "dali/test/dali_operator_test.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {

template <typename Backend, typename RNG>
void GenerateShapeTestInputs(TensorList<Backend> &out, RNG &rng, int num_samples, int sample_dim) {
  TensorListShape<> shape;
  // this should give a distribution such that the batch is no bigger than 1e+8 elements
  int max = std::ceil(std::pow(1e+6 / num_samples, 1.0 / sample_dim));
  std::uniform_int_distribution<int> dist(1, max);

  shape.resize(num_samples, sample_dim);
  for (int i = 0; i < num_samples; i++) {
    for (int j = 0; j < sample_dim; j++) {
      shape.tensor_shape_span(i)[j] = dist(rng);
    }
  }

  TensorLayout out_layout;
  std::string tmp = "A";
  for (int i = 0; i < sample_dim; i++) {
    out_layout = out_layout + TensorLayout(tmp);
    tmp[0]++;
  }
  out.Reset();
  out.Resize(shape);
  out.SetLayout(out_layout);
  (void)out.template mutable_data<uint8_t>();
}

template <typename OutputBackend, typename InputBackend, typename OutputType>
struct ShapesTestArgs {
  using out_backend = OutputBackend;
  using in_backend = InputBackend;
  using output_type = OutputType;

  static DALIDataType type_id() {
    return TypeTable::GetTypeID<output_type>();
  }
};

template <typename TestArgs>
class ShapesOpTest;

template <typename Backend>
static std::vector<std::unique_ptr<TensorList<Backend>>> inputs;


template <typename OutputBackend, typename InputBackend, typename OutputType>
class ShapesOpTest<ShapesTestArgs<OutputBackend, InputBackend, OutputType>>
: public testing::DaliOperatorTest {
 public:
  using TestArgs = ShapesTestArgs<OutputBackend, InputBackend, OutputType>;

  ShapesOpTest() {
    if (inputs<InputBackend>.empty()) {
      std::mt19937_64 rng(12345);
      for (int dim = 1; dim <= max_dim; dim++) {
        inputs<InputBackend>.emplace_back(new TensorList<InputBackend>());
        inputs<InputBackend>.back()->set_pinned(false);
        int num_samples = 1 << (8-dim);  // Start with 128 and halve with each new dimension
        GenerateShapeTestInputs(*inputs<InputBackend>.back(), rng, num_samples, dim);
      }
    }
  }

  testing::GraphDescr GenerateOperatorGraph() const override {
      return {"Shapes"};
  }

  void Run(const testing::Arguments &additional_args = {}, int min_dim = 1) {
    testing::Arguments args;
    args.emplace("type", TestArgs::type_id());
    args.emplace("device", testing::detail::BackendStringName<OutputBackend>());
    for (const auto &additional_arg : additional_args) {
      args.emplace(additional_arg);
    }
    for (int i = min_dim - 1; i < max_dim; i++) {
      testing::TensorListWrapper out;
      this->RunTest(inputs<InputBackend>[i].get(), out, args, VerifyShape);
    }
  }

  static void VerifyShape(
      const testing::TensorListWrapper &in_wrapper,
      const testing::TensorListWrapper &out_wrapper,
      const testing::Arguments &args) {
    ASSERT_TRUE(in_wrapper.has<InputBackend>());
    ASSERT_TRUE(out_wrapper.has<OutputBackend>());
    auto &in = *in_wrapper.get<InputBackend>();
    auto &out = *out_wrapper.get<OutputBackend>();
    VerifyShapeImpl(in, out, args);
  }

  static void VerifyShapeImpl(
      const TensorList<CPUBackend> &in,
      const TensorList<GPUBackend> &out,
      const testing::Arguments &args) {
    TensorList<CPUBackend> tmp;
    tmp.Copy(out, 0);
    cudaDeviceSynchronize();
    VerifyShapeImpl(in, tmp, args);
  }

  static void VerifyShapeImpl(
      const TensorList<CPUBackend> &in,
      const TensorList<CPUBackend> &out,
      const testing::Arguments &args) {
    auto shape = in.shape();
    auto out_shape = out.shape();
    std::vector<int> output_axes;
    if (args.find("axes") != args.end()) {
      output_axes = args.at("axes").GetValue<std::vector<int>>();
    } else if (args.find("axis_names") != args.end()) {
      auto axes = args.at("axis_names").GetValue<TensorLayout>();
      auto layout = in.GetLayout();
      for (const auto &axis : axes) {
        output_axes.push_back(layout.find(axis));
      }
    } else {
      output_axes.resize(shape.sample_dim());
      std::iota(output_axes.begin(), output_axes.end(), 0);
    }
    const int N = shape.num_samples();
    const int D = output_axes.size();
    ASSERT_EQ(N, out_shape.num_samples());
    ASSERT_TRUE(is_uniform(out_shape));
    ASSERT_EQ(out_shape.sample_dim(), 1);
    ASSERT_EQ(out_shape[0][0], D);

    for (int i = 0; i < N; i++) {
      const OutputType *shape_data = out.template tensor<OutputType>(i);
      auto tshape = shape.tensor_shape_span(i);
      for (int j = 0; j < D; j++) {
        EXPECT_EQ(shape_data[j], static_cast<OutputType>(tshape[output_axes[j]]));
      }
    }
  }
  static constexpr int max_dim = 6;
};

using ShapesOpArgs = ::testing::Types<
  ShapesTestArgs<CPUBackend, CPUBackend, int32_t>,
  ShapesTestArgs<CPUBackend, CPUBackend, uint32_t>,
  ShapesTestArgs<CPUBackend, CPUBackend, int64_t>,
  ShapesTestArgs<CPUBackend, CPUBackend, uint64_t>,
  ShapesTestArgs<CPUBackend, CPUBackend, float>,
  ShapesTestArgs<CPUBackend, CPUBackend, double>,

  ShapesTestArgs<GPUBackend, CPUBackend, int32_t>,
  ShapesTestArgs<GPUBackend, CPUBackend, uint32_t>,
  ShapesTestArgs<GPUBackend, CPUBackend, int64_t>,
  ShapesTestArgs<GPUBackend, CPUBackend, uint64_t>,
  ShapesTestArgs<GPUBackend, CPUBackend, float>,
  ShapesTestArgs<GPUBackend, CPUBackend, double>>;

TYPED_TEST_SUITE(ShapesOpTest, ShapesOpArgs);

TYPED_TEST(ShapesOpTest, AllNoArgs) {
  this->Run();
}

TYPED_TEST(ShapesOpTest, EmptyAxes) {
  ASSERT_THROW(this->Run({{"axes", std::vector<int>{}}}), std::runtime_error);
}

TYPED_TEST(ShapesOpTest, OneAxis) {
  this->Run({{"axes", std::vector<int>{1}}}, 2);
}

TYPED_TEST(ShapesOpTest, Axes) {
  this->Run({{"axes", std::vector<int>{0, 1}}}, 2);
}

TYPED_TEST(ShapesOpTest, AxesReorder) {
  this->Run({{"axes", std::vector<int>{1, 0, 2}}}, 3);
}

TYPED_TEST(ShapesOpTest, WrongAxes) {
  ASSERT_THROW(this->Run({{"axes", std::vector<int>{0, 16}}}, 2), std::runtime_error);
}

TYPED_TEST(ShapesOpTest, EmptyAxisNames) {
  ASSERT_THROW(this->Run({{"axis_names", TensorLayout{}}}), std::runtime_error);
}

TYPED_TEST(ShapesOpTest, OneAxisName) {
  this->Run({{"axis_names", TensorLayout{"A"}}}, 2);
}

TYPED_TEST(ShapesOpTest, AxisNames) {
  this->Run({{"axis_names", TensorLayout{"AB"}}}, 2);
}

TYPED_TEST(ShapesOpTest, AxisNamesReorder) {
  this->Run({{"axis_names", TensorLayout{"CAB"}}}, 3);
}

TYPED_TEST(ShapesOpTest, WrongAxisNames) {
  ASSERT_THROW(this->Run({{"axis_names", TensorLayout{"ZW"}}}, 2), std::runtime_error);
}

}  // namespace dali
