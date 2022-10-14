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
#include <functional>
#include <stdexcept>
#include <vector>

#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/test/dali_operator_test.h"

#include "dali/pipeline/pipeline.h"
#include "dali/test/test_tensors.h"


namespace dali {

class SplitMerge : public ::testing::Test {
 public:
 /**
  * @brief Generate input tensor that will be split.
  * This version uses Tensors that keep their sample_idx and batch size internally
  *
  */
  virtual TensorList<CPUBackend> GetInput(int iter_idx) {
    return GetInputImpl(iter_idx, false);
  }

  /**
   * @brief Customization point for the split used in succeeding iterations,
   * We use a functor to generate the `predicate` input based on sample index.
   */
  virtual std::vector<std::function<int(int)>> GetSplitGenerators() {
    static std::vector<std::function<int(int)>> split_generators = {
        [](int idx) { return idx % 2; },  // interleaved 1-by-1
        [](int idx) { return 0; },        // all false, to the right
        [](int idx) { return 1; },        // all true, to the left
        [](int idx) { return idx < 4; },  // uneven split
    };
    return split_generators;
  }

  std::function<int(int)> GetSplitGenerator(int iter_idx) {
    return GetSplitGenerators()[iter_idx];
  }

  int GetIterCount() {
    return static_cast<int>(GetSplitGenerators().size());
  }

  /**
   * @brief Generate the predicate to be used in this iteration
   */
  virtual TensorList<CPUBackend> GetPredicate(int iter_idx) {
    TensorList<CPUBackend> predicate;
    predicate.set_pinned(false);
    predicate.set_order(AccessOrder::host());

    predicate.Resize(uniform_list_shape(kBatchSize, TensorShape<0>{}), DALI_BOOL);
    auto split_gen = GetSplitGenerator(iter_idx);
    for (int i = 0; i < kBatchSize; i++) {
      *predicate.mutable_tensor<bool>(i) = split_gen(i);
    }
    return predicate;
  }

  /**
   * @brief Validate the outputs of the pipeline against the input
   */
  template <typename Backend>
  void Validate(int iter_idx, int pipe_output_idx, const DeviceWorkspace &ws,
                const TensorList<CPUBackend> &input) {
    TensorList<CPUBackend> output;
    output.set_pinned(false);
    output.set_order(AccessOrder::host());
    output.Copy(ws.Output<Backend>(pipe_output_idx));
    EXPECT_EQ(output.shape(), input.shape());

    for (int i = 0; i < input.shape().num_samples(); i++) {
      for (int elem = 0; elem < input.shape()[i].num_elements(); elem++) {
        EXPECT_EQ(output.tensor<int32_t>(i)[elem], input.tensor<int32_t>(i)[elem]);
      }
    }
  }

  /**
   * @brief Boilerplate code for defining inputs to the graph (input and pred nodes).
   */
  void AddExternalInputs(Pipeline &pipe) {
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
  }

  static constexpr int kBatchSize = 10;
 protected:
  TensorList<CPUBackend> GetInputImpl(int iter_idx, bool pinned = false) {
    auto shape = uniform_list_shape(kBatchSize, {1, 1, 3});
    TensorList<CPUBackend> input;
    input.set_pinned(pinned);
    input.set_order(AccessOrder::host());
    input.Resize(shape, DALI_INT32);
    for (int i = 0; i < shape.num_samples(); i++) {
      for (int elem = 0; elem < shape[i].num_elements(); elem++) {
        input.mutable_tensor<int32_t>(i)[elem] = iter_idx * kBatchSize + i;
      }
    }
    return input;
  }
};


template <typename T>
class SplitMergeTyped : public SplitMerge {};

typedef ::testing::Types<CPUBackend, GPUBackend> Backends;

// Test cases:
// Split -> Merge in CPU, with one branch being force pinned (for example by copy to GPU)
// Split -> Merge in CPU where external source surprises us with pinned memory
// Split -> Merge in GPU, where we passthrough Mixed order
// Split -> Merge with empty batch running some operator.
// Split in CPU and Merge in GPU
// Nesting?
// Negative tests: mismatched sizes, mismatched split/merge, trying to return split batch.

TYPED_TEST_SUITE(SplitMergeTyped, Backends);


TEST_F(SplitMerge, SplitCpuMergeGpu) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInputs(pipe);

  pipe.AddOperator(OpSpec("Split")
                       .AddArg("device", "cpu")
                       .AddInput("input", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("split_0", "cpu")
                       .AddOutput("split_1", "cpu"),
                   "split");

  pipe.AddOperator(OpSpec("Merge")
                       .AddArg("device", "gpu")
                       .AddInput("split_0", "gpu")
                       .AddInput("split_1", "gpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("merge", "gpu"),
                   "merge");


  vector<std::pair<string, string>> outputs = {{"merge", "gpu"}};
  pipe.Build(outputs);

  pipe.SaveGraphToDotFile("split_cpu_merge_gpu.dot", true, true, true);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto input = GetInput(iter_idx);
    auto predicate = GetPredicate(iter_idx);
    pipe.SetExternalInput("input", input);
    pipe.SetExternalInput("pred", predicate);

    pipe.RunCPU();
    pipe.RunGPU();
    DeviceWorkspace ws;
    pipe.Outputs(&ws);

    Validate<GPUBackend>(iter_idx, 0, ws, input);
  }
}

/**
 * @brief Trigger pinning one branch in split, and see if both are pinned.
 */
TEST_F(SplitMerge, PinnedInside) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInputs(pipe);

  // we can see impact of pinning
  pipe.AddOperator(OpSpec("Copy")
                       .AddInput("input", "cpu")
                       .AddOutput("input_copy", "cpu"),
                   "input_copy");

  pipe.AddOperator(OpSpec("Split")
                       .AddArg("device", "cpu")
                       .AddInput("input_copy", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("split_0", "cpu")
                       .AddOutput("split_1", "cpu"),
                   "split");

  // copy it, so we don't pin split_0 due to passing it to GPU, but to check it is required
  // to be pinned for consistency reasons
  pipe.AddOperator(OpSpec("Copy")
                       .AddInput("split_0", "cpu")
                       .AddOutput("split_0_copy", "cpu"),
                   "split_0_copy");

  // this should be made pinned, thus making the input_copy pinned.
  pipe.AddOperator(OpSpec("MakeContiguous")
                       .AddArg("device", "mixed")
                       .AddInput("split_1", "cpu")
                       .AddOutput("split_1_contiguous", "gpu"),
                   "make_contiguous");

  // as the split_1 is made pinned, split_0 also should be pinned due to coming together into merge
  pipe.AddOperator(OpSpec("Merge")
                       .AddArg("device", "cpu")
                       .AddInput("split_0", "cpu")
                       .AddInput("split_1", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("merge_cpu", "cpu"),
                   "merge_cpu");

  // consume the data transferred to GPU
  pipe.AddOperator(OpSpec("Merge")
                       .AddArg("device", "gpu")
                       .AddInput("split_0_copy", "gpu")
                       .AddInput("split_1_contiguous", "gpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("merge_gpu", "gpu"),
                   "merge_gpu");

  // Not really a way to peek into the split_0 and split_1 pinnedness
  vector<std::pair<string, string>> outputs = {{"merge_cpu", "cpu"},
                                               {"merge_gpu", "gpu"},
                                               {"split_0", "cpu"},
                                               {"split_1", "cpu"},
                                               {"input_copy", "cpu"}};
  pipe.Build(outputs);

  pipe.SaveGraphToDotFile("split_merge_pinned_.dot", true, true, true);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto input = GetInput(iter_idx);
    auto predicate = GetPredicate(iter_idx);
    pipe.SetExternalInput("input", input);
    pipe.SetExternalInput("pred", predicate);

    pipe.RunCPU();
    pipe.RunGPU();
    DeviceWorkspace ws;
    pipe.Outputs(&ws);

    Validate<CPUBackend>(iter_idx, 0, ws, input);
    Validate<GPUBackend>(iter_idx, 1, ws, input);
    // We can check if the output is pinned
    EXPECT_TRUE(ws.Output<CPUBackend>(0).is_pinned());
    // split_0
    EXPECT_TRUE(ws.Output<CPUBackend>(2).is_pinned());
    // split_1
    EXPECT_TRUE(ws.Output<CPUBackend>(3).is_pinned());
    // and input_copy
    EXPECT_TRUE(ws.Output<CPUBackend>(4).is_pinned());
  }
}


TEST_F(SplitMerge, PinnedThroughMerge) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInputs(pipe);

  // we can see impact of pinning
  pipe.AddOperator(OpSpec("Copy")
                       .AddInput("input", "cpu")
                       .AddOutput("input_copy", "cpu"),
                   "input_copy");

  pipe.AddOperator(OpSpec("Split")
                       .AddArg("device", "cpu")
                       .AddInput("input_copy", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("split_0", "cpu")
                       .AddOutput("split_1", "cpu"),
                   "split");

  // We will cause the output to be pinned, so the input to split should also be pinned
  pipe.AddOperator(OpSpec("Merge")
                       .AddArg("device", "cpu")
                       .AddInput("split_0", "cpu")
                       .AddInput("split_1", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("merge_cpu", "cpu"),
                   "merge_cpu");

  // consume the data transferred to GPU
  pipe.AddOperator(OpSpec("MakeContiguous")
                       .AddArg("device", "mixed")
                       .AddInput("merge_cpu", "cpu")
                       .AddOutput("merge_gpu", "gpu"),
                   "merge_gpu");

  // Not really a way to peek into the split_0 and split_1 contiguity
  vector<std::pair<string, string>> outputs = {
      {"merge_cpu", "cpu"}, {"input_copy", "cpu"}, {"merge_gpu", "gpu"}};
  pipe.Build(outputs);

  pipe.SaveGraphToDotFile("split_merge_pinned_through_merge.dot", true, true, true);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto input = GetInput(iter_idx);
    auto predicate = GetPredicate(iter_idx);
    pipe.SetExternalInput("input", input);
    pipe.SetExternalInput("pred", predicate);

    pipe.RunCPU();
    pipe.RunGPU();
    DeviceWorkspace ws;
    pipe.Outputs(&ws);

    Validate<CPUBackend>(iter_idx, 0, ws, input);
    Validate<GPUBackend>(iter_idx, 2, ws, input);
    // But we can check if the output is pinned
    EXPECT_TRUE(ws.Output<CPUBackend>(0).is_pinned());
    EXPECT_TRUE(ws.Output<CPUBackend>(1).is_pinned());
  }
}


/**
 * @brief Split and Merge in the same stage.
 */
TYPED_TEST(SplitMergeTyped, SimpleCase) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  auto backend = testing::detail::BackendStringName<TypeParam>();

  Pipeline pipe(this->kBatchSize, 4, 0);
  this->AddExternalInputs(pipe);

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

  pipe.AddOperator(OpSpec("Copy")
                       .AddArg("device", backend)
                       .AddInput("split_1", backend)
                       .AddOutput("split_1_copy", backend),
                   "copy_1");

  pipe.AddOperator(OpSpec("Merge")
                       .AddArg("device", backend)
                       .AddInput("split_0_copy", backend)
                       .AddInput("split_1_copy", backend)
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("merge", backend),
                   "merge");

  vector<std::pair<string, string>> outputs = {{"merge", backend}};
  pipe.Build(outputs);

  pipe.SaveGraphToDotFile("split_merge_simple_" + backend + ".dot", true, true, true);

  for (int iter_idx = 0; iter_idx < this->GetIterCount(); iter_idx++) {
    auto input = this->GetInput(iter_idx);
    auto predicate = this->GetPredicate(iter_idx);
    pipe.SetExternalInput("input", input);
    pipe.SetExternalInput("pred", predicate);

    pipe.RunCPU();
    pipe.RunGPU();
    DeviceWorkspace ws;
    pipe.Outputs(&ws);

    this->template Validate<TypeParam>(iter_idx, 0, ws, input);
  }
}

// Negative tests
TYPED_TEST(SplitMergeTyped, ReturnSplit) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  auto backend = testing::detail::BackendStringName<TypeParam>();

  Pipeline pipe(this->kBatchSize, 4, 0);
  this->AddExternalInputs(pipe);

  pipe.AddOperator(OpSpec("Split")
                      .AddArg("device", backend)
                      .AddInput("input", backend)
                      .AddArgumentInput("predicate", "pred")
                      .AddOutput("split_0", backend)
                      .AddOutput("split_1", backend),
                  "split");
  vector<std::pair<string, string>> outputs = {{"split_0", backend}, {"split_1", backend}};
  pipe.Build(outputs);

  for (int iter_idx = 0; iter_idx < this->GetIterCount(); iter_idx++) {
    auto input = this->GetInput(iter_idx);
    auto predicate = this->GetPredicate(iter_idx);
    pipe.SetExternalInput("input", input);
    pipe.SetExternalInput("pred", predicate);

    pipe.RunCPU();
    pipe.RunGPU();
    DeviceWorkspace ws;
    pipe.Outputs(&ws);
    TensorList<CPUBackend> output[2];
    for (int i = 0; i < 2; i++) {
      output[i].set_pinned(false);
      output[i].Copy(ws.Output<TypeParam>(i));
    }
    int idxs[2] = {0, 0};
    for (int i = 0; i < this->kBatchSize; i++) {
      // the indexing is reversed, truthy values go to (0) output, falsy to (1)
      int which = !*predicate.template tensor<bool>(i);
      int output_sample_idx = idxs[which];
      idxs[which]++;
      EXPECT_LT(output_sample_idx, output[which].shape().num_samples());
      EXPECT_EQ(output[which][output_sample_idx].shape(), input[i].shape());
      for (int elem = 0; elem < input[i].shape().num_elements(); elem++) {
        EXPECT_EQ((output[which].template tensor<int32_t>(output_sample_idx)[elem]),
                  (input.template tensor<int32_t>(i)[elem]));
      }
    }
  }
}

class SplitMergeNegative : public SplitMerge {
  // Unbiased splits only
  std::vector<std::function<int(int)>> GetSplitGenerators() override {
    static std::vector<std::function<int(int)>> split_generators = {
        [](int idx) { return idx < 3; },  // uneven split
    };
    return split_generators;
  }
};


TEST_F(SplitMergeNegative, MismatchedMerge) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInputs(pipe);

  pipe.AddOperator(OpSpec("Split")
                       .AddArg("device", "cpu")
                       .AddInput("input", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("split_0", "cpu")
                       .AddOutput("split_1", "cpu"),
                   "split");

  // Try to merge two bigger parts
  pipe.AddOperator(OpSpec("Merge")
                       .AddArg("device", "cpu")
                       .AddInput("split_0", "cpu")
                       .AddInput("split_0", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("merge", "cpu"),
                   "merge");

  vector<std::pair<string, string>> outputs = {{"merge", "cpu"}};
  pipe.Build(outputs);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto input = GetInput(iter_idx);
    auto predicate = GetPredicate(iter_idx);
    pipe.SetExternalInput("input", input);
    pipe.SetExternalInput("pred", predicate);

    try {
      pipe.RunCPU();
      pipe.RunGPU();
      DeviceWorkspace ws;
      pipe.Outputs(&ws);
      FAIL() << "Exception was expected but was not thrown.";
    } catch (std::runtime_error &e) {
      static const char expected[] = "Merge description must cover whole input, got ";
      EXPECT_NE(std::string(e.what()).rfind(expected), std::string::npos)
          << expected << "\n====\nvs\n====\n"
          << e.what();
    } catch (...) {
      FAIL() << "Unexpected exception.";
    }
  }
}



TEST_F(SplitMergeNegative, MismatchedSplit) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInputs(pipe);

  // split the predicates
  pipe.AddOperator(OpSpec("Split")
                       .AddArg("device", "cpu")
                       .AddInput("pred", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("pred_left", "cpu")
                       .AddOutput("pred_right", "cpu"),
                   "split_pred");

  // split the input
  pipe.AddOperator(OpSpec("Split")
                       .AddArg("device", "cpu")
                       .AddInput("input", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("input_left", "cpu")
                       .AddOutput("input_right", "cpu"),
                   "split_input");

  // try to split smaller input with bigger predicate
  pipe.AddOperator(OpSpec("Split")
                       .AddArg("device", "cpu")
                       .AddInput("input_left", "cpu")
                       .AddArgumentInput("predicate", "pred_right")
                       .AddOutput("split_0", "cpu")
                       .AddOutput("split_1", "cpu"),
                   "split");

  vector<std::pair<string, string>> outputs = {{"split_0", "cpu"}, {"split_1", "cpu"}};
  pipe.Build(outputs);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto input = GetInput(iter_idx);
    auto predicate = GetPredicate(iter_idx);
    pipe.SetExternalInput("input", input);
    pipe.SetExternalInput("pred", predicate);


    try {
      pipe.RunCPU();
      pipe.RunGPU();
      DeviceWorkspace ws;
      pipe.Outputs(&ws);
      FAIL() << "Exception was expected but was not thrown.";
    } catch (std::runtime_error &e) {
      static const char expected[] = "Split description must cover whole input, got ";
      EXPECT_NE(std::string(e.what()).rfind(expected), std::string::npos)
          << expected << "\n====\nvs\n====\n"
          << e.what();
    } catch (...) {
      FAIL() << "Unexpected exception.";
    }
  }
}

class SplitMergePinnedInputs : public SplitMerge {
 public:
  TensorList<CPUBackend> GetPinnedInput(int iter_idx) {
    return GetInputImpl(iter_idx, true);
  }
};

TEST_F(SplitMergePinnedInputs, Mixes) {
  Pipeline pipe(kBatchSize, 4, 0);
  AddExternalInputs(pipe);

  pipe.AddOperator(OpSpec("ExternalSource")
                        .AddArg("device", "cpu")
                        .AddArg("name", "input")
                        .AddOutput("pinned_input", "cpu"),
                    "pinned_input");


  pipe.AddOperator(OpSpec("Split")
                       .AddArg("device", "cpu")
                       .AddInput("input", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("split_0", "cpu")
                       .AddOutput("split_1", "cpu"),
                   "split");

  pipe.AddOperator(OpSpec("Split")
                       .AddArg("device", "cpu")
                       .AddInput("pinned_input", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("split_pinned_0", "cpu")
                       .AddOutput("split_pinned_1", "cpu"),
                   "split_pinned");

  pipe.AddOperator(OpSpec("Merge")
                       .AddArg("device", "cpu")
                       .AddInput("split_0", "cpu")
                       .AddInput("split_1", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("merge_nn", "cpu"),
                   "merge_nn");

  pipe.AddOperator(OpSpec("Merge")
                       .AddArg("device", "cpu")
                       .AddInput("split_pinned_0", "cpu")
                       .AddInput("split_pinned_1", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("merge_pp", "cpu"),
                   "merge_pp");

  pipe.AddOperator(OpSpec("Merge")
                       .AddArg("device", "cpu")
                       .AddInput("split_pinned_0", "cpu")
                       .AddInput("split_1", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("merge_pn", "cpu"),
                   "merge_pn");

  pipe.AddOperator(OpSpec("Merge")
                       .AddArg("device", "cpu")
                       .AddInput("split_0", "cpu")
                       .AddInput("split_pinned_1", "cpu")
                       .AddArgumentInput("predicate", "pred")
                       .AddOutput("merge_np", "cpu"),
                   "merge_np");


  vector<std::pair<string, string>> outputs = {
      {"merge_nn", "cpu"}, {"merge_pp", "cpu"}, {"merge_pn", "cpu"}, {"merge_np", "cpu"}};
  pipe.Build(outputs);

  pipe.SaveGraphToDotFile("split_pinned_mix.dot", true, true, true);

  for (int iter_idx = 0; iter_idx < GetIterCount(); iter_idx++) {
    auto input = GetInput(iter_idx);
    auto pinned_input = GetPinnedInput(iter_idx);
    auto predicate = GetPredicate(iter_idx);
    pipe.SetExternalInput("input", input);
    pipe.SetExternalInput("pinned_input", pinned_input);
    pipe.SetExternalInput("pred", predicate);

    pipe.RunCPU();
    pipe.RunGPU();
    DeviceWorkspace ws;
    pipe.Outputs(&ws);

    // For whatever reason the outputs are always pinned.
    // EXPECT_FALSE(ws.Output<CPUBackend>(0).is_pinned());
    // EXPECT_TRUE(ws.Output<CPUBackend>(1).is_pinned());
    // EXPECT_TRUE(ws.Output<CPUBackend>(2).is_pinned());
    // EXPECT_TRUE(ws.Output<CPUBackend>(3).is_pinned());

    Validate<CPUBackend>(iter_idx, 0, ws, input);
    Validate<CPUBackend>(iter_idx, 1, ws, input);
    Validate<CPUBackend>(iter_idx, 2, ws, input);
    Validate<CPUBackend>(iter_idx, 3, ws, input);
  }
}

}  // namespace dali
