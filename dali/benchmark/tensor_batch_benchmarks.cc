// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// distributed under the License is distributed on an "AS IS" BASIS,
// See the License for the specific language governing permissions and
// limitations under the License.

#include <benchmark/benchmark.h>

#include <memory>
#include <vector>
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/tensor_vector.h"
#include "dali/core/tensor_shape.h"


namespace dali {

static constexpr int max_range = 1 << 8;

static void batch_tl_resize_fit_cpu(benchmark::State& state) {
  int num_samples = state.range(0);
  auto full = uniform_list_shape<3>(state.range(0), {1024, 1024, 3});
  std::vector<TensorListShape<>> shapes(3);
  shapes[0] = uniform_list_shape<3>(state.range(0), {800, 1024, 3});
  shapes[1] = uniform_list_shape<3>(state.range(0), {600, 800, 3});
  shapes[2] = uniform_list_shape<3>(state.range(0), {800, 800, 3});
  TensorList<CPUBackend> batch;
  batch.Resize(full, DALI_UINT32);
  int i = 0;
  for (auto _ : state) {
    batch.Resize(shapes[i]);
    i = (i + 1) % 3;
  }
}
BENCHMARK(batch_tl_resize_fit_cpu)->RangeMultiplier(2)->Range(2, max_range);

static void batch_tl_resize_fit_gpu(benchmark::State& state) {
  int num_samples = state.range(0);
  auto full = uniform_list_shape<3>(state.range(0), {1024, 1024, 3});
  std::vector<TensorListShape<>> shapes(3);
  shapes[0] = uniform_list_shape<3>(state.range(0), {800, 1024, 3});
  shapes[1] = uniform_list_shape<3>(state.range(0), {600, 800, 3});
  shapes[2] = uniform_list_shape<3>(state.range(0), {800, 800, 3});
  TensorList<GPUBackend> batch;
  batch.Resize(full, DALI_UINT32);
  int i = 0;
  for (auto _ : state) {
    batch.Resize(shapes[i]);
    i = (i + 1) % 3;
  }
}
BENCHMARK(batch_tl_resize_fit_gpu)->RangeMultiplier(2)->Range(2, max_range);



static void batch_tv_resize_fit_cpu(benchmark::State& state) {
  int num_samples = state.range(0);
  auto full = uniform_list_shape<3>(state.range(0), {1024, 1024, 3});
  std::vector<TensorListShape<>> shapes(3);
  shapes[0] = uniform_list_shape<3>(state.range(0), {800, 1024, 3});
  shapes[1] = uniform_list_shape<3>(state.range(0), {600, 800, 3});
  shapes[2] = uniform_list_shape<3>(state.range(0), {800, 800, 3});
  TensorVector<CPUBackend> batch;
  batch.Resize(full, DALI_UINT32);
  int i = 0;
  for (auto _ : state) {
    batch.Resize(shapes[i]);
    i = (i + 1) % 3;
  }
}
BENCHMARK(batch_tv_resize_fit_cpu)->RangeMultiplier(2)->Range(2, max_range);

static void batch_tv_resize_fit_gpu(benchmark::State& state) {
  int num_samples = state.range(0);
  auto full = uniform_list_shape<3>(state.range(0), {1024, 1024, 3});
  std::vector<TensorListShape<>> shapes(3);
  shapes[0] = uniform_list_shape<3>(state.range(0), {800, 1024, 3});
  shapes[1] = uniform_list_shape<3>(state.range(0), {600, 800, 3});
  shapes[2] = uniform_list_shape<3>(state.range(0), {800, 800, 3});
  TensorVector<GPUBackend> batch;
  batch.Resize(full, DALI_UINT32);
  int i = 0;
  for (auto _ : state) {
    batch.Resize(shapes[i]);
    i = (i + 1) % 3;
  }
}
BENCHMARK(batch_tv_resize_fit_gpu)->RangeMultiplier(2)->Range(2, max_range);

int resize_smashing = 128;
float coef = 0.05f;

static void batch_tl_resize_not_fit_cpu(benchmark::State& state) {
  int num_samples = state.range(0);
  int size = 20;
  auto shape = uniform_list_shape<2>(state.range(0), {size, 3});
  TensorList<CPUBackend> batch;
  batch.Resize(shape, DALI_UINT32);
  int i = 0;
  for (auto _ : state) {
    batch.Resize(shape, DALI_UINT32);
    size += size * coef;
    shape = uniform_list_shape<2>(state.range(0), {size, 3});
  }
}
BENCHMARK(batch_tl_resize_not_fit_cpu)->RangeMultiplier(2)->Range(2, max_range)->Iterations(resize_smashing);

static void batch_tl_resize_not_fit_gpu(benchmark::State& state) {
  int num_samples = state.range(0);
  int size = 20;
  auto shape = uniform_list_shape<2>(state.range(0), {size, 3});
  TensorList<GPUBackend> batch;
  batch.Resize(shape, DALI_UINT32);
  int i = 0;
  for (auto _ : state) {
    batch.Resize(shape, DALI_UINT32);
    size += size * coef;
    shape = uniform_list_shape<2>(state.range(0), {size, 3});
  }
}
BENCHMARK(batch_tl_resize_not_fit_gpu)->RangeMultiplier(2)->Range(2, max_range)->Iterations(resize_smashing);


static void batch_tv_resize_not_fit_cpu(benchmark::State& state) {
  int num_samples = state.range(0);
  int size = 20;
  auto shape = uniform_list_shape<2>(state.range(0), {size, 3});
  TensorVector<CPUBackend> batch;
  batch.Resize(shape, DALI_UINT32);
  int i = 0;
  for (auto _ : state) {
    batch.Resize(shape, DALI_UINT32);
    size += size * coef;
    shape = uniform_list_shape<2>(state.range(0), {size, 3});
  }
}
BENCHMARK(batch_tv_resize_not_fit_cpu)->RangeMultiplier(2)->Range(2, max_range)->Iterations(resize_smashing);

static void batch_tv_resize_not_fit_gpu(benchmark::State& state) {
  int num_samples = state.range(0);
  int size = 20;
  auto shape = uniform_list_shape<2>(state.range(0), {size, 3});
  TensorVector<GPUBackend> batch;
  batch.Resize(shape, DALI_UINT32);
  int i = 0;
  for (auto _ : state) {
    batch.Resize(shape, DALI_UINT32);
    size += size * coef;
    shape = uniform_list_shape<2>(state.range(0), {size, 3});
  }
}
BENCHMARK(batch_tv_resize_not_fit_gpu)->RangeMultiplier(2)->Range(2, max_range)->Iterations(resize_smashing);

static void batch_tl_access_cpu(benchmark::State& state) {
  int num_samples = state.range(0);
  auto full = uniform_list_shape<3>(num_samples, {1024, 1024, 3});
  TensorList<CPUBackend> batch;
  batch.Resize(full, DALI_UINT32);
  int i = 0;
  for (auto _ : state) {
    auto sample = batch.mutable_tensor<uint32_t>(i);
    i = (i + 1) % num_samples;
  }
}
BENCHMARK(batch_tl_access_cpu)->RangeMultiplier(2)->Range(2, max_range);

static void batch_tl_access_gpu(benchmark::State& state) {
  int num_samples = state.range(0);
  auto full = uniform_list_shape<3>(num_samples, {1024, 1024, 3});
  TensorList<GPUBackend> batch;
  batch.Resize(full, DALI_UINT32);
  int i = 0;
  for (auto _ : state) {
    auto sample = batch.mutable_tensor<uint32_t>(i);
    i = (i + 1) % num_samples;
  }
}
BENCHMARK(batch_tl_access_gpu)->RangeMultiplier(2)->Range(2, max_range);


static void batch_tv_access_cpu(benchmark::State& state) {
  int num_samples = state.range(0);
  auto full = uniform_list_shape<3>(num_samples, {1024, 1024, 3});
  TensorVector<CPUBackend> batch;
  batch.Resize(full, DALI_UINT32);
  int i = 0;
  for (auto _ : state) {
    auto sample = batch[i].mutable_data<uint32_t>();
    i = (i + 1) % num_samples;
  }
}
BENCHMARK(batch_tv_access_cpu)->RangeMultiplier(2)->Range(2, max_range);

static void batch_tv_access_gpu(benchmark::State& state) {
  int num_samples = state.range(0);
  auto full = uniform_list_shape<3>(num_samples, {1024, 1024, 3});
  TensorVector<GPUBackend> batch;
  batch.Resize(full, DALI_UINT32);
  int i = 0;
  for (auto _ : state) {
    auto sample = batch[i].mutable_data<uint32_t>();
    i = (i + 1) % num_samples;
  }
}
BENCHMARK(batch_tv_access_gpu)->RangeMultiplier(2)->Range(2, max_range);

}  // namespace dali