// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <benchmark/benchmark.h>

#include <memory>
#include <vector>

namespace dali {

static constexpr int max_range = 1 << 8;

// The shared pointer will be kept in a vector, so we do it this way here.

// Create shared_ptr to nullptr
static void buffer_shared_nullptr(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<std::shared_ptr<int>> v;
    v.reserve(state.range(0));
    benchmark::DoNotOptimize(v.data());
    for (int i = 0; i < state.range(0); i++) {
      v.push_back(std::shared_ptr<int>(nullptr));
      benchmark::DoNotOptimize(v.back());
    }
    benchmark::ClobberMemory();
  }
}

BENCHMARK(buffer_shared_nullptr)->RangeMultiplier(2)->Range(2, max_range);


// Create shared_ptr to a valid pointer that we just newd into existence
static void buffer_shared_alloc(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<std::shared_ptr<int>> v;
    v.reserve(state.range(0));
    benchmark::DoNotOptimize(v.data());
    for (int i = 0; i < state.range(0); i++) {
      v.push_back(std::shared_ptr<int>(new int));
      benchmark::DoNotOptimize(v.back());
    }
    benchmark::ClobberMemory();
  }
}

BENCHMARK(buffer_shared_alloc)->RangeMultiplier(2)->Range(2, max_range);


// Allocate and deallocate pointer
static void buffer_alloc_dealloc(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<int *> v;
    v.reserve(state.range(0));
    benchmark::DoNotOptimize(v.data());
    for (int i = 0; i < state.range(0); i++) {
      v.push_back(new int);
      benchmark::DoNotOptimize(v.back());
    }
    for (auto elem : v) {
      delete elem;
    }
    benchmark::ClobberMemory();
  }
}

BENCHMARK(buffer_alloc_dealloc)->RangeMultiplier(2)->Range(2, max_range);


// creating the aliasing shared ptr
static void buffer_alias(benchmark::State& state) {
  for (auto _ : state) {
    std::shared_ptr<int> ptr(new int[state.range(0)]);
    benchmark::DoNotOptimize(ptr.get());
    std::vector<std::shared_ptr<int>> v;
    v.reserve(state.range(0));
    benchmark::DoNotOptimize(v.data());
    for (int i = 0; i < state.range(0); i++) {
      v.push_back(std::shared_ptr<int>(ptr, ptr.get() + i));
      benchmark::DoNotOptimize(v.back());
    }
    benchmark::ClobberMemory();
  }
}

BENCHMARK(buffer_alias)->RangeMultiplier(2)->Range(2, max_range);


// Allocate and deallocate once, the allocation is not really measured
static void buffer_alloc_dealloc_once(benchmark::State& state) {
  for (auto _ : state) {
    auto *ptr = new int[state.range(0)];
    benchmark::DoNotOptimize(ptr);
    std::vector<int *> v;
    v.reserve(state.range(0));
    benchmark::DoNotOptimize(v.data());
    for (int i = 0; i < state.range(0); i++) {
      v.push_back(ptr + i);
      benchmark::DoNotOptimize(v.back());
    }
    benchmark::ClobberMemory();
    delete[] ptr;
  }
}

BENCHMARK(buffer_alloc_dealloc_once)->RangeMultiplier(2)->Range(2, max_range);


// Copy the shared ptr a lot
static void buffer_copy_ptr(benchmark::State& state) {
  std::shared_ptr<int> ptr(new int[state.range(0)]);
  benchmark::DoNotOptimize(ptr.get());
  std::vector<std::shared_ptr<int>> v;
  v.reserve(state.range(0));
  for (int i = 0; i < state.range(0); i++) {
    v.push_back(std::shared_ptr<int>(ptr));
    benchmark::DoNotOptimize(v.back());
  }
  for (auto _ : state) {
    std::vector<std::shared_ptr<int>> copy;
    copy = v;
    benchmark::DoNotOptimize(copy.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(buffer_copy_ptr)->RangeMultiplier(2)->Range(2, max_range);


// Copy the shared ptr, that is aliasing, a lot
static void buffer_copy_aliasing_ptr(benchmark::State& state) {
  std::shared_ptr<int> ptr(new int[state.range(0)]);
  benchmark::DoNotOptimize(ptr.get());
  std::vector<std::shared_ptr<int>> v;
  v.reserve(state.range(0));
  for (int i = 0; i < state.range(0); i++) {
    v.push_back(std::shared_ptr<int>(ptr, ptr.get() + i));
    benchmark::DoNotOptimize(v.back());
  }
  for (auto _ : state) {
    std::vector<std::shared_ptr<int>> copy;
    copy = v;
    benchmark::DoNotOptimize(copy.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(buffer_copy_aliasing_ptr)->RangeMultiplier(2)->Range(2, max_range);

}  // namespace dali