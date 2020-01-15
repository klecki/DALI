#include <numeric>
#include <benchmark/benchmark.h>
#include "dali/kernels/common/transpose.h"
#include "dali/kernels/alloc_type.h"
#include "dali/operators/transpose/transpose.h"

#include "dbg.h"


namespace dali {

namespace {
static std::tuple<TensorShape<>, std::vector<int>> cases[] = {
  // HWC small cases
    {{8, 6, 3}, {0, 1, 2}}, // HWC
    {{8, 6, 3}, {2, 0, 1}}, // CHW
    {{8, 6, 3}, {2, 1, 0}}, // CWH
  // HWC small cases
    {{8, 6, 4}, {0, 1, 2}}, // HWC
    {{8, 6, 4}, {2, 0, 1}}, // CHW
    {{8, 6, 4}, {2, 1, 0}}, // CWH
  // HWC bigger cases
    {{100, 60, 3}, {0, 1, 2}}, // HWC
    {{100, 60, 3}, {2, 0, 1}}, // CHW
    {{100, 60, 3}, {2, 1, 0}}, // CWH
  // 4D
    {{20, 20, 20, 4}, {0, 1, 2, 3}}, // id
    {{20, 20, 20, 4}, {3, 2, 1, 0}},
    {{20, 20, 20, 4}, {0, 1, 3, 2}},
    {{20, 20, 20, 4}, {0, 3, 1, 2}},
  // 10D
    {{7, 2, 4, 6, 10, 8, 4, 2}, {0, 1, 2, 3, 4, 5, 6, 7}},
    {{7, 2, 4, 6, 10, 8, 4, 2}, {7, 5, 4, 2, 4, 0, 1, 6}},
  };

std::tuple<TensorShape<>, std::vector<int>> GetCase(int id) {

  return cases[id];
}

// TensorShape<> GetShape(int dim, int scale) {
//   TensorShape<8> gold_shape{3, 5, 2, 10, 8, 6, 4, 2};

//   auto ret = gold_shape.last(dim);
//   for (auto &elem : ret) {
//     elem *= scale;
//   }
//   return ret;
// }

// std::vector<int> GetPerm(int dim, int swaps) {
//   std::vector<int> result(dim);
//   std::iota(result.begin(), result.end(), 0);
//   return result;
// }

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (unsigned int i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
    for (int scale = 1; scale <= 8; scale *= 2) {
      b->Args({i, scale});
    }
  }
  // for (int dim = 2; dim <= 8; ++dim)
  //   for (int scale = 1; scale <= 4; scale *= 2)
  //     for (int swaps = 1; swaps < dim; ++swaps)
  //       b->Args({dim, scale, swaps});
}

}


static void BM_Permute(benchmark::State& state) {
  // auto perm = GetPerm(state.range(0), state.range(2));
  // auto in_shape = GetShape(state.range(0), state.range(1));
  // auto out_shape = detail::Permute(in_shape, perm);
  // std::vector<int> dst_mem_, src_mem_;
  // TensorView<StorageCPU, int> out, in;

  // Perform setup here
  for (auto _ : state) {
    // std::cout << _ << std::endl;
    // This code gets timed
    // SomeFunction();
  }
}
// Register the function as a benchmark
// BENCHMARK(BM_Permute)->Apply(CustomArguments);


template <typename T>
class TransposeFixture : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& st) override {
    auto test_case = GetCase(st.range(0));
    src_shape_ = std::get<0>(test_case);
    int dim = 0, total_dims = src_shape_.size();
    for (auto &elem : src_shape_) {
      if (dim++ < 4)
        elem *= st.range(1);
    }
    perm_ = std::get<1>(test_case);
    dst_shape_ = kernels::Permute(src_shape_, perm_);
    auto total_size = volume(src_shape_);
    dst_mem_.resize(total_size);
    src_mem_.resize(total_size);
    for (int64_t i = 0; i < total_size; i++) {
      src_mem_[i] = i;
    }
    src_view_ = TensorView<StorageCPU, const T>(src_mem_.data(), src_shape_);
    dst_view_ = TensorView<StorageCPU, T>(dst_mem_.data(), dst_shape_);
  }

  // void TearDown(benchmark::State& st) override {
  // }
  std::vector<int> perm_;

  TensorView<StorageCPU, T> dst_view_;
  TensorView<StorageCPU, const T> src_view_;
  TensorShape<> src_shape_, dst_shape_;
  std::vector<T> dst_mem_, src_mem_;
};

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, Uint8Test, uint8_t)(benchmark::State& st) {
   for (auto _ : st) {
    benchmark::DoNotOptimize(src_mem_.data());
    kernels::transpose(dst_view_, src_view_, make_span(perm_));
    benchmark::DoNotOptimize(dst_mem_.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, Uint16Test, uint16_t)(benchmark::State& st) {
   for (auto _ : st) {
    benchmark::DoNotOptimize(src_mem_.data());
    kernels::transpose(dst_view_, src_view_, make_span(perm_));
    benchmark::DoNotOptimize(dst_mem_.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, IntTest, int)(benchmark::State& st) {
   for (auto _ : st) {
    benchmark::DoNotOptimize(src_mem_.data());
    kernels::transpose(dst_view_, src_view_, make_span(perm_));
    benchmark::DoNotOptimize(dst_mem_.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, DoubleTest, double)(benchmark::State& st) {
   for (auto _ : st) {
    benchmark::DoNotOptimize(src_mem_.data());
    kernels::transpose(dst_view_, src_view_, make_span(perm_));
    benchmark::DoNotOptimize(dst_mem_.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_REGISTER_F(TransposeFixture, Uint8Test)->Apply(CustomArguments);
BENCHMARK_REGISTER_F(TransposeFixture, Uint16Test)->Apply(CustomArguments);
BENCHMARK_REGISTER_F(TransposeFixture, IntTest)->Apply(CustomArguments);
BENCHMARK_REGISTER_F(TransposeFixture, DoubleTest)->Apply(CustomArguments);


BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, MemcpyUint8Test, uint8_t)(benchmark::State& st) {
   for (auto _ : st) {
    benchmark::DoNotOptimize(src_mem_.data());
    kernels::transpose_memcpy(dst_view_, src_view_, make_span(perm_));
    benchmark::DoNotOptimize(dst_mem_.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, MemcpyUint16Test, uint16_t)(benchmark::State& st) {
   for (auto _ : st) {
    benchmark::DoNotOptimize(src_mem_.data());
    kernels::transpose_memcpy(dst_view_, src_view_, make_span(perm_));
    benchmark::DoNotOptimize(dst_mem_.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, MemcpyIntTest, int)(benchmark::State& st) {
   for (auto _ : st) {
    benchmark::DoNotOptimize(src_mem_.data());
    kernels::transpose_memcpy(dst_view_, src_view_, make_span(perm_));
    benchmark::DoNotOptimize(dst_mem_.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, MemcpyDoubleTest, double)(benchmark::State& st) {
   for (auto _ : st) {
    benchmark::DoNotOptimize(src_mem_.data());
    kernels::transpose_memcpy(dst_view_, src_view_, make_span(perm_));
    benchmark::DoNotOptimize(dst_mem_.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_REGISTER_F(TransposeFixture, MemcpyUint8Test)->Apply(CustomArguments);
BENCHMARK_REGISTER_F(TransposeFixture, MemcpyUint16Test)->Apply(CustomArguments);
BENCHMARK_REGISTER_F(TransposeFixture, MemcpyIntTest)->Apply(CustomArguments);
BENCHMARK_REGISTER_F(TransposeFixture, MemcpyDoubleTest)->Apply(CustomArguments);


}  // namespace dali

