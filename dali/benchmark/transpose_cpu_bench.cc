#include <benchmark/benchmark.h>
#include "dali/kernels/common/transpose.h"
#include "dali/kernels/alloc_type.h"

namespace dali {

static void BM_SomeFunction(benchmark::State& state) {
  std::vector<int> out_mem, in_mem;
  TensorView<StorageCPU, int> out, in;
  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    // SomeFunction();
  }
}
// Register the function as a benchmark
BENCHMARK(BM_SomeFunction);

}  // namespace dali