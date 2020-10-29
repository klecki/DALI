// from https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/
#ifndef MAIN
#include <gtest/gtest.h>
#endif
#include <cstdio>
#include <cmath>


void __host__ __device__ test_arch(int id) {
  #ifdef __CUDA_ARCH__
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf ("GPU %d\n", id);
  #else
  printf("CPU %d\n", id);
  #endif
}

__global__
void saxpy(int n, float a, float *x, float *y)
{
  test_arch(0);
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void run_test(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  test_arch(1);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = std::fmax(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}

#ifndef MAIN
TEST(CUDA_TEST, SIMPLEST) {
  run_test();
}

#else

int main() {
  run_test();
  return 0;
}

#endif