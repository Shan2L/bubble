#pragma once

#include <cuda_runtime.h>

__host__ __device__ int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}