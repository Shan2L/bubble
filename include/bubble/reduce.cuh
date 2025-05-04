#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

#include "utils.cuh"

namespace bubble {
template <typename scalar_t>
__device__ scalar_t operator_add(scalar_t a, scalar_t b) {
  return a + b;
}

namespace alpha {
template <typename scalar_t>
__global__ void reduce_kernel(scalar_t* __restrict__ out,
                              scalar_t* __restrict__ in,
                              scalar_t* __restrict__ intermediate,
                              int hidden_size, int in_stride) {
  scalar_t sum = 0;
  int batch_id = blockIdx.x;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    int offset_in = batch_id * in_stride + idx;
    sum = bubble::operator_add(in[offset_in], sum);
  }
  int pw = ceil(log2(blockDim.x));
  int intermidiate_size = pow(2, pw);

  intermediate[batch_id * intermidiate_size + threadIdx.x] = sum;
  for (unsigned int stride = intermidiate_size / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      intermediate[batch_id * intermidiate_size + threadIdx.x] +=
          intermediate[batch_id * intermidiate_size + threadIdx.x + stride];
    }
  }

  out[batch_id] = intermediate[batch_id * intermidiate_size];
}

template <typename scalar_t>
void reduce(scalar_t* __restrict__ out,           // [batchsize]
            scalar_t* __restrict__ in,            // [batchsize, hidden_size]
            scalar_t* __restrict__ intermediate,  // [batch_size, block.x]
            int batchsize, int hidden_size, int in_stride,
            const cudaStream_t& stream) {
  dim3 block(min(hidden_size, 1024));
  dim3 grid(batchsize);

  // int shared_mem_size = block.x * sizeof(scalar_t);
  bubble::alpha::reduce_kernel<scalar_t><<<grid, block, 0, stream>>>(
      out, in, intermediate, hidden_size, in_stride);
}
}  // namespace alpha

}  // namespace bubble