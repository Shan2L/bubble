#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

#include "utils.cuh"

namespace bubble {

namespace alpha {
template <typename scalar_t>
__global__ void reduce_kernel(scalar_t* __restrict__ out,
                              scalar_t* __restrict__ in,
                              float* __restrict__ intermediate, int hidden_size,
                              int in_stride) {
  float sum = 0;
  int batch_id = blockIdx.x;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    int offset_in = batch_id * in_stride + idx;
    sum += (float)in[offset_in];
  }
  int intermediate_size = pow(2, ceil(log2(blockDim.x)));

  intermediate[batch_id * intermediate_size + threadIdx.x] = sum;
  for (unsigned int stride = intermediate_size / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (threadIdx.x < stride) {
      intermediate[batch_id * intermediate_size + threadIdx.x] +=
          intermediate[batch_id * intermediate_size + threadIdx.x + stride];
    }
  }

  if (threadIdx.x == 0) {
    out[batch_id] = intermediate[batch_id * intermediate_size];
  }
}

template <typename scalar_t>
void reduce(
    scalar_t* __restrict__ out,        // [batchsize]
    scalar_t* __restrict__ in,         // [batchsize, hidden_size]
    float* __restrict__ intermediate,  // [batch_size, intermediate_size]
    int batchsize, int hidden_size, int in_stride,
    const cudaStream_t& stream = nullptr) {
  dim3 block(min(hidden_size, 1024));
  dim3 grid(batchsize);
  std::cout << "[Bubble Debug Info] Using reduce operator with alpha version."
            << std::endl;
  bubble::alpha::reduce_kernel<scalar_t><<<grid, block, 0, stream>>>(
      out, in, intermediate, hidden_size, in_stride);
}
}  // namespace alpha

namespace beta {

template <typename scalar_t>
__global__ void reduce_kernel(scalar_t* __restrict__ out,
                              scalar_t* __restrict__ input, int hidden_size,
                              int in_stride) {
  extern __shared__ char shared_mem[];
  float* shm = reinterpret_cast<float*>(&shared_mem);

  int shm_size = pow(2, ceil(log2(blockDim.x)));
  for (int i = threadIdx.x; i < shm_size; i += blockDim.x) {
    shm[i] = 0;
  }
  __syncthreads();

  float sum = 0;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int64_t offset = blockIdx.x * in_stride + i;
    sum += (float)input[offset];
  }
  shm[threadIdx.x] = sum;

  for (unsigned int stride = shm_size / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (threadIdx.x < stride) {
      shm[threadIdx.x] += shm[threadIdx.x + stride];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    out[blockIdx.x] = shm[0];
  }
}

template <typename scalar_t>
void reduce(scalar_t* __restrict__ out, scalar_t* __restrict__ input,
            int batchsize, int hidden_size, int in_stride,
            const cudaStream_t& stream = nullptr) {
  dim3 grid(batchsize);
  dim3 block(std::min(hidden_size, 1024));
  int64_t shm_size = pow(2, ceil(log2(block.x)));
  std::cout << "[Bubble Debug Info] Using reduce operator with beta version."
            << std::endl;
  bubble::beta::reduce_kernel<scalar_t>
      <<<grid, block, shm_size * sizeof(float), stream>>>(
          out, input, hidden_size, in_stride);
}

}  // namespace beta

}  // namespace bubble