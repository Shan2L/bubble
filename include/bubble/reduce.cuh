#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

#include "utils.cuh"

namespace bubble {

namespace alpha {

__forceinline__ __device__ int offset(int offset, int base)
{
  return offset + base*blockIdx.x;
}

template <typename scalar_t>
__global__ void reduce_kernel(float* __restrict__ out,
                              scalar_t* __restrict__ in,
                              float* __restrict__ space,
                              int hidden_size) {
    int space_size = pow(2, ceil(log2(blockDim.x)));
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    if (idx < hidden_size)
    {
      space[offset(threadIdx.x, space_size)] = in[idx];
    }

    for (unsigned int stride=space_size>>1; stride>0; stride>>=1)
    {
      __syncthreads();
      if (threadIdx.x < stride)
      {
        space[offset(threadIdx.x, space_size)] += space[offset(threadIdx.x+stride, space_size)];
      }
    }

    __syncthreads();
    if(threadIdx.x == 0)
    {
      atomicAdd(out, space[offset(0, space_size)]);
    }
}

template <typename scalar_t>
void reduce(
    float* __restrict__ out,        // [1]
    scalar_t* __restrict__ in,         // [hidden_size]
    float* __restrict__ intermediate,  // [intermediate_size]
    int hidden_size,
    const cudaStream_t& stream = nullptr) {
  dim3 block(min(hidden_size, 1024));
  dim3 grid(ceil_div(hidden_size, block.x));
  std::cout << "grid: "<< grid.x <<std::endl;
  std::cout << "block: "<< block.x <<std::endl;
  std::cout << "[Bubble Debug Info] Using reduce operator with alpha version."
            << std::endl;
  bubble::alpha::reduce_kernel<scalar_t><<<grid, block, 0, stream>>>(
      out, in, intermediate, hidden_size);
}
}  // namespace alpha

namespace beta {

template <typename scalar_t>
__global__ void reduce_kernel(float* __restrict__ out,
                              scalar_t* __restrict__ input, int hidden_size) {
  extern __shared__ float shared_mem[];

  int shm_size = pow(2, ceil(log2(blockDim.x)));
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int idx=threadIdx.x; idx<shm_size; idx+=blockDim.x)
  {
    shm[idx] = 0.f;
  }
 
  __syncthreads();

  if (idx < hidden_size)
  {
    shm[threadIdx.x] = input[idx];
  }
  
  for (unsigned int stride=shm_size>>1; stride>0; stride>>=1)
  {
    __syncthreads();
    if (threadIdx.x < stride)
    {
      shm[threadIdx.x]+= shm[threadIdx.x+stride];
    }
  }

  __syncthreads();
  
  if(threadIdx.x == 0)
  {
    atomicAdd(out, shm[0]);
  }
}

template <typename scalar_t>
void reduce(float* __restrict__ out, scalar_t* __restrict__ input,
            int hidden_size, const cudaStream_t& stream = nullptr) {
  dim3 block(std::min(hidden_size, 1024));
  dim3 grid(ceil_div(hidden_size, block.x));

  std::cout << "grid: "<< grid.x <<std::endl;
  std::cout << "block: "<< block.x <<std::endl;
  int64_t shm_size = pow(2, ceil(log2(block.x)));
  std::cout << "[Bubble Debug Info] Using reduce operator with beta version."
            << std::endl;
  bubble::beta::reduce_kernel<scalar_t>
      <<<grid, block, shm_size * sizeof(float), stream>>>(
          out, input, hidden_size);
}

}  // namespace beta


namespace delta {

template <typename scalar_t>
__global__ void reduce_kernel(float* out, scalar_t* input, int hidden_size)
{
  extern __shared__ float shm[];

  // Execute reduction within warp using shuffle instruction
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  float val = (global_idx < hidden_size) ? (float)input[global_idx] : 0.f;

#pragma unroll 
  for (unsigned int stride=warpSize>>1; stride>0; stride>>=1)
  {
    val += __shfl_down_sync(0xFFFFFFFF, val, stride);
  }

  // put the warp-wise result into shared memory
  int num_warps = ceil_div(blockDim.x, warpSize);
  int warp_id = threadIdx.x / warpSize;
  int lane_id = threadIdx.x % warpSize;
  if (lane_id == 0)
  {
    shm[warp_id] = val;
  }
  
  __syncthreads();

  if (warp_id == 0)
  {
    val = lane_id >= num_warps ? 0.f : shm[lane_id];
#pragma unroll 
    for (unsigned int stride=warpSize>>1; stride>0; stride>>=1)
    {
      val += __shfl_down_sync(0xFFFFFFFF, val, stride);
    }

    if (lane_id == 0)
    {
      atomicAdd(out, val);
    }
  }
}


template <typename scalar_t>
void reduce(float* __restrict__ out, scalar_t* __restrict__ input,
            int hidden_size, const cudaStream_t& stream=nullptr)
{
    int num_warps = ceil_div(hidden_size, 32);
    int block_x = num_warps > 32 ? 1024 : num_warps * 32;
    dim3 block(block_x);
    dim3 grid(ceil_div(hidden_size, block_x));
    std::cout << "[Bubble Debug Info] Using reduce operator with delta version."
    << std::endl;
    std::cout << "[bubble DEBUG info] grid: " << grid.x << std::endl;
    std::cout << "[bubble DEBUG info] block: " << block.x << std::endl;
    bubble::delta::reduce_kernel<scalar_t>
          <<<grid, block, block_x/32*sizeof(float), stream>>>(out, input, hidden_size);
}

}

}  // namespace bubble