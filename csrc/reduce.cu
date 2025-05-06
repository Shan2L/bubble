#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <iostream>

#include <bubble/reduce.cuh>
#include "dispatch_utils.h"
#include "kernel_version_utils.h"
#include "timer.cuh"

double reduce_add(torch::Tensor& out, torch::Tensor& input,
                  const std::string& version) {
  int batchsize = input.size(0);
  int hidden_size = input.size(1);
  int in_stride = input.stride(0);

  TORCH_INTERNAL_ASSERT(batchsize == out.size(0),
                        "The dimension of out and in Tensor are not same.");
  TORCH_INTERNAL_ASSERT(
      version == "alpha" || version == "beta" || version == "gamma",
      "The version is incorrect.");

  const at::cuda::OptionalCUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  CUDAKernelTimer& timer = CUDAKernelTimer::getInstance();
  timer.begin();
  BUBBLE_DISPATCH_FLOATING_TYPES(out.scalar_type(), "reduce_add", [&] {
    int version_id = str2version(version);
    switch (version_id) {
      case 0: {
        int block_dim = std::min(hidden_size, 1024);
        int intermediate_size = pow(2, ceil(log2(block_dim)));
        torch::Tensor intermediate =
            torch::zeros({batchsize, intermediate_size},
                         input.options().dtype(torch::kFloat32));
        bubble::alpha::reduce<scalar_t>(
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            intermediate.data_ptr<float>(), batchsize, hidden_size, in_stride,
            stream);
      } break;
      case 1: {
        bubble::beta::reduce<scalar_t>(out.data_ptr<scalar_t>(),
                                       input.data_ptr<scalar_t>(), batchsize,
                                       hidden_size, in_stride, stream);
      } break;
      default:
        std::cerr << "The version has not been supported yet." << std::endl;
        std::exit(-1);
    }
  });
  timer.end();
  return timer.elapse();
}

TORCH_LIBRARY_FRAGMENT(bubble, m) {
  m.def("bubble::reduce_add(Tensor! out, Tensor input, str version) -> float");
}

TORCH_LIBRARY_IMPL(bubble, CUDA, m) {
  m.impl("bubble::reduce_add", &reduce_add);
}
