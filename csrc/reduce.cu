#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

#include <bubble/reduce.cuh>
#include "dispatch_utils.h"
#include "kernel_version_utils.h"

void reduce_add(torch::Tensor& out, torch::Tensor& input,
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

  int block_dim = std::min(hidden_size, 1024);
  torch::Tensor intermediate = torch::empty({block_dim}, input.options());

  BUBBLE_DISPATCH_FLOATING_TYPES(out.scalar_type(), "reduce_add", [&] {
    switch (str2version(version)) {
      case 0:
        bubble::alpha::reduce(out.data_ptr<scalar_t>(),
                              input.data_ptr<scalar_t>(),
                              intermediate.data_ptr<scalar_t>(), batchsize,
                              hidden_size, in_stride, stream);
        break;
      default:
        std::cerr << "The version has not been supported yet." << std::endl;
        std::exit(-1);
    }
  });
}

TORCH_LIBRARY_FRAGMENT(bubble, m) {
  m.def("bubble::reduce_add(Tensor! out, Tensor input, str version) -> ()");
}

TORCH_LIBRARY_IMPL(bubble, CUDA, m) {
  m.impl("bubble::reduce_add", &reduce_add);
}
