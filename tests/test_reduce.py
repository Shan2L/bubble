import torch
import pytest
from bubble.ops import reduce_add
from utils import get_default_rtol, get_default_atol


HIDDEN_SIZES = [100, 500, 2048, 10086, 12312312]
DTYPES = [torch.float, torch.half, torch.bfloat16]
VERSIONS = ["alpha", "beta", "delta"]
SEEDS = [1111]
CUDA_DEVICES = ["cuda:0"]

@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("version", VERSIONS)
@torch.inference_mode()
def test_reduce_add(
    hidden_size: int,
    version: str,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    input = torch.ones((hidden_size, ), dtype=dtype, device="cuda")
    input2 = input.clone()
    output = torch.zeros(1, dtype=torch.float, device="cuda")
    golden = torch.sum(input2.float(), dim=-1, keepdim=True).float()
    reduce_add(output, input, version)

    torch.testing.assert_close(golden.cpu(), output.cpu(), rtol=1e-3, atol=1e-3)