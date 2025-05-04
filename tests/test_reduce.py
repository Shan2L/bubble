import torch
import pytest
from bubble.ops import reduce_add
from utils import get_default_rtol, get_default_atol


BATCHSIZES = [1, 33, 255]
HIDDEN_SIZES = [255, 511, 1024]
DTYPES = [torch.float, torch.half, torch.bfloat16]
SEEDS = [2025]
CUDA_DEVICES = ["cuda:0"]

@pytest.mark.parametrize("batchsize", BATCHSIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_reduce_add(
    batchsize: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)
    input = torch.randn(batchsize, hidden_size)
    output = torch.empty(batchsize)
    reduce_add(output, input)
    golden = torch.sum(input, dim=-1)

    torch.testing.assert_close(golden, output, rtol=get_default_rtol(output), atol=get_default_atol(output))