import torch
import pytest
from bubble.ops import reduce_add
from utils import get_default_rtol, get_default_atol


BATCHSIZES = [2, 33]
HIDDEN_SIZES = [255, 277, 333, 1024]
DTYPES = [torch.float, torch.half, torch.bfloat16]
VERSIONS = ["alpha", "beta"]
SEEDS = [1111]
CUDA_DEVICES = ["cuda:0"]

@pytest.mark.parametrize("batchsize", BATCHSIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("version", VERSIONS)
@torch.inference_mode()
def test_reduce_add(
    batchsize: int,
    hidden_size: int,
    version: str,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)
    input = torch.randn(batchsize, hidden_size, dtype=dtype)
    output = torch.empty(batchsize, dtype=dtype)
    golden = torch.sum(input, dim=-1)
    reduce_add(output, input, version)
    print(f"golden: {golden}")
    print(f"result: {output}")

    print(golden.shape)
    print(output.shape)

    torch.testing.assert_close(golden, output, rtol=get_default_rtol(input), atol=get_default_atol(input))