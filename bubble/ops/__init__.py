import torch
import bubble._C

def reduce_add(out: torch.Tensor, input: torch.Tensor, version: str):
    op = torch.ops.bubble.reduce_add
    return op(out, input, version)