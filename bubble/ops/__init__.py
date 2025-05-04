import torch
import bubble._C

def reduce_add(out: torch.Tensor, input: torch.Tensor, version: str="alpha"):
    op = torch.ops.bubble.reduce_add
    op(out, input, version)