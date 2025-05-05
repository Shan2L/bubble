import torch
from bubble.ops import reduce_add
from typing import Tuple


def warm_up(func: callable, paras: Tuple, round: int=10):
    for _ in range(round):
        func(*paras)


# torch.manual_seed(0)
batch_size = 1024
hidden_dim = 1024
dtype = torch.float

torch.set_default_device("cuda")
input = torch.randn(batch_size, hidden_dim)
output = torch.randn(batch_size)

warm_up(reduce_add, (output, input, "alpha"), 30)

time2 = reduce_add(output, input, "BETA")
print(time2)

time1 = reduce_add(output, input, "alpha")
print(time1)
