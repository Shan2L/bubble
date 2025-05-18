import torch
from bubble.ops import reduce_add
from typing import Tuple

from benchmark_utils import BENCHMARK_OPERATOR


def warm_up(func: callable, paras: Tuple, round: int=10):
    for _ in range(round):
        func(*paras)

# torch.manual_seed(0)
hidden_dim = 10086344
dtype = torch.float

torch.set_default_device("cuda")
input = torch.ones(hidden_dim)
output1 = torch.zeros(1, dtype=torch.float)
output2 = torch.zeros(1, dtype=torch.float)
output3 = torch.zeros(1, dtype=torch.float)

warm_up(torch.add, (input, input), 100)

htime_alpha, dtime_alpha = BENCHMARK_OPERATOR(reduce_add, output2, (input, "alpha"))
htime_beta, dtime_beta = BENCHMARK_OPERATOR(reduce_add, output1, (input, "beta"))
htime_delta, dtime_deta = BENCHMARK_OPERATOR(reduce_add, output3, (input, "delta"))

print("=======================runtime statistical info==================")
print("|version|===========| host time| ======| device time| ==============")
print(f"alpha===============|{htime_alpha}|========|{dtime_alpha}|")
print(f"beta===============|{htime_beta}|========|{dtime_beta}|")
print(f"delta===============|{htime_delta}|========|{dtime_deta}|")



