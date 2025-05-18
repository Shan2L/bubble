import time
from typing import Callable, Tuple
from time import perf_counter_ns

import torch

def BENCHMARK_OPERATOR(op: Callable, 
                        output: torch.Tensor, 
                        input: Tuple[torch.Tensor], 
                        iter: int=1000):
    kernel_time: float = 0
    host_time: float = 0 
    for _ in range(iter):
        start = perf_counter_ns()
        single_time = op(output, *input)
        end = perf_counter_ns()
        kernel_time += single_time
        host_time+=(end-start)
    return host_time/iter, kernel_time/iter