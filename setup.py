from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from glob import glob
import os
from pathlib import Path


root = Path(__file__).parent
bubble_include_dirs = root / "include"
print(bubble_include_dirs)

if __name__ == "__main__":

    setup(name='bubble', 
          version="0.0.1",
          ext_modules=[
              CUDAExtension("bubble._C",
                            sources=["csrc/stub.c", "csrc/reduce.cu"],
                            include_dirs = [bubble_include_dirs.resolve()])
          ],
          cmdclass={
              'build_ext': BuildExtension
          }
          )

