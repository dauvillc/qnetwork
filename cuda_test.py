"""
A little testing file to verify the implementation of CUDA
"""

import torch


if __name__ == "__main__":
    print(torch.cuda.is_available())