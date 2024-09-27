#!/usr/bin/env python3
import os
import torch
from lightning.pytorch import seed_everything

from src.utils.runner_utils import start_of_a_run
from task_processor import task_processor


if __name__ == "__main__":
    config = start_of_a_run()
    
    os.environ['TOKENIZERS_PARALLELISM'] = config['TOKENIZERS_PARALLELISM']
    os.environ['CUDA_LAUNCH_BLOCKING'] = config['CUDA_LAUNCH_BLOCKING']
    os.environ['TORCH_USE_CUDA_DS'] = config['TORCH_USE_CUDA_DS']
    torch.set_float32_matmul_precision(config['float32_matmul_precision'])

    # sets seeds for numpy, torch and python.random.
    seed_everything(config['seed'], workers=True)

    task_processor(config)