#!/usr/bin/env python3
import os
from datetime import datetime
import pytz
import yaml
import copy
import torch
from lightning.pytorch import seed_everything
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from src.utils.runner_utils import parse_args, register_run, get_logger, start_of_a_run
from task_processor import task_processor
from task_processor_ns import task_processor_ns
from task_processor_inbatch import task_processor_inbatch



if __name__ == "__main__":
    config = start_of_a_run()
    
    os.environ['TOKENIZERS_PARALLELISM'] = config['TOKENIZERS_PARALLELISM']
    os.environ['CUDA_LAUNCH_BLOCKING'] = config['CUDA_LAUNCH_BLOCKING']
    os.environ['TORCH_USE_CUDA_DS'] = config['TORCH_USE_CUDA_DS']
    torch.set_float32_matmul_precision(config['float32_matmul_precision'])

    # sets seeds for numpy, torch and python.random.
    seed_everything(config['seed'], workers=True)

    task_processor_inbatch(config)




    
