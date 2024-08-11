#!/usr/bin/env python3
import os
import argparse
from datetime import datetime
import pytz
import yaml
import logging
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import seed_everything

from src.utils.runner_utils import parse_args, register_run, get_logger
from src.utils.config_check import check_config
from src.datasets.lasco_datasets import lasco_dataset_train, lasco_dataset_val
from task_processor import task_processor


if __name__ == "__main__":
    run_start_time_utc = datetime.now(pytz.utc)

    args = parse_args()
    config = copy.deepcopy(yaml.safe_load(open(args.config_file, 'r')))

    run_start_time_local = run_start_time_utc.astimezone(pytz.timezone(config["local_time_zone"])).strftime("%Y-%m-%d-%H-%M-%S-%f")

    # create a directory to store the results of the run
    runs_dir = config["runs_dir"]
    run_dir = os.path.join(runs_dir, run_start_time_local)
    #os.mkdir(run_dir)
    config['run_dir'] = run_dir
    config['wandb_name'] = run_start_time_local

    # create a logger to log the results of the run
    logger = get_logger(run_dir, config)
    # register the run in the run_registry.txt file
    #register_run(args, run_start_time_local, config, run_dir, logger)

    # save the configuration file in the run directory. Usefule to later check the configuration used for the run.
    #yaml.dump(config, open(os.path.join(run_dir, 'config.yaml'), 'w'), default_flow_style=False, sort_keys=False)

    # check_config(config, logger)
    os.environ['TOKENIZERS_PARALLELISM'] = config['TOKENIZERS_PARALLELISM']
    logger.info('TOKENIZERS_PARALLELISM is set to : {}'.format(config['TOKENIZERS_PARALLELISM']))

    # CUDA LAUNCH BLOCKING to report CUDA errors
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
    export TORCH_USE_CUDA_DSA=1
    # sets seeds for numpy, torch and python.random.
    seed_everything(42, workers=True)

    torch.set_float32_matmul_precision('high')
    task_processor(config, logger)




    
