#!/usr/bin/env python3
import os
import argparse
from datetime import datetime
import pytz
import yaml
import logging
import copy
from src.utils.runner_utils import parse_args, register_run, get_logger
from src.utils.config_check import check_config
from src.datasets.lasco_datasets import lasco_dataset_train, lasco_dataset_val
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import seed_everything

if __name__ == "__main__":
    run_start_time_utc = datetime.now(pytz.utc)

    args = parse_args()
    config = copy.deepcopy(yaml.safe_load(open(args.config_file, 'r')))

    run_start_time_local = run_start_time_utc.astimezone(pytz.timezone(config["local_time_zone"])).strftime("%Y-%m-%d-%H-%M-%S-%f")

    runs_dir = config["runs_dir"]
    run_dir = os.path.join(runs_dir, run_start_time_local)
    os.mkdir(run_dir)

    logger = get_logger(run_dir, config)
    register_run(args, run_start_time_local, config, run_dir, logger)

    yaml.dump(config, open(os.path.join(run_dir, 'config.yaml'), 'w'), default_flow_style=False, sort_keys=False)

    #check_config(config, logger)
    os.environ['TOKENIZERS_PARALLELISM'] = config['TOKENIZERS_PARALLELISM']

    """
    #train_dataset = lasco_dataset_train(config, logger)
    #train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers = 0, collate_fn = train_dataset.collate_fn, pin_memory =  True)

    for i, batch in enumerate(train_dataloader):
        print(f" ------------------------------------- {i} -------------------------------")
        print(batch)
        if i == 5:
            break
    """
    """
    val_dataset = lasco_dataset_val(config, logger)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers = 0, collate_fn = val_dataset.collate_fn, pin_memory =  True)

    for i, batch in enumerate(val_dataloader):
        print(f" ------------------------------------- {i} -------------------------------")
        print(batch)
        if i == 5:
            break
    """

    
    # sets seeds for numpy, torch and python.random.
    seed_everything(42, workers=True)
    
