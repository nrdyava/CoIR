#!/usr/bin/env python3
import os
import argparse
from datetime import datetime
import pytz
import yaml
import logging
import copy
from src.utils.runner_utils import parse_args, register_run, get_logger

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
