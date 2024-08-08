#!/usr/bin/env python3
import os
import argparse
from datetime import datetime
import pytz
import yaml
import logging
import copy

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='configuration file path for the task ', dest='config_file')
    parser.add_argument('--run_name', type=str, required=True, help='name of the task you want to run', dest='run_name')
    parser.add_argument('--notes', type=str, default = "No notes mentioned", help='write a note about the task you want to run', dest='notes')

    args = parser.parse_args()
    return args

def register_run(args, run_start_time_local, config, run_dir, logger):
    write_text = f"\n"
    ltz = config["local_time_zone"]
    write_text += f"run start time: {run_start_time_local} {ltz}\n"
    write_text += f"run name: {args.run_name}\n"
    write_text += f"configuration file path: {args.config_file}\n"
    write_text += f"notes: {args.notes}\n"
    write_text += f"results of the run are stored at: {run_dir}\n"
    write_text += f"\n==================================================================================================================\n"

    logger.info(f"run start time: {run_start_time_local} {ltz}")
    logger.info(f"run name: {args.run_name}")
    logger.info(f"configuration file path: {args.config_file}")
    logger.info(f"notes: {args.notes}")
    logger.info(f"results of the run are stored at: {run_dir}")

    with open('run_registry.txt', 'a') as file:
        file.write(write_text)

def get_logger(run_dir, config):
    class CustomFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            return datetime.fromtimestamp(record.created, pytz.timezone(config["local_time_zone"])).strftime("%Y-%m-%d %H:%M:%S.%f")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.path.join(run_dir, 'runner_log.log'))
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = CustomFormatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

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
