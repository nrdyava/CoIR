import os
import argparse
from datetime import datetime
import pytz
import yaml
import logging
import copy
from pytorch_lightning.utilities.rank_zero import rank_zero_only

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='configuration file path for the task ', dest='config_file')
    #parser.add_argument('--run_name', type=str, required=True, help='name of the task you want to run', dest='run_name')
    #parser.add_argument('--notes', type=str, default = "No notes mentioned", help='write a note about the task you want to run', dest='notes')

    args = parser.parse_args()
    return args

def register_run(args, run_start_time_local, config, run_dir):
    write_text = f"\n"
    ltz = config["local_time_zone"]
    write_text += f"run start time: {run_start_time_local} {ltz}\n"
    run_name = config["run_name"]
    write_text += f"run name: {run_name}\n"
    write_text += f"configuration file path: {args.config_file}\n"
    write_text += f"results of the run are stored at: {run_dir}\n"
    write_text += f"\n==================================================================================================================\n"

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


def start_of_a_run():
    args = parse_args()
    config = copy.deepcopy(yaml.safe_load(open(args.config_file, 'r')))
    start_of_a_run_rank_zero(args, config)
    
    return config

@rank_zero_only
def start_of_a_run_rank_zero(args, config):
    run_start_time_utc = datetime.now(pytz.utc)
    run_start_time_local = run_start_time_utc.astimezone(pytz.timezone(config["local_time_zone"])).strftime("%Y-%m-%d-%H-%M-%S-%f")
    run_name = run_start_time_local + ' + ' + config['exp_name']
    
    # create a directory to store the results of the run
    runs_dir = config["runs_dir"]
    run_dir = os.path.join(runs_dir, run_name)
    os.mkdir(run_dir)
    config['run_dir'] = run_dir
    config['wandb_name'] = run_name
    config['run_name'] = run_name
    
    # register the run in the run_registry.txt file
    register_run(args, run_start_time_local, config, run_dir)
    
    # save the configuration file in the run directory. Usefule to later check the configuration used for the run.
    yaml.dump(config, open(os.path.join(run_dir, 'config.yaml'), 'w'), default_flow_style=False, sort_keys=False)
    
    
    
    
    
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    