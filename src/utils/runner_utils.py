import os
import argparse
from datetime import datetime
import pytz
import yaml
import copy
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='configuration file path for the task ', dest='config_file')
    args = parser.parse_args()
    return args


def register_run(args, run_start_time_local, config, run_dir):
    write_text = f"\n"
    
    ltz = config["local_time_zone"]
    write_text += f"run start time: {run_start_time_local} {ltz}\n"
    
    task = config["task"]
    write_text += f"exp name: {task}\n"
    
    run_name = config["run_name"]
    write_text += f"run name: {run_name}\n"
    
    exp_name = config["exp_name"]
    write_text += f"exp name: {exp_name}\n"
    
    exp_type = config["exp_type"]
    write_text += f"exp_type: {exp_type}\n"
    
    model_type = config["model_type"]
    write_text += f"model_type: {model_type}\n"
        
    dataset = config["dataset_to_use"]
    write_text += f"exp name: {dataset}\n"
    
    config_file_path = os.path.join(run_dir, 'config.yaml')
    write_text += f"configuration file path: {config_file_path}\n"
    
    write_text += f"results of the run are stored at: {run_dir}\n"
    
    run_comments = config['comments']
    write_text += f"comments: {run_comments}\n"
    
    resume_training_from_checkpoint = config["resume_training_from_checkpoint"]
    write_text += f"resume_training_from_checkpoint: {resume_training_from_checkpoint}\n"
    
    test_at_checkpoint = config["test_at_checkpoint"]
    write_text += f"test_at_checkpoint: {test_at_checkpoint}\n"
    
    write_text += f"\n==================================================================================================================\n"

    with open(config['run_registry_file'], 'a') as file:
        file.write(write_text)
        
        
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
    config['run_name'] = run_name
    config['wandb_name'] = run_name
    
    # create a directory to store the results of the run
    runs_dir = config["runs_dir"]
    run_dir = os.path.join(runs_dir, run_name)
    os.mkdir(run_dir)
    config['run_dir'] = run_dir
    
    # save the configuration file in the run directory. Useful to later check the configuration used for the run.
    yaml.dump(config, open(os.path.join(run_dir, 'config.yaml'), 'w'), default_flow_style=False, sort_keys=False)
    
    # register the run in the run_registry.txt file
    register_run(args, run_start_time_local, config, run_dir) 