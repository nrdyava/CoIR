exp_type_to_model_module_map = {
    'clip_inbatch_2en_ST_F': 'clip_inbatch_2en_ST_F'
}

exp_type_to_data_module_map = {
    'clip_inbatch_2en_ST_F': 'generic_dm_with_dist_sampler'
}

def exp_type_to_model_module_mapper(config):
    try:
        return exp_type_to_model_module_map[config['exp_type']]
    except:
        raise Exception("Please select a valid exp_type")
    
    
def exp_type_to_data_module_mapper(config):
    try:
        return exp_type_to_data_module_map[config['exp_type']]
    except:
        raise Exception("Please select a valid exp_type")
    

def get_trainer_args(config):
    trainer_args = {
        'accelerator': config['trainer']['accelerator'],
        'strategy': config['trainer']['strategy'],
        'devices': config['trainer']['devices'],
        'num_nodes': config['trainer']['num_nodes'],
        'precision': config['trainer']['precision'],
        # 'logger': [wandb_logger],
        # callbacks=[checkpoint_callback],
        'fast_dev_run': config['trainer']['fast_dev_run'],
        'max_epochs': config['trainer']['max_epochs'],
        'min_epochs': config['trainer']['min_epochs'],
        'limit_train_batches': config['trainer']['limit_train_batches'],
        'limit_val_batches': config['trainer']['limit_val_batches'],
        'limit_test_batches': config['trainer']['limit_test_batches'],
        'limit_predict_batches': config['trainer']['limit_predict_batches'],
        'check_val_every_n_epoch': config['trainer']['check_val_every_n_epoch'],
        'num_sanity_val_steps': config['trainer']['num_sanity_val_steps'],
        'log_every_n_steps': config['trainer']['log_every_n_steps'],
        'enable_checkpointing': config['trainer']['enable_checkpointing'],
        'enable_progress_bar': config['trainer']['enable_progress_bar'],
        'enable_model_summary': config['trainer']['enable_model_summary'],
        'deterministic': config['trainer']['deterministic'],
        'benchmark': config['trainer']['benchmark'],
        'use_distributed_sampler': config['trainer']['use_distributed_sampler'],
        'profiler': None if config['trainer']['profiler']=='None' else config['trainer']['profiler'],
        'default_root_dir': config['run_dir']
        }
        
    return trainer_args