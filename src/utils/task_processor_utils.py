exp_type_to_model_module_map = {
    'clip_inbatch_2en_ST_F': 'clip_inbatch_2en_ST_F',
    'clip_inbatch_2en_ST_F_T_HAT_Proj': 'clip_inbatch_2en_ST_F_T_HAT_Proj',
    'clip_inbatch_2en_ST_F_WA': 'clip_inbatch_2en_ST_F_WA',
    'clip_inbatch_2en_ST_F_img_proj': 'clip_inbatch_2en_ST_F_img_proj',
    'clip_inbatch_2en_ST_F_img_proj_txt_proj': 'clip_inbatch_2en_ST_F_img_proj_txt_proj',
    'clip_inbatch_3en_MT_F_img_cap': 'clip_inbatch_3en_MT_F_img_cap',
    'clip_inbatch_2en_ST_F_ON_GT': 'clip_inbatch_2en_ST_F_ON_GT',
    'flava_inbatch_2en_ST_F_ON_GT': 'flava_inbatch_2en_ST_F_ON_GT',
    'clip_inbatch_2en_ST_F_ON_GT_QN': 'clip_inbatch_2en_ST_F_ON_GT_QN',
    'clip_inbatch_2en_ST_F_ON_GT_QN_WA': 'clip_inbatch_2en_ST_F_ON_GT_QN_WA'
}

exp_type_to_data_module_map = {
    'clip_inbatch_2en_ST_F': 'generic_dm_with_dist_sampler',
    'clip_inbatch_2en_ST_F_T_HAT_Proj': 'generic_dm_with_dist_sampler',
    'clip_inbatch_2en_ST_F_WA': 'generic_dm_with_dist_sampler',
    'clip_inbatch_2en_ST_F_img_proj': 'generic_dm_with_dist_sampler',
    'clip_inbatch_2en_ST_F_img_proj_txt_proj': 'generic_dm_with_dist_sampler', 
    'clip_inbatch_3en_MT_F_img_cap': 'MT_3en_F_img_caps_dm_with_dist_sampler',
    'clip_inbatch_2en_ST_F_ON_GT': 'generic_dm_with_dist_sampler',
    'flava_inbatch_2en_ST_F_ON_GT': 'generic_dm_with_dist_sampler_flava',
    'clip_inbatch_2en_ST_F_ON_GT_QN': 'generic_dm_with_dist_sampler',
    'clip_inbatch_2en_ST_F_ON_GT_QN_WA': 'generic_dm_with_dist_sampler'
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
    
from lightning.pytorch.strategies import DDPStrategy
def get_trainer_args(config):
    trainer_args = {
        'accelerator': config['trainer']['accelerator'],
        'strategy': DDPStrategy(find_unused_parameters=True) if (config['trainer']['strategy'] == 'ddp' and config['find_unused_parameters'] == True) else config['trainer']['strategy'],
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