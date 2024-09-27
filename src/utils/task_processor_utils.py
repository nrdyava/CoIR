
def exp_type_to_model_module_mapper(config):
    if config['exp_type'] == 'inbatch_2en_ST_F':
        return 'inbatch_2en_ST_F'
    else:
        raise Exception("Please select a valid exp_type")
    
    
def exp_type_to_data_module_mapper(config):
    if config['exp_type'] == 'inbatch_2en_ST_F':
        return 'generic_dm_with_dist_sampler'
    else:
        raise Exception("Please select a valid exp_type")