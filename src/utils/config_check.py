import os

def check_config(config, logger):
    logger.debug('==========  performing configuration checks  ==========')

    assert config['task'] in ['train', 'test_only'], 'task should be in [train, test_only]'
    logger.debug(f'task is set to: {config["task"]}')

    assert config['training_type'] in ['finetune_pretrained', 'start_from_checkpoint', 'train_from_scratch'], 'training_type should be in [finetune_pretrained, start_from_checkpoint, train_from_scratch]'
    logger.debug(f'training_type is set to: {config["training_type"]}')

    assert config['dataset_to_use'] in ['lasco', 'fashioniq', 'cirr', 'circo'], 'dataset_to_use should be in [lasco, fashioniq, cirr, circo]'
    logger.debug(f'dataset_to_use is set to: {config["dataset_to_use"]}')

    assert config['load_data_into_memory'] in [True, False], 'load_data_into_memory should be in [True, False]'
    logger.debug(f'load_data_into_memory is set to: {config["load_data_into_memory"]}')

    assert config['model_to_use']['choose_model_category'] in config['model_registry']['models_categories_available'], f'model_to_use.choose_model_category is not among available model categories in the model registry'
    logger.debug('model_to_use.choose_model_category is set to: {}'.format(config['model_to_use']['choose_model_category']))

    assert config['model_to_use']['choose_model_name'] in config['model_registry']['models_by_category'][config['model_to_use']['choose_model_category']]['options'], f'model_to_use.choose_model_name is not among available models for the selected model category'
    logger.debug(f'model_to_use.choose_model_name is set to: {config["model_to_use"]["choose_model_name"]}')

    if config['training_type'] in ['start_from_checkpoint', 'finetune_pretrained']:
        assert os.path.exists(config['checkpoint_path']), 'checkpoint_path does not exist'
        logger.debug(f'checkpoint_path is set to: {config["checkpoint_path"]}')

    logger.debug('==========  configuration checks passed! good to go!  ==========')





