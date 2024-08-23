import os
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.models import model_registry
from src.datamodules import datamodule_registry
from src.utils.trainer_utils import get_wandb_logger, get_tb_logger, get_checkpoint_callback


def task_processor(config):
    if config['task'] == 'train':
        
        # Load the model
        try:
            model = model_registry[config['model_type']](config)
        except:
            #logger.error('model type in not in the model registry')
            return
        
        try :
            datamodule = datamodule_registry[config['dataset_to_use']](config)
        except:
            #logger.error('dataset type in not in the datamodule registry')
            return
        
        wandb_logger = get_wandb_logger(config)
        #tb_logger = get_tb_logger(config)
        checkpoint_callback = get_checkpoint_callback(config)

        # Configure the trainer
        trainer = Trainer(
            accelerator=config['trainer']['accelerator'],
            strategy=config['trainer']['strategy'],
            devices=config['trainer']['devices'],
            num_nodes=config['trainer']['num_nodes'],
            precision=config['trainer']['precision'],
            logger=[wandb_logger],
            callbacks=[checkpoint_callback],
            fast_dev_run=config['trainer']['fast_dev_run'],
            max_epochs=config['trainer']['max_epochs'],
            min_epochs=config['trainer']['min_epochs'],
            limit_train_batches = config['trainer']['limit_train_batches'],
            limit_val_batches = config['trainer']['limit_val_batches'],
            limit_test_batches = config['trainer']['limit_test_batches'],
            limit_predict_batches = config['trainer']['limit_predict_batches'],
            check_val_every_n_epoch=config['trainer']['check_val_every_n_epoch'],
            num_sanity_val_steps=config['trainer']['num_sanity_val_steps'],
            log_every_n_steps=config['trainer']['log_every_n_steps'],
            enable_checkpointing=config['trainer']['enable_checkpointing'],
            enable_progress_bar=config['trainer']['enable_progress_bar'],
            enable_model_summary=config['trainer']['enable_model_summary'],
            deterministic=config['trainer']['deterministic'],
            benchmark=config['trainer']['benchmark'],
            use_distributed_sampler=config['trainer']['use_distributed_sampler'],
            profiler=None if config['trainer']['profiler']=='None' else config['trainer']['profiler'],
            default_root_dir=config['run_dir']
            )

        datamodule.setup('fit')

        trainer.fit(
            model, 
            datamodule.train_dataloader(), 
            datamodule.val_dataloader(), 
            #ckpt_path = config['trainer']['pl_ckpt_path']
            )

    else:
        #logger.error('task not recognized')
        return