from lightning.pytorch.trainer.trainer import Trainer

from src.models import model_module_registry
from src.datamodules import datamodule_registry
from src.utils.task_processor_utils import exp_type_to_model_module_mapper, exp_type_to_data_module_mapper, get_trainer_args
from src.utils.trainer_utils import get_wandb_logger, get_checkpoint_callback

def task_processor(config):
    if config['task'] == 'finetune':
        try:
            model = model_module_registry[exp_type_to_model_module_mapper(config)](config)
        except:
            raise Exception("Loading model failed.")
        
        try :
            datamodule = datamodule_registry[exp_type_to_data_module_mapper(config)](config)
        except:
            return Exception("Loading datamodule failed.")

        wandb_logger = get_wandb_logger(config)
        checkpoint_callback = get_checkpoint_callback(config)
        
        trainer_params = get_trainer_args(config)
        trainer_params.update({
            'logger': [wandb_logger], 
            'callbacks': [checkpoint_callback]
            })
        
        trainer = Trainer(**trainer_params)
        
        datamodule.setup('fit')
        trainer.fit(
            model = model, 
            train_dataloaders = datamodule.train_dataloader(), 
            val_dataloaders = datamodule.val_dataloader(),
            #datamodule = datamodule
            )
        
    
    elif config['task'] == 'continue_training':
        # To be implemented
        return Exception("Method not implemented yet")
    
    
    elif config['task'] == 'test_only':
        # To be implemented
        return Exception("Method not implemented yet")
    
    
    else:
        raise Exception("Please select a valid task to perform")