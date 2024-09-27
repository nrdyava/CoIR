from src.models import model_registry
from src.utils.task_processor_utils import exp_type_to_model_module_mapper

def task_processor(config):
    if config['task'] == 'finetune':
        try:
            model = model_registry[exp_type_to_model_module_mapper(config)](config)
        except:
            raise Exception("Loading model failed. Please check the model type or the model definition")
        
        try :
            datamodule = datamodule_registry['lasco_inbatch'](config)
        except:
            return Exception("Loading datamodule failed. Please check the datamodule type or the datamodule definition")
    
    elif config['task'] == 'continue_training':
        # To be implemented
        return
    
    
    elif config['task'] == 'test_only':
        # To be implemented
        return
    
    
    else:
        raise Exception("Please select a valid task to perform")