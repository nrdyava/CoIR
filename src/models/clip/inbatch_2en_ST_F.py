import lightning as L

class CLIPModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.save_hyperparameters()