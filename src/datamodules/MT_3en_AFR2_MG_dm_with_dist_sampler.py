from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.datasets.MT_3en_FR2_MG_img_caps_dataset import dataset as MT_3en_FR2_MG_img_caps_dataset

class DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.batch_size = config['dataloader']['batch_size']
        self.num_workers = config['dataloader']['num_workers']
        self.pin_memory = config['dataloader']['pin_memory']
        self.drop_last = config['dataloader']['drop_last']
        self.persistent_workers = config['dataloader']['persistent_workers']
        self.shuffle_train = config['dataloader']['shuffle']['train']
        self.shuffle_val = config['dataloader']['shuffle']['val']
        self.shuffle_test = config['dataloader']['shuffle']['test']
        
    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = MT_3en_FR2_MG_img_caps_dataset(self.config, 'train')
            self.val_dataset = MT_3en_FR2_MG_img_caps_dataset(self.config, 'val')
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle_train,
            num_workers=self.num_workers, 
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle_val,
            num_workers=self.num_workers, 
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers
            )
        
        