from pytorch_lightning import LightningDataModule
from src.datasets.lasco_datasets import lasco_dataset_train, lasco_dataset_val


class LASCODataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.pin_memory = config['pin_memory']
        self.drop_last = config['drop_last']
        self.persist_workers = config['persist_workers']
        self.shuffle_train = config['shuffle']['train']
        self.shuffle_val = config['shuffle']['val']
        self.shuffle_test = config['shuffle']['test']

    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = lasco_dataset_train(self.config)
            self.val_dataset = lasco_dataset_val(self.config)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle_train,
            num_workers=self.num_workers, 
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persist_workers=self.persist_workers
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
            persist_workers=self.persist_workers
            )

    #def test_dataloader(self):
    #    return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)