from src.datamodules.lasco_data_module import LASCODataModule
from src.datamodules.lasco_data_module_neg_samps import LASCODataModuleNS

datamodule_registry = {
    'lasco': LASCODataModule,
    'lasco_neg_samps': LASCODataModuleNS
}