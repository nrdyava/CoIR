from src.datamodules.generic_dm_with_dist_sampler import DataModule as generic_dm_with_dist_sampler
from src.datamodules.MT_3en_F_img_caps_dm_with_dist_sampler import DataModule as MT_3en_F_img_caps_dm_with_dist_sampler

datamodule_registry = {
    'generic_dm_with_dist_sampler': generic_dm_with_dist_sampler,
    'MT_3en_F_img_caps_dm_with_dist_sampler': MT_3en_F_img_caps_dm_with_dist_sampler
}