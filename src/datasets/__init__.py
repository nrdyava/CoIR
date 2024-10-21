from src.datasets.clip_inbatch_2en import dataset as clip_inbatch_2en 
from src.datasets.MT_3en_F_img_caps_dataset import dataset as MT_3en_F_img_caps_dataset
from src.datasets.flava_inbatch_2en import dataset as flava_inbatch_2en

datasets_registry = {
    'clip_inbatch_2en': clip_inbatch_2en,
    'MT_3en_F_img_caps_dataset': MT_3en_F_img_caps_dataset,
    'flava_inbatch_2en': flava_inbatch_2en,
}