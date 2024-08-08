import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from transformers import AutoProcessor, AutoTokenizer



class lasco_clip_dataset(Dataset):
    def __init__(self, config):
        self.config = config
        
        


