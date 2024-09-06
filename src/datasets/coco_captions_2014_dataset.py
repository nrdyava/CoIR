import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from transformers import AutoProcessor, AutoTokenizer


class coco_captions_dataset_image_tensors(Dataset):
    def __init__(self, config, dataset_split):
        self.config = config
        self.coco_caps_2014_path = self.config['data']['coco_2014_annotations']['dir']
        self.img_tensors_dir = self.config['data']['lasco']['img_tensors_dir']

        if dataset_split == 'train':
            caps = json.load(open(os.path.join(self.coco_caps_2014_path, 'captions_train2014.json'), 'r'))
            ids_and_image_name = dict(map(lambda x: (x['id'], x['file_name']), caps['images']))
            self.triplets = list(
                map(
                    lambda x: dict(
                        [
                            ('image_id', x['image_id']), 
                            ('image_name', os.path.join('train2014', ids_and_image_name[x['image_id']])), 
                            ('caption', x['caption'])
                        ]
                        ), 
                    caps['annotations']
                    )
                )

        elif dataset_split == 'val':
            caps = json.load(open(os.path.join(self.coco_caps_2014_path, 'captions_val2014.json'), 'r'))
            ids_and_image_name = dict(map(lambda x: (x['id'], x['file_name']), caps['images']))
            self.triplets = list(
                map(
                    lambda x: dict(
                        [
                            ('image_id', x['image_id']), 
                            ('image_name', os.path.join('val2014', ids_and_image_name[x['image_id']])), 
                            ('caption', x['caption'])
                        ]
                        ), 
                    caps['annotations']
                    )
                )
            
        else:
            lasco_splits = {'train': 1, 'val': 2}
            print(lasco_splits[self.config['dataset_split']])

        # Image processor & tokenizer
        self.image_processor = AutoProcessor.from_pretrained(config["checkpoint_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["checkpoint_path"])

    def load_image_tensors(self, img_tensor_path):
        return torch.load(img_tensor_path)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        return {
            'image': self.load_image_tensors(os.path.join(self.img_tensors_dir, 'coco', triplet['image_name'].replace('.jpg', '.pt'))),
            'caption': triplet['caption']
        }

    def collate_fn(self, batch):
        caption = list(map(lambda x: x['caption'], batch))
        caption = self.tokenizer(caption, padding=True, return_tensors="pt")
        
        image = {'pixel_values': torch.cat(list(map(lambda x: x['image']['pixel_values'], batch)), dim=0)}

        return {
            'image': image,
            'caption': caption
        }
