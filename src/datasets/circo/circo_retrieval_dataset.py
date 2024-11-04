import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from transformers import AutoProcessor, AutoTokenizer
from transformers import AutoTokenizer, AutoImageProcessor
from torch.utils.data import DataLoader, Dataset

class circo_retrieval_dataset_clip(Dataset):
    def __init__(self, dataset_split, lasco_data_path, clip_ckpt_path):
        """
        Args:
            dataset_split (string): options: ['train', 'val']
            lasco_data_path (string): path to the lasco dataset
            clip_ckpt_path (string): path to the clip checkpoint
        """

        if dataset_split == 'test':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'metadata', 'val-retrieval.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['query-image-id'],
                        'query-image': os.path.join(lasco_data_path, x['query-image-path']),
                        'target-image-id': x['target-image-id'],
                        'query-text': x['query-text'],
                        'gt-image-ids': x['gt-image-ids'],
                        'id': x['id']
                    },
                    lasco_json
                    )
                )
            
        else:
            print('Invalid dataset split. Please choose from [test]')

        # Image processor & tokenizer
        self.image_processor = AutoProcessor.from_pretrained(clip_ckpt_path)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_ckpt_path)

    def load_and_process_image(self, img_path):
        image = Image.open(img_path)
        processed_image = self.image_processor(images=image, return_tensors="pt")
        return processed_image

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        return {
            'query-image-id': triplet['query-image-id'],
            'query-image': self.load_and_process_image(triplet['query-image']),
            'target-image-id': triplet['target-image-id'],
            'query-text': triplet['query-text'],
            'gt-image-ids': triplet['gt-image-ids'],
            'id': triplet['id']
        }

    def collate_fn(self, batch):
        query_text_raw = list(map(lambda x: x['query-text'], batch))
        query_text = self.tokenizer(query_text_raw, padding=True, return_tensors="pt")
        
        query_image_ids = list(map(lambda x: x['query-image-id'], batch))
        target_image_ids = list(map(lambda x: x['target-image-id'], batch))
        ids = list(map(lambda x: x['id'], batch))
        
        gt_img_ids = list(map(lambda x: x['gt-image-ids'], batch))
        
        query_image = {'pixel_values': torch.cat(list(map(lambda x: x['query-image']['pixel_values'], batch)), dim=0)}
        
        return {
            'query-image-id': query_image_ids,
            'query-image': query_image,
            'target-image-id': target_image_ids,
            'query-text': query_text,
            'query-text-raw': query_text_raw,
            'gt-image-ids': gt_img_ids,
            'id': ids
        }
        
   
   
        
        
class circo_retrieval_dataset_flava(Dataset):
    def __init__(self, dataset_split, lasco_data_path):
        """
        Args:
            dataset_split (string): options: ['train', 'val']
            lasco_data_path (string): path to the lasco dataset
        """

        if dataset_split == 'test':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'metadata', 'val-retrieval.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['query-image-id'],
                        'query-image': os.path.join(lasco_data_path, x['query-image-path']),
                        'target-image-id': x['target-image-id'],
                        'query-text': x['query-text'],
                        'gt-image-ids': x['gt-image-ids'],
                        'id': x['id']
                    },
                    lasco_json
                    )
                )
            
        else:
            print('Invalid dataset split. Please choose from [test]')

        # Image processor & tokenizer
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/flava-full")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")

    def load_and_process_image(self, img_path):
        image = Image.open(img_path)
        image = image.convert('RGB')
        
        processed_image = self.image_processor(images=image, return_tensors="pt")
        return processed_image

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        return {
            'query-image-id': triplet['query-image-id'],
            'query-image': self.load_and_process_image(triplet['query-image']),
            'target-image-id': triplet['target-image-id'],
            'query-text': triplet['query-text'],
            'gt-image-ids': triplet['gt-image-ids'],
            'id': triplet['id']
        }

    def collate_fn(self, batch):
        query_text_raw = list(map(lambda x: x['query-text'], batch))
        query_text = self.tokenizer(query_text_raw, padding=True, return_tensors="pt")
        
        query_image_ids = list(map(lambda x: x['query-image-id'], batch))
        target_image_ids = list(map(lambda x: x['target-image-id'], batch))
        ids = list(map(lambda x: x['id'], batch))
        
        gt_img_ids = list(map(lambda x: x['gt-image-ids'], batch))
        
        query_image = {'pixel_values': torch.cat(list(map(lambda x: x['query-image']['pixel_values'], batch)), dim=0)}
        
        return {
            'query-image-id': query_image_ids,
            'query-image': query_image,
            'target-image-id': target_image_ids,
            'query-text': query_text,
            'query-text-raw': query_text_raw,
            'gt-image-ids': gt_img_ids,
            'id': ids
        }