import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from transformers import AutoProcessor, AutoTokenizer
from transformers import AutoTokenizer, AutoImageProcessor
from torch.utils.data import DataLoader, Dataset

class fiq_retrieval_dataset_clip(Dataset):
    def __init__(self, dataset_split, lasco_data_path, clip_ckpt_path):
        """
        Args:
            dataset_split (string): options: ['dress', 'shirt', 'toptee']
        """

        if dataset_split == 'dress':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'captions', 'cap.dress.val.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['candidate'],
                        'query-image': os.path.join(lasco_data_path, 'images', x['candidate']+'.png'),
                        'target-image-id': x['target'],
                        'query-text': x['captions'][0]
                    },
                    lasco_json
                    )
                )
            
        elif dataset_split == 'shirt':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'captions', 'cap.shirt.val.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['candidate'],
                        'query-image': os.path.join(lasco_data_path, 'images', x['candidate']+'.png'),
                        'target-image-id': x['target'],
                        'query-text': x['captions'][0]
                    },
                    lasco_json
                    )
                )
            
        elif dataset_split == 'toptee':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'captions', 'cap.toptee.val.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['candidate'],
                        'query-image': os.path.join(lasco_data_path, 'images', x['candidate']+'.png'),
                        'target-image-id': x['target'],
                        'query-text': x['captions'][0]
                    },
                    lasco_json
                    )
                )
            
        else:
            print('Invalid dataset split. Please choose from [train, val]')

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
            'query-text': triplet['query-text']
        }

    def collate_fn(self, batch):
        query_text_raw = list(map(lambda x: x['query-text'], batch))
        query_text = self.tokenizer(query_text_raw, padding=True, return_tensors="pt")
        
        query_image_ids = list(map(lambda x: x['query-image-id'], batch))
        target_image_ids = list(map(lambda x: x['target-image-id'], batch))
        
        query_image = {'pixel_values': torch.cat(list(map(lambda x: x['query-image']['pixel_values'], batch)), dim=0)}
        
        return {
            'query-image-id': query_image_ids,
            'query-image': query_image,
            'target-image-id': target_image_ids,
            'query-text': query_text,
            'query-text-raw': query_text_raw
        }
        
   
   
        
        
class fiq_retrieval_dataset_flava(Dataset):
    def __init__(self, dataset_split, lasco_data_path):
        """
        Args:
            dataset_split (string): options: ['train', 'val']
            lasco_data_path (string): path to the lasco dataset
        """

        if dataset_split == 'dress':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'captions', 'cap.dress.val.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['candidate'],
                        'query-image': os.path.join(lasco_data_path, 'images', x['candidate']+'.png'),
                        'target-image-id': x['target'],
                        'query-text': x['captions'][0]
                    },
                    lasco_json
                    )
                )
            
        elif dataset_split == 'shirt':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'captions', 'cap.shirt.val.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['candidate'],
                        'query-image': os.path.join(lasco_data_path, 'images', x['candidate']+'.png'),
                        'target-image-id': x['target'],
                        'query-text': x['captions'][0]
                    },
                    lasco_json
                    )
                )
            
        elif dataset_split == 'toptee':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'captions', 'cap.toptee.val.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['candidate'],
                        'query-image': os.path.join(lasco_data_path, 'images', x['candidate']+'.png'),
                        'target-image-id': x['target'],
                        'query-text': x['captions'][0]
                    },
                    lasco_json
                    )
                )
            
        else:
            print('Invalid dataset split. Please choose from [train, val]')

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
            'query-text': triplet['query-text']
        }

    def collate_fn(self, batch):
        query_text_raw = list(map(lambda x: x['query-text'], batch))
        query_text = self.tokenizer(query_text_raw, padding=True, return_tensors="pt")
        
        query_image_ids = list(map(lambda x: x['query-image-id'], batch))
        target_image_ids = list(map(lambda x: x['target-image-id'], batch))
        
        query_image = {'pixel_values': torch.cat(list(map(lambda x: x['query-image']['pixel_values'], batch)), dim=0)}
        
        return {
            'query-image-id': query_image_ids,
            'query-image': query_image,
            'target-image-id': target_image_ids,
            'query-text': query_text,
            'query-text-raw': query_text_raw
        }