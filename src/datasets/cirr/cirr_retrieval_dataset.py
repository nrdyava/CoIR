import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from transformers import AutoProcessor, AutoTokenizer
from transformers import AutoTokenizer, AutoImageProcessor
from torch.utils.data import DataLoader, Dataset

class cirr_retrieval_dataset_clip(Dataset):
    def __init__(self, dataset_split, lasco_data_path, clip_ckpt_path):
        """
        Args:
            dataset_split (string): options: ['train', 'val']
            lasco_data_path (string): path to the lasco dataset
            clip_ckpt_path (string): path to the clip checkpoint
        """

        if dataset_split == 'train':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'captions', 'cap.rc2.train.json'), 'r'))
            img_name_to_img_path = json.load(open(os.path.join(lasco_data_path, 'image_splits', 'split.rc2.train.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['reference'],
                        'query-image': os.path.join(lasco_data_path, 'img_raw', img_name_to_img_path[x['reference']][2:]),
                        'target-image-id': x['target_hard'],
                        'query-text': x['caption'],
                        'id': x['pairid'],
                        'subset': x['img_set']['members']
                    },
                    lasco_json
                    )
                )
            
        elif dataset_split == 'val':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'captions', 'cap.rc2.val.json'), 'r'))
            img_name_to_img_path = json.load(open(os.path.join(lasco_data_path, 'image_splits', 'split.rc2.val.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['reference'],
                        'query-image': os.path.join(lasco_data_path, 'img_raw', img_name_to_img_path[x['reference']][2:]),
                        'target-image-id': x['target_hard'],
                        'query-text': x['caption'],
                        'id': x['pairid'],
                        'subset': x['img_set']['members']
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
            'query-text': triplet['query-text'],
            'id': triplet['id'],
            'subset': triplet['subset']
        }

    def collate_fn(self, batch):
        query_text_raw = list(map(lambda x: x['query-text'], batch))
        query_text = self.tokenizer(query_text_raw, padding=True, return_tensors="pt")
        
        query_image_ids = list(map(lambda x: x['query-image-id'], batch))
        target_image_ids = list(map(lambda x: x['target-image-id'], batch))
        ids = list(map(lambda x: x['id'], batch))
        subsets = list(map(lambda x: x['subset'], batch))
        
        query_image = {'pixel_values': torch.cat(list(map(lambda x: x['query-image']['pixel_values'], batch)), dim=0)}
        
        return {
            'query-image-id': query_image_ids,
            'query-image': query_image,
            'target-image-id': target_image_ids,
            'query-text': query_text,
            'query-text-raw': query_text_raw,
            'id': ids,
            'subset': subsets
        }
        
   
   
        
        
class cirr_retrieval_dataset_flava(Dataset):
    def __init__(self, dataset_split, lasco_data_path):
        """
        Args:
            dataset_split (string): options: ['train', 'val']
            lasco_data_path (string): path to the lasco dataset
        """

        if dataset_split == 'train':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'captions', 'cap.rc2.train.json'), 'r'))
            img_name_to_img_path = json.load(open(os.path.join(lasco_data_path, 'image_splits', 'split.rc2.train.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['reference'],
                        'query-image': os.path.join(lasco_data_path, 'img_raw', img_name_to_img_path[x['reference']][2:]),
                        'target-image-id': x['target_hard'],
                        'query-text': x['caption'],
                        'id': x['pairid'],
                        'subset': x['img_set']['members']
                    },
                    lasco_json
                    )
                )
            
        elif dataset_split == 'val':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'captions', 'cap.rc2.val.json'), 'r'))
            img_name_to_img_path = json.load(open(os.path.join(lasco_data_path, 'image_splits', 'split.rc2.val.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['reference'],
                        'query-image': os.path.join(lasco_data_path, 'img_raw', img_name_to_img_path[x['reference']][2:]),
                        'target-image-id': x['target_hard'],
                        'query-text': x['caption'],
                        'id': x['pairid'],
                        'subset': x['img_set']['members']
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
            'query-text': triplet['query-text'],
            'id': triplet['id'],
            'subset': triplet['subset']
        }

    def collate_fn(self, batch):
        query_text_raw = list(map(lambda x: x['query-text'], batch))
        query_text = self.tokenizer(query_text_raw, padding=True, return_tensors="pt")
        
        query_image_ids = list(map(lambda x: x['query-image-id'], batch))
        target_image_ids = list(map(lambda x: x['target-image-id'], batch))
        ids = list(map(lambda x: x['id'], batch))
        subsets = list(map(lambda x: x['subset'], batch))
        
        query_image = {'pixel_values': torch.cat(list(map(lambda x: x['query-image']['pixel_values'], batch)), dim=0)}
        
        return {
            'query-image-id': query_image_ids,
            'query-image': query_image,
            'target-image-id': target_image_ids,
            'query-text': query_text,
            'query-text-raw': query_text_raw,
            'id': ids,
            'subset': subsets
        }