import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from transformers import AutoProcessor, AutoTokenizer
from transformers import AutoTokenizer, AutoImageProcessor
from torch.utils.data import DataLoader, Dataset

class cirr_corpus_dataset_clip(Dataset):
    def __init__(self, dataset_split, lasco_data_path, clip_ckpt_path):
        """
        Args:
            dataset_split (string): options: ['train', 'val']
            lasco_data_path (string): path to the lasco dataset
            clip_ckpt_path (string): path to the clip checkpoint
        """

        if dataset_split == 'train':
            corpus = json.load(open(os.path.join(lasco_data_path, 'metadata', 'train_corpus.json'), 'r'))
            self.image_book = list(
                map(
                    lambda x: {
                        'image-key': x['image-id'], 
                        'image': os.path.join(lasco_data_path, x['image-path'])
                        },
                    corpus
                    )
                )
            
        elif dataset_split == 'val':
            corpus = json.load(open(os.path.join(lasco_data_path, 'metadata', 'val_corpus.json'), 'r'))
            self.image_book = list(
                map(
                    lambda x: {
                        'image-key': x['image-id'], 
                        'image': os.path.join(lasco_data_path, x['image-path'])
                        },
                    corpus
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
        return len(self.image_book)

    def __getitem__(self, idx):
        return {
            'image-key': self.image_book[idx]['image-key'],
            'image': self.load_and_process_image(self.image_book[idx]['image'])
        }

    def collate_fn(self, batch):
        image_key = list(map(lambda x: x['image-key'], batch))
        image = {'pixel_values': torch.cat(list(map(lambda x: x['image']['pixel_values'], batch)), dim=0)}
        return {'image-key': image_key, 'image': image}
    
    

    
    
class cirr_corpus_dataset_flava(Dataset):
    def __init__(self, dataset_split, lasco_data_path):
        """
        Args:
            dataset_split (string): options: ['train', 'val']
            lasco_data_path (string): path to the lasco dataset
        """

        if dataset_split == 'train':
            corpus = json.load(open(os.path.join(lasco_data_path, 'metadata', 'train_corpus.json'), 'r'))
            self.image_book = list(
                map(
                    lambda x: {
                        'image-key': x['image-id'], 
                        'image': os.path.join(lasco_data_path, x['image-path'])
                        },
                    corpus
                    )
                )
            
        elif dataset_split == 'val':
            corpus = json.load(open(os.path.join(lasco_data_path, 'metadata', 'val_corpus.json'), 'r'))
            self.image_book = list(
                map(
                    lambda x: {
                        'image-key': x['image-id'], 
                        'image': os.path.join(lasco_data_path, x['image-path'])
                        },
                    corpus
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
        return len(self.image_book)

    def __getitem__(self, idx):
        return {
            'image-key': self.image_book[idx]['image-key'],
            'image': self.load_and_process_image(self.image_book[idx]['image'])
        }

    def collate_fn(self, batch):
        image_key = list(map(lambda x: x['image-key'], batch))
        image = {'pixel_values': torch.cat(list(map(lambda x: x['image']['pixel_values'], batch)), dim=0)}
        return {'image-key': image_key, 'image': image}