import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from transformers import AutoProcessor, AutoTokenizer
from transformers import AutoTokenizer, AutoImageProcessor
from torch.utils.data import DataLoader, Dataset



class mscoco_5k_retrieval_dataset_clip(Dataset):
    def __init__(self, dataset_split, lasco_data_path, clip_ckpt_path):
        """
        Args:
            dataset_split (string): options: ['train', 'val']
            lasco_data_path (string): path to the lasco dataset
            clip_ckpt_path (string): path to the clip checkpoint
        """

        if dataset_split == 'test':
            lasco_json = json.load(open(os.path.join(lasco_data_path, 'test_5k_mscoco_2014_retrieval_data.csv'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'image-id': x['image-id'],
                        'image': os.path.join(lasco_data_path, x['image-path']),
                        'text-id': x['text-id'],
                        'text': x['raw-text'],
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
            'image-id': triplet['image-id'],
            'image': self.load_and_process_image(triplet['image']),
            'text-id': triplet['text-id'],
            'text': triplet['text'],
            'id': triplet['id']
        }

    def collate_fn(self, batch):
        query_image_ids = list(map(lambda x: x['image-id'], batch))
        query_image = {'pixel_values': torch.cat(list(map(lambda x: x['image']['pixel_values'], batch)), dim=0)}
        target_image_ids = list(map(lambda x: x['text-id'], batch))
        query_text_raw = list(map(lambda x: x['text'], batch))
        query_text = self.tokenizer(query_text_raw, padding=True, return_tensors="pt")
        ids = list(map(lambda x: x['id'], batch))
        
        return {
            'image-id': query_image_ids,
            'image': query_image,
            'text-id': target_image_ids,
            'text': query_text,
            'text-raw': query_text_raw,
            'id': ids
        }