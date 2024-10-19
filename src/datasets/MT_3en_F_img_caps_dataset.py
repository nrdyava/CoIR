import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from transformers import AutoProcessor, AutoTokenizer


class dataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset_split):
        self.config = config
        self.lasco_data_path = config['data']['lasco']['dir']

        if dataset_split == 'train':
            lasco_json = json.load(open(os.path.join(self.lasco_data_path, 'metadata', 'lasco_train_indexed_MT(img_cap).json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['coir']['query-image-id'],
                        'query-image': x['coir']['query-image-path'],
                        'target-image-id': x['coir']['target-image-id'],
                        'target-image': x['coir']['target-image-path'],
                        'query-text': x['coir']['query-text'],
                        'align-image-id': x['img_cap']['image-id'],
                        'align-image': x['img_cap']['image-path'],
                        'align-text': x['img_cap']['caption']
                    },
                    lasco_json
                )
            )

        elif dataset_split == 'val':
            lasco_json = json.load(open(os.path.join(self.lasco_data_path, 'metadata', 'lasco_val_indexed_MT(img_cap).json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['coir']['query-image-id'],
                        'query-image': x['coir']['query-image-path'],
                        'target-image-id': x['coir']['target-image-id'],
                        'target-image': x['coir']['target-image-path'],
                        'query-text': x['coir']['query-text'],
                        'align-image-id': x['img_cap']['image-id'],
                        'align-image': x['img_cap']['image-path'],
                        'align-text': x['img_cap']['caption']
                    },
                    lasco_json
                )
            )
            
        else:
            lasco_splits = {'train': 1, 'val': 2}
            print(lasco_splits[self.config['dataset_split']])

        # Image processor & tokenizer
        self.image_processor = AutoProcessor.from_pretrained(config['clip_checkpoints'][config['model_type']])
        self.tokenizer = AutoTokenizer.from_pretrained(config['clip_checkpoints'][config['model_type']])
        self.dataset_split = dataset_split
        self.data_aug_enabled = config['data_augmentation']['enable']
        self.data_aug_transform = transforms.RandomHorizontalFlip(p=0.5)

    def load_and_process_image(self, img_path):
        image = Image.open(img_path)
        processed_image = self.image_processor(images=image, return_tensors="pt")
        
        if self.dataset_split == 'train' and self.data_aug_enabled == True:
            processed_image['pixel_values'] = self.data_aug_transform(processed_image['pixel_values'])
        
        return processed_image

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        return {
            'query-image-id': triplet['query-image-id'],
            'query-image': self.load_and_process_image(os.path.join(self.lasco_data_path, triplet['query-image'])),
            'target-image-id': triplet['target-image-id'],
            'target-image': self.load_and_process_image(os.path.join(self.lasco_data_path, triplet['target-image'])),
            'query-text': triplet['query-text'],
            'align-image-id': triplet['align-image-id'],
            'align-image': self.load_and_process_image(os.path.join(self.lasco_data_path, triplet['align-image'])),
            'align-text': triplet['align-text']
            }

    def collate_fn(self, batch):
        query_text = list(map(lambda x: x['query-text'], batch))
        query_text = self.tokenizer(query_text, padding=True, return_tensors="pt")
        
        align_text = list(map(lambda x: x['align-text'], batch))
        align_text = self.tokenizer(align_text, padding=True, return_tensors="pt")

        query_image_ids = list(map(lambda x: x['query-image-id'], batch))
        target_image_ids = list(map(lambda x: x['target-image-id'], batch))
        align_image_ids = list(map(lambda x: x['align-image-id'], batch))
        query_image = {'pixel_values': torch.cat(list(map(lambda x: x['query-image']['pixel_values'], batch)), dim=0)}
        target_image = {'pixel_values': torch.cat(list(map(lambda x: x['target-image']['pixel_values'], batch)), dim=0)}
        align_image = {'pixel_values': torch.cat(list(map(lambda x: x['align-image']['pixel_values'], batch)), dim=0)}

        return {
            'query-image-id': query_image_ids,
            'query-image': query_image,
            'target-image-id': target_image_ids,
            'target-image': target_image,
            'query-text': query_text,
            'align-image-id': align_image_ids,
            'align-image': align_image,
            'align-text': align_text
        }