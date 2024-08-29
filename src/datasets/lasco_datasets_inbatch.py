import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from transformers import AutoProcessor, AutoTokenizer


class lasco_dataset_inbatch(Dataset):
    def __init__(self, config, dataset_split):
        self.config = config
        self.lasco_data_path = config['data']['lasco']['dir']

        if dataset_split == 'train':
            lasco_json = json.load(open(os.path.join(self.lasco_data_path, 'lasco_train.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['query-image'][0],
                        'query-image': x['query-image'][1],
                        'target-image-id': x['target-image'][0],
                        'target-image': x['target-image'][1],
                        'query-text': x['query-text']
                    },
                    lasco_json
                )
            )

        elif dataset_split == 'val':
            lasco_json = json.load(open(os.path.join(self.lasco_data_path, 'lasco_val.json'), 'r'))
            self.triplets = list(
                map(
                    lambda x: {
                        'query-image-id': x['query-image'][0],
                        'query-image': x['query-image'][1],
                        'target-image-id': x['target-image'][0],
                        'target-image': x['target-image'][1],
                        'query-text': x['query-text']
                    },
                    lasco_json
                )
            )
        else:
            lasco_splits = {'train': 1, 'val': 2}
            print(lasco_splits[self.config['dataset_split']])

        # Image processor & tokenizer
        self.image_processor = AutoProcessor.from_pretrained(config["checkpoint_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["checkpoint_path"])

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
            'query-image': self.load_and_process_image(os.path.join(self.lasco_data_path, 'coco', triplet['query-image'])),
            'target-image-id': triplet['target-image-id'],
            'target-image': self.load_and_process_image(os.path.join(self.lasco_data_path, 'coco', triplet['target-image'])),
            'query-text': triplet['query-text']
        }

    def collate_fn(self, batch):
        query_text = list(map(lambda x: x['query-text'], batch))
        query_text = self.tokenizer(query_text, padding=True, return_tensors="pt")

        query_image_ids = list(map(lambda x: x['query-image-id'], batch))
        target_image_ids = list(map(lambda x: x['target-image-id'], batch))
        query_image = {'pixel_values': torch.cat(list(map(lambda x: x['query-image']['pixel_values'], batch)), dim=0)}
        target_image = {'pixel_values': torch.cat(list(map(lambda x: x['target-image']['pixel_values'], batch)), dim=0)}

        return {
            'query-image-id': query_image_ids,
            'query-image': query_image,
            'target-image-id': target_image_ids,
            'target-image': target_image,
            'query-text': query_text
        }
