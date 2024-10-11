import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from transformers import AutoProcessor, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

class lasco_corpus_dataset(Dataset):
    def __init__(self, retriever_config, checkpoint_config):
        self.retriever_config = retriever_config
        self.checkpoint_config = checkpoint_config
        self.lasco_data_path = self.retriever_config['data']['lasco']['dir']

        if self.retriever_config['dataset_split'] == 'train':
            corpus = json.load(open(os.path.join(self.lasco_data_path, 'metadata', 'lasco_train_corpus.json'), 'r'))
            self.image_book = list(
                map(
                    lambda x: {
                        'image-key': x['image-id'], 
                        'image': os.path.join(self.lasco_data_path, x['image-path'])
                        },
                    corpus
                    )
                )
        elif self.retriever_config['dataset_split'] == 'val':
            corpus = json.load(open(os.path.join(self.lasco_data_path, 'metadata', 'lasco_val_corpus.json'), 'r'))
            self.image_book = list(
                map(
                    lambda x: {
                        'image-key': x['image-id'], 
                        'image': os.path.join(self.lasco_data_path, x['image-path'])
                        },
                    corpus
                    )
                )
        else:
            lasco_splits = {'train': 1, 'val': 2}
            print(lasco_splits[self.retriever_config['dataset_split']])

        # Image processor & tokenizer
        self.image_processor = AutoProcessor.from_pretrained(self.retriever_config['clip_checkpoints'][self.checkpoint_config['model_type']])
        self.tokenizer = AutoTokenizer.from_pretrained(self.retriever_config['clip_checkpoints'][self.checkpoint_config['model_type']])

    def load_and_process_image(self, img_path):
        image = Image.open(img_path)
        processed_image = self.image_processor(images=image, return_tensors="pt")
        return processed_image

    def __len__(self):
        return len(self.image_book)

    def __getitem__(self, idx):
        image_tuple = self.image_book[idx]
        return {
            'image-key': image_tuple[0],
            'image': self.load_and_process_image(image_tuple[1])
        }

    def collate_fn(self, batch):
        image_key = list(map(lambda x: x['image-key'], batch))
        image = {'pixel_values': torch.cat(list(map(lambda x: x['image']['pixel_values'], batch)), dim=0)}
        return {'image-key': image_key, 'image': image}