import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from transformers import AutoProcessor, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor



class lasco_dataset_train(Dataset):
    def __init__(self, config):
        self.config = config
        #self.read_n_process_threads = config["read_n_process_threads"]
        self.lasco_data_path = config['data']['lasco']['dir']

        # Image processor & tokenizer
        self.image_processor = AutoProcessor.from_pretrained(config["checkpoint_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["checkpoint_path"])
        
        lasco_json = json.load(open(os.path.join(self.lasco_data_path, 'lasco_train.json'), 'r'))
        self.triplets = list(map(lambda x: {'query-image': x['query-image'][1], 'target-image': x['target-image'][1], 'query-text': x['query-text']}, lasco_json))

        """
        all_query_images = list(map(lambda x: x['query-image'][1], lasco_json))
        all_target_images = list(map(lambda x: x['target-image'][1], lasco_json))
        all_images = list(set(all_query_images+all_target_images))

        all_images_paths_n_names = list(map(lambda x: (os.path.join(self.lasco_data_path, 'coco', x), x), all_images))
        with ThreadPoolExecutor(max_workers=self.read_n_process_threads) as executor:
            results = executor.map(self.load_and_process_image, all_images_paths_n_names)
            self.image_pixvals_map = dict(results)
        """

    def load_and_process_image(self, paths_n_names):
        image = Image.open(paths_n_names[0])
        processed_image = self.image_processor(images=image, return_tensors="pt")
        return (paths_n_names[1], processed_image)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        #return {'query-image': self.image_pixvals_map[triplet['query-image']], 'target-image': self.image_pixvals_map[triplet['target-image']], 'query-text': triplet['query-text']}
        return {
            'query-image': self.load_and_process_image((os.path.join(self.lasco_data_path, 'coco', triplet['query-image']), triplet['query-image']))[1], 
            'target-image': self.load_and_process_image((os.path.join(self.lasco_data_path, 'coco', triplet['target-image']), triplet['target-image']))[1], 
            'query-text': triplet['query-text']
            }
    
    def collate_fn(self, batch):
        query_text = list(map(lambda x: x['query-text'], batch))
        query_text = self.tokenizer(query_text, padding=True, return_tensors="pt")
        
        query_image = {'pixel_values': torch.cat(list(map(lambda x: x['query-image']['pixel_values'], batch)), dim=0)}
        target_image = {'pixel_values': torch.cat(list(map(lambda x: x['target-image']['pixel_values'], batch)), dim=0)}

        return {'query-image': query_image, 'target-image': target_image, 'query-text': query_text}




class lasco_dataset_val(Dataset):
    def __init__(self, config):
        self.config = config
        #self.read_n_process_threads = config["read_n_process_threads"]
        self.lasco_data_path = config['data']['lasco']['dir']

        # Image processor & tokenizer
        self.image_processor = AutoProcessor.from_pretrained(config["checkpoint_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["checkpoint_path"])
        
        lasco_json = json.load(open(os.path.join(self.lasco_data_path, 'lasco_val.json'), 'r'))
        self.triplets = list(map(lambda x: {'query-image': x['query-image'][1], 'target-image': x['target-image'][1], 'query-text': x['query-text']}, lasco_json))

        """
        all_query_images = list(map(lambda x: x['query-image'][1], lasco_json))
        all_target_images = list(map(lambda x: x['target-image'][1], lasco_json))
        all_images = list(set(all_query_images+all_target_images))

        all_images_paths_n_names = list(map(lambda x: (os.path.join(self.lasco_data_path, 'coco', x), x), all_images))
        with ThreadPoolExecutor(max_workers=self.read_n_process_threads) as executor:
            results = executor.map(self.load_and_process_image, all_images_paths_n_names)
            self.image_pixvals_map = dict(results)
        """


    def load_and_process_image(self, paths_n_names):
        image = Image.open(paths_n_names[0])
        processed_image = self.image_processor(images=image, return_tensors="pt")
        return (paths_n_names[1], processed_image)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        #return {'query-image': self.image_pixvals_map[triplet['query-image']], 'target-image': self.image_pixvals_map[triplet['target-image']], 'query-text': triplet['query-text']}
        return {
            'query-image': self.load_and_process_image((os.path.join(self.lasco_data_path, 'coco', triplet['query-image']), triplet['query-image']))[1], 
            'target-image': self.load_and_process_image((os.path.join(self.lasco_data_path, 'coco', triplet['target-image']), triplet['target-image']))[1], 
            'query-text': triplet['query-text']
            }
    
    def collate_fn(self, batch):
        query_text = list(map(lambda x: x['query-text'], batch))
        query_text = self.tokenizer(query_text, padding=True, return_tensors="pt")
        
        query_image = {'pixel_values': torch.cat(list(map(lambda x: x['query-image']['pixel_values'], batch)), dim=0)}
        target_image = {'pixel_values': torch.cat(list(map(lambda x: x['target-image']['pixel_values'], batch)), dim=0)}

        return {'query-image': query_image, 'target-image': target_image, 'query-text': query_text}