import os
import sys
import json
import argparse
import copy
import yaml
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
import faiss
import copy
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from src.models.clip.clip import CLIPModel
from src.metrics.metrics import calculate_recall
from src.datasets.coco_captions_2014_dataset import coco_captions_dataset_image_tensors


def main(config):
    # Set devices for the model and FAISS index
    model_device = torch.device("cuda:{}".format(config['model_gpu_device_id']) if torch.cuda.is_available() else "cpu")

    print('Model device: {}'.format(model_device))

    # Load the model and map to appropriate device
    #model = CLIPModel.load_from_checkpoint(config['pl_ckpt_path'], map_location=config['model_gpu_device_id'])

    if config['eval_model_type'] == 'baseline':
        model = CLIPModel(config)
        model.image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path=config['checkpoint_path'], local_files_only=True)
        model.text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path=config['checkpoint_path'], local_files_only=True)

    model.eval()
    model.to(model_device)
    print('Model loaded')


    # Datasets & DataLoaders
    coco_2014_annotations_dataset = coco_captions_dataset_image_tensors(config, 'val')
    coco_2014_annotations_dataloader = DataLoader(
        dataset=coco_2014_annotations_dataset,
        collate_fn=coco_2014_annotations_dataset.collate_fn,
        batch_size=config['dataloader']['batch_size'],
        shuffle=config['dataloader']['shuffle'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        drop_last=config['dataloader']['drop_last'],
        persistent_workers=config['dataloader']['persistent_workers']
    )
    
    dotps_accumulated = []
    
    for batch_idx, batch in tqdm(enumerate(coco_2014_annotations_dataloader)):
        with torch.no_grad():
            batch['image']['pixel_values'] = batch['image']['pixel_values'].to(model_device)
            batch['caption']['input_ids'] = batch['caption']['input_ids'].to(model_device)
            batch['caption']['attention_mask'] = batch['caption']['attention_mask'].to(model_device)
            outs = model.coco_2014_caps_forward(batch)
        
        dotps = torch.nn.functional.cosine_similarity(outs['image_embeds'], outs['caption_embeds'], dim=1)
        
        dotps_accumulated.extend(dotps.cpu().tolist())
        break
        
    table = [
        ['Dot product mean', sum(dotps_accumulated)/len(dotps_accumulated)]
    ]

    formatted_table = tabulate(table, headers=["Metric", "Value"], floatfmt=".8f")
    print(formatted_table)

    with open(config['results_file_name'], "a") as f:
        f.write(formatted_table)
        f.write("\n\n\n\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        f.write("\n\n\n\n\n#########################################################################\n\n\n\n\n")

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='configuration file path for the task ', dest='config_file')
    args = parser.parse_args()
    config = copy.deepcopy(yaml.safe_load(open(args.config_file, 'r')))

    main(config)