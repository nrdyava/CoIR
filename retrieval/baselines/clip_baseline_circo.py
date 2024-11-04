#!/usr/bin/env python
# coding: utf-8

device = 3
results_name = 'CLIP-ViT-L-14-336'
clip_model = 'CLIP-ViT-L/14@336' # Options: ['CLIP-ViT-B/16', 'CLIP-ViT-B/32', 'CLIP-ViT-L/14', 'CLIP-ViT-L/14@336']
batch_size=64


out_dir = '/proj/vondrick4/naveen/coir-ret-results'
dataset_split = 'test'
lasco_data_path = '/local/vondrick/naveen/coir-data/CIRCO'
device_map = 'cuda:{}'.format(device)


clip_checkpoints = {
    'CLIP-ViT-B/16': '/local/vondrick/naveen/pretrained_models/clip/clip-vit-base-patch16',
    'CLIP-ViT-B/32': '/local/vondrick/naveen/pretrained_models/clip/clip-vit-base-patch32',
    'CLIP-ViT-L/14': '/local/vondrick/naveen/pretrained_models/clip/clip-vit-large-patch14',
    'CLIP-ViT-L/14@336': '/local/vondrick/naveen/pretrained_models/clip/clip-vit-large-patch14-336'
}


import sys
sys.path.append('/proj/vondrick4/naveen/CoIR')
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import faiss
import torch
import numpy as np
import os
import json

from src.datasets.circo.circo_corpus_dataset import circo_corpus_dataset_clip
from src.datasets.circo.circo_retrieval_dataset import circo_retrieval_dataset_clip
from src.metrics.metrics import calculate_recall, calculate_APs


os.makedirs(os.path.join(out_dir, results_name), exist_ok=True)
clip_checkpoint_path = clip_checkpoints[clip_model]
print('Using device: {}'.format(device_map))



image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path=clip_checkpoint_path, local_files_only=True).to(device)
text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path=clip_checkpoint_path, local_files_only=True).to(device)

image_encoder.eval()
text_encoder.eval()
print('Model loaded')



d = image_encoder.config.projection_dim
index = faiss.IndexFlatIP(d)

res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, device, index)



corpus_dataset = circo_corpus_dataset_clip(dataset_split, lasco_data_path, clip_checkpoint_path)
corpus_dataloader = DataLoader(
    dataset=corpus_dataset,
    collate_fn=corpus_dataset.collate_fn,
    batch_size=batch_size,
    shuffle=False,
    num_workers=10,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True
)


retrieval_dataset = circo_retrieval_dataset_clip(dataset_split, lasco_data_path, clip_checkpoint_path)
retrieval_dataloader = DataLoader(
    dataset=retrieval_dataset,
    collate_fn=retrieval_dataset.collate_fn,
    batch_size=batch_size,
    shuffle=False,
    num_workers=10,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True
)



index_cntr = 0
index_id_to_image_id_map = {}



for batch_idx, batch in enumerate(tqdm(corpus_dataloader, desc="Indexing Corpus")):
    with torch.no_grad():
        batch['image']['pixel_values'] = batch['image']['pixel_values'].to(device_map)
        image_embeds = image_encoder(**batch['image']).image_embeds
        image_embeds = image_embeds / torch.linalg.vector_norm(image_embeds, ord=2, dim=1,keepdim=True)
    
    index.add(image_embeds.cpu())

    batch_len = len(batch['image-key'])
    batch_start_indx = index_cntr
    batch_end_indx = batch_start_indx + batch_len

    for key, value in zip(list(range(batch_start_indx, batch_end_indx)), batch['image-key']):
        index_id_to_image_id_map[key] = value
    index_cntr += batch_len
    
    

map_func = np.vectorize(lambda x: index_id_to_image_id_map[x])


output = []
output_light = []


for batch_idx, batch in enumerate(tqdm(retrieval_dataloader , desc="Retrieval Task")):
    with torch.no_grad():
        batch['query-image']['pixel_values'] = batch['query-image']['pixel_values'].to(device_map)
        batch['query-text']['input_ids'] = batch['query-text']['input_ids'].to(device_map)
        batch['query-text']['attention_mask'] = batch['query-text']['attention_mask'].to(device_map)
        
        query_image_embeds = image_encoder(**batch['query-image']).image_embeds
        query_text_embeds = text_encoder(**batch['query-text']).text_embeds
        
    target_hat_embeds = query_image_embeds + query_text_embeds
    target_hat_embeds = target_hat_embeds / torch.linalg.vector_norm(target_hat_embeds, ord=2, dim=1, keepdim=True)

    D, I = index.search(target_hat_embeds.cpu(), k=1000)
    I = map_func(I)

    batch_size = len(batch['query-image-id'])
    for i in range(batch_size):
        output.append({
            'id': batch['id'][i],
            'query-image-id': batch['query-image-id'][i],
            'target-image-id': batch['target-image-id'][i],
            'query-text-raw': batch['query-text-raw'][i],
            'gt-image-ids': batch['gt-image-ids'][i],
            'top_1000_ret_cands': I[i][:].tolist(),
            'top_1000_ret_cands_cos_sims': D[i][:].tolist()
            })
        
        output_light.append({
            'id': batch['id'][i],
            'query-image-id': batch['query-image-id'][i],
            'target-image-id': batch['target-image-id'][i],
            'query-text-raw': batch['query-text-raw'][i],
            'gt-image-ids': batch['gt-image-ids'][i],
            'top_50_ret_cands': I[i][:].tolist()[:50],
            'top_50_ret_cands_cos_sims': D[i][:].tolist()[:50]
            })
        
    
    with open(os.path.join(out_dir, results_name, f'CIRCO_outputs'+'.json'), "w") as json_file:
        json.dump(output, json_file, indent=4)
    with open(os.path.join(out_dir, results_name, f'CIRCO_outputs_[light]'+'.json'), "w") as json_file:
        json.dump(output_light, json_file, indent=4)
        
    ranks = [1, 5, 10, 25, 50, 100, 500, 1000]
    metrics_calculated = calculate_APs(output, ranks)
    metrics = []
    
    for rank in ranks:
        metrics.append({"mAP@{}".format(rank): 100*metrics_calculated[rank]})
        
    with open(os.path.join(out_dir, results_name, f'CIRCO_metrics'+'.json'), "w") as json_file:
            json.dump(metrics, json_file, indent=4)
    