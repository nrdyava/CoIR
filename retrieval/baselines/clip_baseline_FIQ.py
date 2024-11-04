#!/usr/bin/env python
# coding: utf-8

device = 7
results_name = 'CLIP-ViT-L-14-336'
clip_model = 'CLIP-ViT-L/14@336' # Options: ['CLIP-ViT-B/16', 'CLIP-ViT-B/32', 'CLIP-ViT-L/14', 'CLIP-ViT-L/14@336']
batch_size=64

out_dir = '/proj/vondrick4/naveen/coir-ret-results'
dataset_split = 'val'
lasco_data_path = '/local/vondrick/naveen/coir-data/FashionIQ'
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

from src.datasets.fiq.fiq_corpus_dataset import fiq_corpus_dataset_clip
from src.datasets.fiq.fiq_retrieval_dataset import fiq_retrieval_dataset_clip
from src.metrics.metrics import calculate_recall



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




corpus_dataset = fiq_corpus_dataset_clip('val', lasco_data_path, clip_checkpoint_path)
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


for split in ['dress', 'shirt', 'toptee']:
    retrieval_dataset = fiq_retrieval_dataset_clip(split, lasco_data_path, clip_checkpoint_path)
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
    
    output = []
    output_light = []


    for batch_idx, batch in enumerate(tqdm(retrieval_dataloader , desc=f"Retrieval Task_{split}")):
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
                'query-image-id': batch['query-image-id'][i],
                'target-image-id': batch['target-image-id'][i],
                'query-text-raw': batch['query-text-raw'][i],
                'top_1000_ret_cands': I[i][:].tolist(),
                'top_1000_ret_cands_cos_sims': D[i][:].tolist()
                })

            output_light.append({
                'query-image-id': batch['query-image-id'][i],
                'target-image-id': batch['target-image-id'][i],
                'query-text-raw': batch['query-text-raw'][i],
                'top_50_ret_cands': I[i][:].tolist()[:50],
                'top_50_ret_cands_cos_sims': D[i][:].tolist()[:50]
                })
            
    
    with open(os.path.join(out_dir, results_name, f'FIQ_outputs_{split}'+'.json'), "w") as json_file:
        json.dump(output, json_file, indent=4)
    with open(os.path.join(out_dir, results_name, f'FIQ_outputs_{split}_[light]'+'.json'), "w") as json_file:
        json.dump(output_light, json_file, indent=4)
        
    
    metrics = []
    ground_truths = np.array(list(map(lambda x: x['target-image-id'], output)))
    retrieved_candidates = np.array(list(map(lambda x: x['top_1000_ret_cands'], output)))
    
    Recall_1 = 100*calculate_recall(ground_truths, retrieved_candidates, 1)
    Recall_5 = 100*calculate_recall(ground_truths, retrieved_candidates, 5)
    Recall_10 = 100*calculate_recall(ground_truths, retrieved_candidates, 10)
    Recall_50 = 100*calculate_recall(ground_truths, retrieved_candidates, 50)
    Recall_100 = 100*calculate_recall(ground_truths, retrieved_candidates, 100)
    Recall_500 = 100*calculate_recall(ground_truths, retrieved_candidates, 500)
    Recall_1000 = 100*calculate_recall(ground_truths, retrieved_candidates, 1000)
    metrics.append({"Recall@1": Recall_1})
    metrics.append({"Recall@5": Recall_5})
    metrics.append({"Recall@10": Recall_10})
    metrics.append({"Recall@50": Recall_50})
    metrics.append({"Recall@100": Recall_100})
    metrics.append({"Recall@500": Recall_500})
    metrics.append({"Recall@1000": Recall_1000})
    
    
    with open(os.path.join(out_dir, results_name, f'FIQ_metrics_{split}'+'.json'), "w") as json_file:
            json.dump(metrics, json_file, indent=4)