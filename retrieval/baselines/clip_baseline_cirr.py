#!/usr/bin/env python
# coding: utf-8

device = 3
results_name = 'CLIP-ViT-L-14-336'
clip_model = 'CLIP-ViT-L/14@336' # Options: ['CLIP-ViT-B/16', 'CLIP-ViT-B/32', 'CLIP-ViT-L/14', 'CLIP-ViT-L/14@336']
batch_size=64

out_dir = '/proj/vondrick4/naveen/coir-ret-results'
dataset_split = 'val'
lasco_data_path = '/local/vondrick/naveen/coir-data/CIRR'
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

from src.datasets.cirr.cirr_corpus_dataset import cirr_corpus_dataset_clip
from src.datasets.cirr.cirr_retrieval_dataset import cirr_retrieval_dataset_clip
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




corpus_dataset = cirr_corpus_dataset_clip(dataset_split, lasco_data_path, clip_checkpoint_path)
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



retrieval_dataset = cirr_retrieval_dataset_clip(dataset_split, lasco_data_path, clip_checkpoint_path)
retrieval_dataloader = DataLoader(
    dataset=retrieval_dataset,
    collate_fn=retrieval_dataset.collate_fn,
    batch_size=1,
    shuffle=False,
    num_workers=10,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True
)




index_cntr = 0
index_id_to_image_id_map = {}
image_id_to_index_id_map = {}




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
        image_id_to_index_id_map[value] = key
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

    target_hat_embeds_numpy = target_hat_embeds.cpu().numpy().reshape(-1)
    subset_preds = list(map(lambda x: x[0], sorted(list(map(lambda x: (x, np.dot(index.reconstruct(image_id_to_index_id_map[x]), target_hat_embeds_numpy)), batch['subset'][0])), key=lambda x: x[1], reverse=True)))
    
    D, I = index.search(target_hat_embeds.cpu(), k=1000)
    I = map_func(I)

    batch_size = len(batch['query-image-id'])
    for i in range(batch_size):
        output.append({
            'id': batch['id'][i],
            'query-image-id': batch['query-image-id'][i],
            'target-image-id': batch['target-image-id'][i],
            'query-text-raw': batch['query-text-raw'][i],
            'subset_preds': subset_preds,
            'top_1000_ret_cands': I[i][:].tolist(),
            'top_1000_ret_cands_cos_sims': D[i][:].tolist()
            })
        
        output_light.append({
            'id': batch['id'][i],
            'query-image-id': batch['query-image-id'][i],
            'target-image-id': batch['target-image-id'][i],
            'query-text-raw': batch['query-text-raw'][i],
            'subset_preds': subset_preds,
            'top_50_ret_cands': I[i][:].tolist()[:50],
            'top_50_ret_cands_cos_sims': D[i][:].tolist()[:50]
            })



with open(os.path.join(out_dir, results_name, f'CIRR_outputs'+'.json'), "w") as json_file:
    json.dump(output, json_file, indent=4)
with open(os.path.join(out_dir, results_name, f'CIRR_outputs_[light]'+'.json'), "w") as json_file:
    json.dump(output_light, json_file, indent=4)



metrics = []
ground_truths = np.array(list(map(lambda x: x['target-image-id'], output)))
retrieved_candidates = np.array(list(map(lambda x: x['top_1000_ret_cands'], output)))
retrieved_candidates_subset = np.array(list(map(lambda x: x['subset_preds'], output)))




metrics.append({"Recall@1": 100*calculate_recall(ground_truths, retrieved_candidates, 1)})
metrics.append({"Recall@5": 100*calculate_recall(ground_truths, retrieved_candidates, 5)})
metrics.append({"Recall@10": 100*calculate_recall(ground_truths, retrieved_candidates, 10)})
metrics.append({"Recall@50": 100*calculate_recall(ground_truths, retrieved_candidates, 50)})
metrics.append({"Recall@100": 100*calculate_recall(ground_truths, retrieved_candidates, 100)})
metrics.append({"Recall@500": 100*calculate_recall(ground_truths, retrieved_candidates, 500)})
metrics.append({"Recall@1000": 100*calculate_recall(ground_truths, retrieved_candidates, 1000)})




metrics.append({"Recall_Subset@1": 100*calculate_recall(ground_truths, retrieved_candidates_subset, 1)})
metrics.append({"Recall_Subset@2": 100*calculate_recall(ground_truths, retrieved_candidates_subset, 2)})
metrics.append({"Recall_Subset@3": 100*calculate_recall(ground_truths, retrieved_candidates_subset, 3)})
metrics.append({"Recall_Subset@4": 100*calculate_recall(ground_truths, retrieved_candidates_subset, 4)})
metrics.append({"Recall_Subset@5": 100*calculate_recall(ground_truths, retrieved_candidates_subset, 5)})
metrics.append({"Recall_Subset@6": 100*calculate_recall(ground_truths, retrieved_candidates_subset, 6)})


with open(os.path.join(out_dir, results_name, f'CIRR_metrics'+'.json'), "w") as json_file:
    json.dump(metrics, json_file, indent=4)

