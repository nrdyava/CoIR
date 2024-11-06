#!/usr/bin/env python
# coding: utf-8

device = 3
exp_name = '2024-11-02-12-48-16-027523 + clip_inbatch_2en_MT_FR1_ON_GT_QN_LR_1e-5'
batch_size=64

out_dir = '/proj/vondrick4/naveen/coir-ret-results'
runs_dir = '/proj/vondrick4/naveen/coir-runs'
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
import pandas as pd

from src.datasets.cirr.cirr_corpus_dataset import cirr_corpus_dataset_clip
from src.datasets.cirr.cirr_retrieval_dataset import cirr_retrieval_dataset_clip
from src.metrics.metrics import calculate_recall, calculate_APs
from src.models.clip.inbatch_2en_MT_FR1_ON_GT_QN import CLIPModel




exp_path = os.path.join(runs_dir, exp_name)
checkpoints = sorted(list(filter(lambda x: x[:10] == 'checkpoint', os.listdir(exp_path))))
os.makedirs(os.path.join(out_dir, exp_name), exist_ok=True)


print('Evaluating Experiment: {}'.format(exp_name))

exp_results = pd.DataFrame(columns=['checkpoint', 'epoch', 'Recall@1', 'Recall@5', 'Recall@10', 'Recall@50', 'Recall@100', 'Recall@500', 'Recall@1000', 'Recall_Subset@1', 'Recall_Subset@2', 'Recall_Subset@3', 'Recall_Subset@4', 'Recall_Subset@5', 'Recall_Subset@6'])
exp_results.to_csv(os.path.join(out_dir, exp_name, 'CIRR-experiment_results.csv'), index=False)


for checkpoint in checkpoints:
    checkpoint_path = os.path.join(runs_dir, exp_name, checkpoint)

    print('Evaluating checkpoint: {}'.format(checkpoint))
    model = CLIPModel.load_from_checkpoint(
        checkpoint_path = checkpoint_path, 
        map_location=device_map,
        strict=False
    )
    model.eval()
    print('Model loaded')

    clip_checkpoint_path = clip_checkpoints[model.config['model_type']]
    print('Using device: {}'.format(device_map))

    d = model.image_encoder.config.projection_dim
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
        batch['image']['pixel_values'] = batch['image']['pixel_values'].to(device_map)
        with torch.no_grad():
            image_embeds = model.img_forward(batch)
        index.add(image_embeds['image-embeds'].cpu())

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
        
            target_hat_embeds = model.img_txt_forward(batch)

        target_hat_embeds_numpy = target_hat_embeds['target-hat-embeds'].cpu().numpy().reshape(-1)
        subset_preds = list(map(lambda x: x[0], sorted(list(map(lambda x: (x, np.dot(index.reconstruct(image_id_to_index_id_map[x]), target_hat_embeds_numpy)), batch['subset'][0])), key=lambda x: x[1], reverse=True)))
        
        D, I = index.search(target_hat_embeds['target-hat-embeds'].cpu(), k=1000)
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

    with open(os.path.join(out_dir, exp_name, f'CIRR_outputs-'+checkpoint+'.json'), "w") as json_file:
        json.dump(output, json_file, indent=4)
    with open(os.path.join(out_dir, exp_name, f'CIRR_outputs_[light]-'+checkpoint+'.json'), "w") as json_file:
        json.dump(output_light, json_file, indent=4)
    
    
    metrics = []
    ground_truths = np.array(list(map(lambda x: x['target-image-id'], output)))
    retrieved_candidates = np.array(list(map(lambda x: x['top_1000_ret_cands'], output)))
    retrieved_candidates_subset = np.array(list(map(lambda x: x['subset_preds'], output)))
    
    
    
    R1 = 100*calculate_recall(ground_truths, retrieved_candidates, 1)
    R5 = 100*calculate_recall(ground_truths, retrieved_candidates, 5)
    R10 = 100*calculate_recall(ground_truths, retrieved_candidates, 10)
    R50 = 100*calculate_recall(ground_truths, retrieved_candidates, 50)
    R100 = 100*calculate_recall(ground_truths, retrieved_candidates, 100)
    R500 = 100*calculate_recall(ground_truths, retrieved_candidates, 500)
    R1000 = 100*calculate_recall(ground_truths, retrieved_candidates, 1000)
    
    Rs1 = 100*calculate_recall(ground_truths, retrieved_candidates_subset, 1)
    Rs2 = 100*calculate_recall(ground_truths, retrieved_candidates_subset, 2)
    Rs3 = 100*calculate_recall(ground_truths, retrieved_candidates_subset, 3)
    Rs4 = 100*calculate_recall(ground_truths, retrieved_candidates_subset, 4)
    Rs5 = 100*calculate_recall(ground_truths, retrieved_candidates_subset, 5)
    Rs6 = 100*calculate_recall(ground_truths, retrieved_candidates_subset, 6)
    
    
    metrics.append({"Recall@1": R1})
    metrics.append({"Recall@5": R5})
    metrics.append({"Recall@10": R10})
    metrics.append({"Recall@50": R50})
    metrics.append({"Recall@100": R100})
    metrics.append({"Recall@500": R500})
    metrics.append({"Recall@1000": R1000})

    metrics.append({"Recall_Subset@1": Rs1})
    metrics.append({"Recall_Subset@2": Rs2})
    metrics.append({"Recall_Subset@3": Rs3})
    metrics.append({"Recall_Subset@4": Rs4})
    metrics.append({"Recall_Subset@5": Rs5})
    metrics.append({"Recall_Subset@6": Rs6})
    
    with open(os.path.join(out_dir, exp_name, f'CIRR_metrics-'+checkpoint+'.json'), "w") as json_file:
        json.dump(metrics, json_file, indent=4)
        
    
    epoch = int(checkpoint[17:20])
    new_row = pd.DataFrame([{
        'checkpoint': checkpoint, 
        'epoch': epoch, 
        'Recall@1': R1,
        'Recall@5': R5,
        'Recall@10': R10,
        'Recall@50': R50,
        'Recall@100': R100,
        'Recall@500': R500,
        'Recall@1000': R1000,
        'Recall_Subset@1': Rs1,
        'Recall_Subset@2': Rs2,
        'Recall_Subset@3': Rs3,
        'Recall_Subset@4': Rs4,
        'Recall_Subset@5': Rs5,
        'Recall_Subset@6': Rs6
    }])
    exp_results = pd.concat([exp_results, new_row], ignore_index=True)
    exp_results.to_csv(os.path.join(out_dir, exp_name, 'CIRR-experiment_results.csv'), index=False)