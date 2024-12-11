#!/usr/bin/env python
# coding: utf-8

device = 2
exp_name = '2024-12-10-19-07-46-975795 + clip_inbatch_3en_MT_AR1CDM_SO_LR_1e-5'
batch_size=64

out_dir = '/proj/vondrick4/naveen/coir-ret-results'
runs_dir = '/proj/vondrick4/naveen/coir-runs'
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
import pandas as pd

from src.datasets.circo.circo_corpus_dataset import circo_corpus_dataset_clip
from src.datasets.circo.circo_retrieval_dataset import circo_retrieval_dataset_clip
from src.metrics.metrics import calculate_recall, calculate_APs
from src.models.clip.inabatch_3en_MT_AR1CDM_SO import CLIPModel




exp_path = os.path.join(runs_dir, exp_name)
checkpoints = sorted(list(filter(lambda x: x[:10] == 'checkpoint', os.listdir(exp_path))))
os.makedirs(os.path.join(out_dir, exp_name), exist_ok=True)


print('Evaluating Experiment: {}'.format(exp_name))

exp_results = pd.DataFrame(columns=['checkpoint', 'epoch', 'mAP@1', 'mAP@5', 'mAP@10', 'mAP@25', 'mAP@50', 'mAP@100', 'mAP@500', 'mAP@1000'])
exp_results.to_csv(os.path.join(out_dir, exp_name, 'CIRCO-experiment_results.csv'), index=False)


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
        batch['image']['pixel_values'] = batch['image']['pixel_values'].to(device_map)
        with torch.no_grad():
            image_embeds = model.img_forward(batch)
        index.add(image_embeds['image-embeds'].cpu())

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
        
            target_hat_embeds = model.img_txt_forward(batch)

        D, I = index.search(target_hat_embeds['target-hat-embeds'].cpu(), k=1000)
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

    with open(os.path.join(out_dir, exp_name, f'CIRCO_outputs-'+checkpoint+'.json'), "w") as json_file:
        json.dump(output, json_file, indent=4)
    with open(os.path.join(out_dir, exp_name, f'CIRCO_outputs_[light]-'+checkpoint+'.json'), "w") as json_file:
        json.dump(output_light, json_file, indent=4)
        
    ranks = [1, 5, 10, 25, 50, 100, 500, 1000]
    metrics_calculated = calculate_APs(output, ranks)
    metrics = []
    
    for rank in ranks:
        metrics.append({"mAP@{}".format(rank): 100*metrics_calculated[rank]})
        
    with open(os.path.join(out_dir, exp_name, f'CIRCO_metrics-'+checkpoint+'.json'), "w") as json_file:
        json.dump(metrics, json_file, indent=4)
        
    
    epoch = int(checkpoint[17:20])
    new_row = pd.DataFrame([{
        'checkpoint': checkpoint, 
        'epoch': epoch, 
        'mAP@1': 100*metrics_calculated[1],
        'mAP@5': 100*metrics_calculated[5],
        'mAP@10': 100*metrics_calculated[10],
        'mAP@25': 100*metrics_calculated[25],
        'mAP@50': 100*metrics_calculated[50],
        'mAP@100': 100*metrics_calculated[100],
        'mAP@500': 100*metrics_calculated[500],
        'mAP@1000': 100*metrics_calculated[1000]
    }])
    exp_results = pd.concat([exp_results, new_row], ignore_index=True)
    exp_results.to_csv(os.path.join(out_dir, exp_name, 'CIRCO-experiment_results.csv'), index=False)
