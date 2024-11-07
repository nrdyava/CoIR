#!/usr/bin/env python
# coding: utf-8


device = 4
exp_name = '2024-11-06-22-57-32-856388 + clip_inbatch_3en_MT_AF_SO_GT_QN_MG_LR_1e-5'
batch_size=64

out_dir = '/proj/vondrick4/naveen/coir-ret-results'
runs_dir = '/proj/vondrick4/naveen/coir-runs'
dataset_split = 'test'
lasco_data_path = '/proj/vondrick4/naveen/coir-data/MSCOCO_5k'
device_map = 'cuda:{}'.format(device)


# In[2]:


clip_checkpoints = {
    'CLIP-ViT-B/16': '/local/vondrick/naveen/pretrained_models/clip/clip-vit-base-patch16',
    'CLIP-ViT-B/32': '/local/vondrick/naveen/pretrained_models/clip/clip-vit-base-patch32',
    'CLIP-ViT-L/14': '/local/vondrick/naveen/pretrained_models/clip/clip-vit-large-patch14',
    'CLIP-ViT-L/14@336': '/local/vondrick/naveen/pretrained_models/clip/clip-vit-large-patch14-336'
}


# In[3]:


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

from src.datasets.mscoco_5k.mscoco_5k_corpus_dataset import mscoco_5k_image_corpus_dataset_clip, mscoco_5k_text_corpus_dataset_clip
from src.datasets.mscoco_5k.mscoco_5k_retrieval_dataset import mscoco_5k_retrieval_dataset_clip
from src.metrics.metrics import calculate_recall
from src.models.clip.inbatch_3en_MT_AF_SO_GT_QN_MG import CLIPModel




exp_path = os.path.join(runs_dir, exp_name)
checkpoints = sorted(list(filter(lambda x: x[:10] == 'checkpoint', os.listdir(exp_path))))
os.makedirs(os.path.join(out_dir, exp_name), exist_ok=True)




print('Evaluating Experiment: {}'.format(exp_name))

exp_results_i2t = pd.DataFrame(columns=['checkpoint', 'epoch', 'Recall@1', 'Recall@5', 'Recall@10', 'Recall@50', 'Recall@100', 'Recall@500', 'Recall@1000'])
exp_results_t2i = pd.DataFrame(columns=['checkpoint', 'epoch', 'Recall@1', 'Recall@5', 'Recall@10', 'Recall@50', 'Recall@100', 'Recall@500', 'Recall@1000'])
exp_results_i2t.to_csv(os.path.join(out_dir, exp_name, 'mscoco-5k-[image-2-text]-experiment_results.csv'), index=False)
exp_results_t2i.to_csv(os.path.join(out_dir, exp_name, 'mscoco-5k-[text-2-image]-experiment_results.csv'), index=False)

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
    image_index = faiss.IndexFlatIP(d)
    text_index = faiss.IndexFlatIP(d)

    res = faiss.StandardGpuResources()
    image_index = faiss.index_cpu_to_gpu(res, device, image_index)
    text_index = faiss.index_cpu_to_gpu(res, device, text_index)
    
    corpus_dataset_image = mscoco_5k_image_corpus_dataset_clip(dataset_split, lasco_data_path, clip_checkpoint_path)
    corpus_dataloader_image = DataLoader(
        dataset=corpus_dataset_image,
        collate_fn=corpus_dataset_image.collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
        )
    
    corpus_dataset_text = mscoco_5k_text_corpus_dataset_clip(dataset_split, lasco_data_path, clip_checkpoint_path)
    corpus_dataloader_text = DataLoader(
        dataset=corpus_dataset_text,
        collate_fn=corpus_dataset_text.collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
        )

    

    index_cntr = 0
    index_id_to_image_id_map = {}
    
    
    for batch_idx, batch in enumerate(tqdm(corpus_dataloader_image, desc="Indexing Corpus: IMAGES")):
        with torch.no_grad():
            batch['image']['pixel_values'] = batch['image']['pixel_values'].to(device_map)
            image_embeds = model.img_forward(batch)
            image_embeds = image_embeds['image-embeds']
            
        image_index.add(image_embeds.cpu())

        batch_len = len(batch['image-key'])
        batch_start_indx = index_cntr
        batch_end_indx = batch_start_indx + batch_len

        for key, value in zip(list(range(batch_start_indx, batch_end_indx)), batch['image-key']):
            index_id_to_image_id_map[key] = value
        index_cntr += batch_len
    
    
    
    index_cntr = 0
    index_id_to_text_id_map = {}
    
    for batch_idx, batch in enumerate(tqdm(corpus_dataloader_text, desc="Indexing Corpus: TEXT")):
        with torch.no_grad():
            batch['text']['input_ids'] = batch['text']['input_ids'].to(device_map)
            batch['text']['attention_mask'] = batch['text']['attention_mask'].to(device_map)
            text_embeds = model.text_forward(batch)
            text_embeds = text_embeds['text-embeds']
    
        text_index.add(text_embeds.cpu())

        batch_len = len(batch['text-key'])
        batch_start_indx = index_cntr
        batch_end_indx = batch_start_indx + batch_len

        for key, value in zip(list(range(batch_start_indx, batch_end_indx)), batch['text-key']):
            index_id_to_text_id_map[key] = value
        index_cntr += batch_len
    
    
    
    map_func_image = np.vectorize(lambda x: index_id_to_image_id_map[x])
    map_func_text = np.vectorize(lambda x: index_id_to_text_id_map[x])
    
    
    output_img_2_text = []

    retrieval_dataset = mscoco_5k_retrieval_dataset_clip(dataset_split, lasco_data_path, clip_checkpoint_path)
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

    for batch_idx, batch in enumerate(tqdm(retrieval_dataloader , desc="Retrieval Task: IMG-2-TEXT")):
        with torch.no_grad():
            batch['image']['pixel_values'] = batch['image']['pixel_values'].to(device_map)
            image_embeds = model.img_forward(batch)
            image_embeds = image_embeds['image-embeds']

        D, I = text_index.search(image_embeds.cpu(), k=1000)
        I = map_func_text(I)

        batch_size = len(batch['image-id'])
        for i in range(batch_size):
            output_img_2_text.append({
                'id': batch['id'][i],
                'image-id': batch['image-id'][i],
                'text-id': batch['text-id'][i],
                'text-raw': batch['text-raw'][i],
                'top_1000_ret_cands': I[i][:].tolist(),
                'top_1000_ret_cands_cos_sims': D[i][:].tolist()
                })

    with open(os.path.join(out_dir, exp_name, 'outputs'+'-mscoco-5k-[image-2-text]-'+checkpoint+'.json'), "w") as json_file:
        json.dump(output_img_2_text, json_file, indent=4)


    metrics = []
    ground_truths = np.array(list(map(lambda x: x['text-id'], output_img_2_text)))
    retrieved_candidates = np.array(list(map(lambda x: x['top_1000_ret_cands'], output_img_2_text)))

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

    with open(os.path.join(out_dir, exp_name, 'metrics'+'-mscoco-5k-[image-2-text]-'+checkpoint+'.json'), "w") as json_file:
        json.dump(metrics, json_file, indent=4)
        
    epoch = int(checkpoint[17:20])
    new_row = pd.DataFrame([{
        'checkpoint': checkpoint, 
        'epoch': epoch, 
        'Recall@1': Recall_1,
        'Recall@5': Recall_5,
        'Recall@10': Recall_10,
        'Recall@50': Recall_50,
        'Recall@100': Recall_100,
        'Recall@500': Recall_500,
        'Recall@1000': Recall_1000
    }])
    exp_results_i2t = pd.concat([exp_results_i2t, new_row], ignore_index=True)
    exp_results_i2t.to_csv(os.path.join(out_dir, exp_name, 'mscoco-5k-[image-2-text]-experiment_results.csv'), index=False)
        
    
    output_text_2_img = []

    retrieval_dataset = mscoco_5k_retrieval_dataset_clip(dataset_split, lasco_data_path, clip_checkpoint_path)
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

    for batch_idx, batch in enumerate(tqdm(retrieval_dataloader , desc="Retrieval Task: IMG-2-TEXT")):
        with torch.no_grad():
            batch['text']['input_ids'] = batch['text']['input_ids'].to(device_map)
            batch['text']['attention_mask'] = batch['text']['attention_mask'].to(device_map)
        
            text_embeds = model.text_forward(batch)
            text_embeds = text_embeds['text-embeds']

        D, I = image_index.search(text_embeds.cpu(), k=1000)
        I = map_func_image(I)

        batch_size = len(batch['text-id'])
        for i in range(batch_size):
            output_text_2_img.append({
                'id': batch['id'][i],
                'image-id': batch['image-id'][i],
                'text-id': batch['text-id'][i],
                'text-raw': batch['text-raw'][i],
                'top_1000_ret_cands': I[i][:].tolist(),
                'top_1000_ret_cands_cos_sims': D[i][:].tolist()
                })

    with open(os.path.join(out_dir, exp_name, 'outputs'+'-mscoco-5k-[text-2-image]-'+checkpoint+'.json'), "w") as json_file:
        json.dump(output_text_2_img, json_file, indent=4)


    metrics = []
    ground_truths = np.array(list(map(lambda x: x['image-id'], output_text_2_img)))
    retrieved_candidates = np.array(list(map(lambda x: x['top_1000_ret_cands'], output_text_2_img)))

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

    with open(os.path.join(out_dir, exp_name, 'metrics'+'-mscoco-5k-[text-2-image]-'+checkpoint+'.json'), "w") as json_file:
        json.dump(metrics, json_file, indent=4)
        
    epoch = int(checkpoint[17:20])
    new_row = pd.DataFrame([{
        'checkpoint': checkpoint, 
        'epoch': epoch, 
        'Recall@1': Recall_1,
        'Recall@5': Recall_5,
        'Recall@10': Recall_10,
        'Recall@50': Recall_50,
        'Recall@100': Recall_100,
        'Recall@500': Recall_500,
        'Recall@1000': Recall_1000
    }])
    exp_results_t2i = pd.concat([exp_results_t2i, new_row], ignore_index=True)
    exp_results_t2i.to_csv(os.path.join(out_dir, exp_name, 'mscoco-5k-[text-2-image]-experiment_results.csv'), index=False)