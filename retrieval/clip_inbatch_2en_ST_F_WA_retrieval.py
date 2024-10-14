#!/usr/bin/env python
# coding: utf-8

device = 3
checkpoint = 'checkpoint-epoch=04-val_loss=0.6134115458.ckpt'
exp_name = '2024-09-28-13-07-44-728101 + clip_inbatch_2en_ST_F_WA'
batch_size=64

out_dir = '/proj/vondrick4/naveen/coir-ret-results'
runs_dir = '/proj/vondrick4/naveen/coir-runs'
dataset_split = 'val'
lasco_data_path = '/local/vondrick/naveen/coir-data/LaSCo'
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

from src.datasets.lasco_corpus_dataset import lasco_corpus_dataset_clip
from src.datasets.lasco_retrieval_dataset import lasco_retrieval_dataset_clip
from src.metrics.metrics import calculate_recall
from src.models.clip.inbatch_2en_ST_F_WA import CLIPModel


# In[ ]:





# In[4]:


os.makedirs(os.path.join(out_dir, exp_name), exist_ok=True)


# In[ ]:





# In[5]:


checkpoint_path = os.path.join(runs_dir, exp_name, checkpoint)


# In[6]:


model = CLIPModel.load_from_checkpoint(
    checkpoint_path = checkpoint_path, 
    map_location=device_map
)
print('Model loaded')


# In[ ]:





# In[7]:


model.eval()


# In[ ]:





# In[8]:


clip_checkpoint_path = clip_checkpoints[model.config['model_type']]


# In[ ]:





# In[9]:


print('Using device: {}'.format(device_map))


# In[ ]:





# In[10]:


d = model.image_encoder.config.projection_dim
index = faiss.IndexFlatIP(d)

res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, device, index)


# In[ ]:





# In[11]:


corpus_dataset = lasco_corpus_dataset_clip(dataset_split, lasco_data_path, clip_checkpoint_path)
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


# In[12]:


retrieval_dataset = lasco_retrieval_dataset_clip(dataset_split, lasco_data_path, clip_checkpoint_path)
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


# In[ ]:





# In[13]:


# Create embeddings of images in the corpus
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


# In[ ]:





# In[14]:


map_func = np.vectorize(lambda x: index_id_to_image_id_map[x])


# In[ ]:





# In[15]:


output = []

for batch_idx, batch in enumerate(tqdm(retrieval_dataloader , desc="Retrieval Task")):
    with torch.no_grad():
        batch['query-image']['pixel_values'] = batch['query-image']['pixel_values'].to(device_map)
        batch['query-text']['input_ids'] = batch['query-text']['input_ids'].to(device_map)
        batch['query-text']['attention_mask'] = batch['query-text']['attention_mask'].to(device_map)
        
        target_hat_embeds = model.img_txt_forward(batch)

    D, I = index.search(target_hat_embeds['target-hat-embeds'].cpu(), k=50)
    I = map_func(I)

    batch_size = len(batch['query-image-id'])
    for i in range(batch_size):
        output.append({
            'id': batch['id'][i],
            'query-image-id': batch['query-image-id'][i],
            'target-image-id': batch['target-image-id'][i],
            'query-text-raw': batch['query-text-raw'][i],
            'top_50_ret_cands': I[i][:].tolist(),
            'top_50_ret_cands_cos_sims': D[i][:].tolist()
        })


# In[ ]:





# In[16]:


with open(os.path.join(out_dir, exp_name, 'outputs'+'-'+checkpoint+'.json'), "w") as json_file:
    json.dump(output, json_file, indent=4)


# In[ ]:





# In[17]:


metrics = []
ground_truths = np.array(list(map(lambda x: x['target-image-id'], output)))
retrieved_candidates = np.array(list(map(lambda x: x['top_50_ret_cands'], output)))

metrics.append({"Recall@1": 100*calculate_recall(ground_truths, retrieved_candidates, 1)})
metrics.append({"Recall@5": 100*calculate_recall(ground_truths, retrieved_candidates, 5)})
metrics.append({"Recall@10": 100*calculate_recall(ground_truths, retrieved_candidates, 10)})
metrics.append({"Recall@50": 100*calculate_recall(ground_truths, retrieved_candidates, 50)})

with open(os.path.join(out_dir, exp_name, 'metrics'+'-'+checkpoint+'.json'), "w") as json_file:
    json.dump(metrics, json_file, indent=4)


# In[ ]:





# In[18]:


metrics


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




