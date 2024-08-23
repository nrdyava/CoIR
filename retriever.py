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
from src.datasets.lasco_datasets import lasco_corpus_dataset, lasco_retrieval_dataset


def main(config):
    # Set devices for the model and FAISS index
    model_device = 'cuda: {}'.format(config['model_gpu_device_id'])

    if config['faiss_use_gpu'] == True:
        faiss_device = 'cuda: {}'.format(config['faiss_gpu_device_id'])
    else:
        faiss_device = 'cpu'

    print('Model device: {}'.format(model_device))
    print('FAISS device: {}'.format(faiss_device))

    # Load the model and map to appropriate device
    model = CLIPModel.load_from_checkpoint(config['pl_ckpt_path'], map_location=config['model_gpu_device_id'])

    if config['eval_model_type'] == 'baseline':
        model.image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path=config['checkpoint_path'], local_files_only=True)
        model.text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path=config['checkpoint_path'], local_files_only=True)

    model.eval()
    print('Model loaded')

    # Configure the FAISS Index
    d = config['d']
    index = faiss.IndexFlatIP(d)

    if config['faiss_use_gpu'] == True:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, config['faiss_gpu_device_id'], index)

    # Datasets & DataLoaders
    corpus_dataset = lasco_corpus_dataset(config)
    corpus_dataloader = DataLoader(
        dataset=corpus_dataset,
        collate_fn=corpus_dataset.collate_fn,
        batch_size=config['dataloader']['batch_size'],
        shuffle=config['dataloader']['shuffle'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        drop_last=config['dataloader']['drop_last'],
        persistent_workers=config['dataloader']['persistent_workers']
    )

    retrieval_dataset = lasco_retrieval_dataset(config)
    retrieval_dataloader = DataLoader(
        dataset=retrieval_dataset,
        collate_fn=retrieval_dataset.collate_fn,
        batch_size=config['dataloader']['batch_size'],
        shuffle=config['dataloader']['shuffle'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        drop_last=config['dataloader']['drop_last'],
        persistent_workers=config['dataloader']['persistent_workers']
    )


    # Create embeddings of images in the corpus
    index_cntr = 0
    index_id_to_image_id_map = {}

    for batch_idx, batch in tqdm(enumerate(corpus_dataloader)):
        with torch.no_grad():
            image_embeds = model.image_forward(batch)
        index.add(image_embeds['image-embeds'].numpy())

        batch_len = len(batch['image-key'])
        batch_start_indx = index_cntr
        batch_end_indx = batch_start_indx + batch_len

        for key, value in zip(list(range(batch_start_indx, batch_end_indx)), batch['image-key']):
            index_id_to_image_id_map[key] = value
        index_cntr += batch_len

    # Function for fast mapping of retrieved index keys to Image-IDs
    map_func = np.vectorize(lambda x: index_id_to_image_id_map[x])

    # Table to store results after search from the index
    results_table = np.zeros((1, 52), dtype='int64')

    for batch_idx, batch in tqdm(enumerate(retrieval_dataloader)):
        with torch.no_grad():
            outs = model.retriever_forward(batch)

        target_hat_embeds = outs['query_image_embeds'] + outs['query_text_embeds']
        target_hat_embeds = target_hat_embeds / torch.linalg.vector_norm(target_hat_embeds, ord=2, dim=1, keepdim=True)

        D, I = index.search(target_hat_embeds, 50)
        I = map_func(I)

        batch_result_table = np.concatenate(
            (
                np.array(batch['query-image-id'], dtype='int64').reshape(-1, 1),
                np.array(batch['target-image-id'], dtype='int64').reshape(-1, 1),
                I
            ), axis=1)

        results_table = np.concatenate((results_table, batch_result_table), axis=0)

    # Remove the initialized dummy first row
    results_table = results_table[1:, :]

    recall_1 = calculate_recall(results_table, 1)
    recall_5 = calculate_recall(results_table, 5)
    recall_10 = calculate_recall(results_table, 10)
    recall_50 = calculate_recall(results_table, 50)

    table = [
        ["Recall@1", recall_1],
        ["Recall@5", recall_5],
        ["Recall@10", recall_10],
        ["Recall@50", recall_50]
    ]
    formatted_table = tabulate(table, headers=["Recall Type", "Value"], floatfmt=".6f")
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