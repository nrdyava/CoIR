import os
import sys
import torch
import copy
import yaml
import pytz
import lightning as L
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from src.losses import loss_fn_registry



class CLIPModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.modelckpt = config['checkpoint_path']
        self.image_encoder_mode = config['image_encoder_mode']
        self.text_encoder_mode = config['text_encoder_mode']

        self.temperature = config['loss_fn']['temperature']
        self.train_temperature = config['loss_fn']['train_temperature']
        self.dotp_clip = config['loss_fn']["dotp_clip"]

        self.lr = config['optimizer']['lr']

        if self.train_temperature == True:
            self.temperature = torch.nn.Parameter(torch.tensor(self.temperature))
        else:
            self.temperature = torch.tensor(self.temperature)

        self.loss_fn = loss_fn_registry[config['loss_fn']['name']]

        #Models
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path = self.modelckpt, local_files_only = True)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path = self.modelckpt, local_files_only = True)
        
        if self.image_encoder_mode == 'freeze':
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        if self.text_encoder_mode == 'freeze':
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                
        self.save_hyperparameters()

             
    def forward(self, batch):
        query_image = batch['query-image']
        target_image = batch['target-image']
        query_text = batch['query-text']

        query_image_embeds = self.image_encoder(**query_image).image_embeds
        query_image_embeds = query_image_embeds/torch.linalg.vector_norm(query_image_embeds, ord=2, dim=1,  keepdim=True)

        target_image_embeds = self.image_encoder(**target_image).image_embeds
        target_image_embeds = target_image_embeds/torch.linalg.vector_norm(target_image_embeds, ord=2, dim=1,  keepdim=True)

        query_text_embeds = self.text_encoder(**query_text).text_embeds
        query_text_embeds = query_text_embeds/torch.linalg.vector_norm(query_text_embeds, ord=2, dim=1,  keepdim=True)

        return {'query_image_embeds': query_image_embeds, 'target_image_embeds': target_image_embeds, 'query_text_embeds': query_text_embeds}


    def training_step(self, batch, batch_idx):
        outs = self.forward(batch)

        query_image_embeds = outs['query_image_embeds']
        target_image_embeds = outs['target_image_embeds']
        query_text_embeds = outs['query_text_embeds']
        
        target_hat_embeds = query_image_embeds + query_text_embeds
        target_hat_embeds = target_hat_embeds/torch.linalg.vector_norm(target_hat_embeds, ord=2, dim=1,  keepdim=True)

        loss = self.loss_fn(target_hat_embeds, target_image_embeds, self.temperature, self.dotp_clip)
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        outs = self.forward(batch)

        query_image_embeds = outs['query_image_embeds']
        target_image_embeds = outs['target_image_embeds']
        query_text_embeds = outs['query_text_embeds']
        
        target_hat_embeds = query_image_embeds + query_text_embeds
        target_hat_embeds = target_hat_embeds/torch.linalg.vector_norm(target_hat_embeds, ord=2, dim=1,  keepdim=True)

        loss = self.loss_fn(target_hat_embeds, target_image_embeds, self.temperature, self.dotp_clip)
        self.log('val_loss', loss)


    #def test_step(self, batch, batch_idx):

    #def predict_steps(self):

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer