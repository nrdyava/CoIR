import os
import sys
import torch
import copy
import yaml
import pytz
import lightning as L
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from src.losses import loss_fn_registry


class CLIPModelINBATCH_MT(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.modelckpt = config['checkpoint_path']
        self.image_encoder_mode = config['image_encoder_mode']
        self.text_encoder_mode = config['text_encoder_mode']

        self.temperature = config['loss_fn']['temperature']
        self.train_temperature = config['loss_fn']['train_temperature']
        self.temp_clamp_max = config['loss_fn']['temp_clamp_max']

        if self.train_temperature == True:
            self.temperature = torch.nn.parameter.Parameter(torch.tensor(self.temperature), requires_grad=True)
        else:
            self.temperature = torch.nn.parameter.Parameter(torch.tensor(self.temperature), requires_grad=False)

        self.lr = config['optimizer']['lr']

        self.loss_fn = loss_fn_registry[config['loss_fn']['name']]

        # Models
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path=self.modelckpt, local_files_only=True)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path=self.modelckpt, local_files_only=True)

        if self.image_encoder_mode == 'freeze':
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        if self.text_encoder_mode == 'freeze':
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.save_hyperparameters()

    def image_forward(self, batch):
        image = batch['image']
        image_embeds = self.image_encoder(**image).image_embeds
        image_embeds = image_embeds / torch.linalg.vector_norm(image_embeds, ord=2, dim=1, keepdim=True)
        return {'image-embeds': image_embeds}

    def retriever_forward(self, batch):
        query_image = batch['query-image']
        query_text = batch['query-text']

        query_image_embeds = self.image_encoder(**query_image).image_embeds
        query_image_embeds = query_image_embeds / torch.linalg.vector_norm(query_image_embeds, ord=2, dim=1,keepdim=True)

        query_text_embeds = self.text_encoder(**query_text).text_embeds
        query_text_embeds = query_text_embeds / torch.linalg.vector_norm(query_text_embeds, ord=2, dim=1, keepdim=True)

        return {'query_image_embeds': query_image_embeds, 'query_text_embeds': query_text_embeds}

    def forward(self, batch):
        query_image = batch['query-image']
        target_image = batch['target-image']
        query_text = batch['query-text']

        query_image_embeds = self.image_encoder(**query_image).image_embeds
        query_image_embeds = query_image_embeds / torch.linalg.vector_norm(query_image_embeds, ord=2, dim=1,keepdim=True)

        target_image_embeds = self.image_encoder(**target_image).image_embeds
        target_image_embeds = target_image_embeds / torch.linalg.vector_norm(target_image_embeds, ord=2, dim=1,keepdim=True)

        query_text_embeds = self.text_encoder(**query_text).text_embeds
        query_text_embeds = query_text_embeds / torch.linalg.vector_norm(query_text_embeds, ord=2, dim=1, keepdim=True)

        return {'query_image_embeds': query_image_embeds, 'target_image_embeds': target_image_embeds, 'query_text_embeds': query_text_embeds}

    def training_step(self, batch, batch_idx):
        outs = self.forward(batch)

        query_image_embeds = outs['query_image_embeds']
        target_image_embeds = outs['target_image_embeds']
        query_text_embeds = outs['query_text_embeds']

        target_hat_embeds_forward = query_image_embeds + query_text_embeds
        target_hat_embeds_forward = target_hat_embeds_forward / torch.linalg.vector_norm(target_hat_embeds_forward, ord=2, dim=1, keepdim=True)

        target_hat_embeds_backward = target_image_embeds - query_text_embeds
        target_hat_embeds_backward = target_hat_embeds_backward / torch.linalg.vector_norm(target_hat_embeds_backward, ord=2, dim=1, keepdim=True)

        loss_forward =  self.loss_fn(target_hat_embeds_forward, target_image_embeds, self.temperature, self.temp_clamp_max, self.config)
        loss_backward = self.loss_fn(target_hat_embeds_backward, query_image_embeds, self.temperature, self.temp_clamp_max, self.config)
        loss = (loss_forward + loss_backward)/2.0

        self.log('train_loss_forward', loss_forward, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_loss_backward', loss_backward, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        ## Debug metrics here. Comment out when not needed
        target_hat_n_target_cosine_sim_forward_mean = torch.mean(torch.nn.functional.cosine_similarity(target_hat_embeds_forward, target_image_embeds, dim=1))
        target_hat_n_target_cosine_sim_backward_mean = torch.mean(torch.nn.functional.cosine_similarity(target_hat_embeds_backward, query_image_embeds, dim=1))
        self.log('train_target_hat_n_target_cosine_sim_forward_mean', target_hat_n_target_cosine_sim_forward_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_target_hat_n_target_cosine_sim_backward_mean', target_hat_n_target_cosine_sim_backward_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outs = self.forward(batch)

        query_image_embeds = outs['query_image_embeds']
        target_image_embeds = outs['target_image_embeds']
        query_text_embeds = outs['query_text_embeds']

        target_hat_embeds_forward = query_image_embeds + query_text_embeds
        target_hat_embeds_forward = target_hat_embeds_forward / torch.linalg.vector_norm(target_hat_embeds_forward, ord=2, dim=1, keepdim=True)

        target_hat_embeds_backward = target_image_embeds - query_text_embeds
        target_hat_embeds_backward = target_hat_embeds_backward / torch.linalg.vector_norm(target_hat_embeds_backward, ord=2, dim=1, keepdim=True)

        loss_forward = self.loss_fn(target_hat_embeds_forward, target_image_embeds, self.temperature, self.temp_clamp_max, self.config)
        loss_backward = self.loss_fn(target_hat_embeds_backward, query_image_embeds, self.temperature, self.temp_clamp_max, self.config)
        loss = (loss_forward + loss_backward) / 2.0

        self.log('val_loss_forward', loss_forward, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_loss_backward', loss_backward, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        ## Debug metrics here. Comment out when not needed
        target_hat_n_target_cosine_sim_forward_mean = torch.mean(torch.nn.functional.cosine_similarity(target_hat_embeds_forward, target_image_embeds, dim=1))
        target_hat_n_target_cosine_sim_backward_mean = torch.mean(torch.nn.functional.cosine_similarity(target_hat_embeds_backward, query_image_embeds, dim=1))
        self.log('val_target_hat_n_target_cosine_sim_forward_mean', target_hat_n_target_cosine_sim_forward_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_target_hat_n_target_cosine_sim_backward_mean', target_hat_n_target_cosine_sim_backward_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)


    # def test_step(self, batch, batch_idx):

    # def predict_steps(self):

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return self.optimizer