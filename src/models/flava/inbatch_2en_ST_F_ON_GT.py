import numpy as np
import torch
import torch.distributed as dist
import lightning as L
from transformers import FlavaModel

from src.models.model_utils import optimizer_dict
from src.losses import loss_fn_registry
from deepspeed.ops.adam import FusedAdam


class FLAVA_Model(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.model = FlavaModel.from_pretrained("facebook/flava-full")
        self.model.mm_hidden_size = None
        self.model.multimodal_model = None
        self.model.image_to_mm_projection = None
        self.model.text_to_mm_projection = None
        
        #Temperature parameters
        self.logit_scale = self.model.logit_scale
        if not self.config['loss_fn']['train_temperature']:
            self.logit_scale.requires_grad = False
        
        if self.config['image_encoder_mode'] == 'freeze':
            for param in self.model.image_model.parameters():
                param.requires_grad = False
        if self.config['text_encoder_mode'] == 'freeze':
            for param in self.model.text_model.parameters():
                param.requires_grad = False
        
        self.loss_fn = loss_fn_registry[config['loss_fn']['name']]
        self.gather_embeddings = config['gather_embeddings']
        self.save_hyperparameters()
    
    
    def img_forward(self, batch):
        image = batch['image']
        image_embeds = self.get_image_embeds(image)
        image_embeds = self.normalize_embeddings(image_embeds)
        return {'image-embeds': image_embeds}
    
    def img_txt_forward(self, batch):
        query_image = batch['query-image']
        query_text = batch['query-text']
        query_image_embeds = self.get_image_embeds(query_image)
        query_text_embeds = self.get_text_embeds(query_text)
        target_hat_embeds = self.fuse_embeddings(query_image_embeds, query_text_embeds)
        target_hat_embeds = self.normalize_embeddings(target_hat_embeds)
        return{
            'target-hat-embeds': target_hat_embeds
        }
       
    def get_image_embeds(self, image_inputs):
        image_features = self.model.get_image_features(**image_inputs)
        image_features = image_features[:, 0, :]
        return image_features

    def get_text_embeds(self, text_inputs):
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features[:, 0, :]
        return text_features
    
    def fuse_embeddings(self, img_embeds, txt_embeds):
        return img_embeds + txt_embeds

    def normalize_embeddings(self, embeds):
        return embeds / torch.linalg.vector_norm(embeds, ord=2, dim=1, keepdim=True)
    
    def img_txt_img_forward(self, batch):
        query_image = batch['query-image']
        query_text = batch['query-text']
        target_image = batch['target-image']

        query_image_embeds = self.get_image_embeds(query_image)
        query_text_embeds = self.get_text_embeds(query_text)
        target_image_embeds = self.get_image_embeds(target_image)

        return {
            'query-image-embeds': query_image_embeds, 
            'query-text-embeds': query_text_embeds, 
            'target-image-embeds': target_image_embeds
        }
    
    def training_step(self, batch, batch_idx):
        outs = self.img_txt_img_forward(batch)

        query_image_embeds = outs['query-image-embeds']
        query_text_embeds = outs['query-text-embeds']
        target_image_embeds = outs['target-image-embeds']
        
        batch_size = query_image_embeds.size(0)

        target_hat_embeds = self.fuse_embeddings(query_image_embeds, query_text_embeds)
        target_hat_embeds = self.normalize_embeddings(target_hat_embeds)
        
        target_image_embeds = self.normalize_embeddings(target_image_embeds)
        
        gpu_id = dist.get_rank()
        
        if self.gather_embeddings and dist.is_initialized():
            gathered_image_embeds = [torch.zeros_like(target_image_embeds) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_image_embeds, target_image_embeds)
            target_image_embeds = torch.cat(gathered_image_embeds, dim=0)
            loss, avg_rank, acc = self.loss_fn(target_hat_embeds, target_image_embeds, self.logit_scale, gpu_id)
        else:
            loss, avg_rank, acc = self.loss_fn(target_hat_embeds, target_image_embeds, self.logit_scale, gpu_id)
        
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        #print("DEBUG: batch_size: ", batch_size) ## Comment this after debugging
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size = batch_size)
        self.log('train_avg_rank', avg_rank.detach().cpu().numpy().item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size = batch_size)
        self.log('train_acc', acc.detach().cpu().numpy().item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size = batch_size)
        self.log('logit_scale', self.logit_scale.exp().detach().cpu().numpy().item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size = batch_size)
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size = batch_size)

        ## Debug metrics here. Comment out when not needed
        #target_hat_n_target_cosine_sim_mean = torch.mean(torch.nn.functional.cosine_similarity(target_hat_embeds, target_image_embeds[:, gpu_id * batch_size: (gpu_id+1) * batch_size], dim=1))
        #self.log('train_target_hat_n_target_cosine_sim_mean', target_hat_n_target_cosine_sim_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size = batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        outs = self.img_txt_img_forward(batch)

        query_image_embeds = outs['query-image-embeds']
        query_text_embeds = outs['query-text-embeds']
        target_image_embeds = outs['target-image-embeds']
        
        batch_size = query_image_embeds.size(0)

        target_hat_embeds = self.fuse_embeddings(query_image_embeds, query_text_embeds)
        target_hat_embeds = self.normalize_embeddings(target_hat_embeds)
        
        target_image_embeds = self.normalize_embeddings(target_image_embeds)
        
        gpu_id = dist.get_rank()
        
        if self.gather_embeddings and dist.is_initialized():
            gathered_image_embeds = [torch.zeros_like(target_image_embeds) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_image_embeds, target_image_embeds)
            target_image_embeds = torch.cat(gathered_image_embeds, dim=0)
            loss, avg_rank, acc = self.loss_fn(target_hat_embeds, target_image_embeds, self.logit_scale, gpu_id)
        else:
            loss, avg_rank, acc = self.loss_fn(target_hat_embeds, target_image_embeds, self.logit_scale, gpu_id)
        
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size = batch_size)
        self.log('val_avg_rank', avg_rank.detach().cpu().numpy().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size = batch_size)
        self.log('val_acc', acc.detach().cpu().numpy().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size = batch_size)

        ## Debug metrics here. Comment out when not needed
        #target_hat_n_target_cosine_sim_mean = torch.mean(torch.nn.functional.cosine_similarity(target_hat_embeds, target_image_embeds[:, gpu_id * batch_size: (gpu_id+1) * batch_size], dim=1))
        #self.log('val_target_hat_n_target_cosine_sim_mean', target_hat_n_target_cosine_sim_mean, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size = batch_size)  
    
    def configure_optimizers(self):
        self.optimizer = optimizer_dict[self.config['optimizer']['name']](
            params = filter(lambda p: p.requires_grad, self.parameters()), 
            lr = self.config['optimizer']['lr'],
            weight_decay = self.config['optimizer']['weight_decay']
        )

        """
        self.optimizer = FusedAdam(
            params = filter(lambda p: p.requires_grad, self.parameters()), 
            lr = self.config['optimizer']['lr'],
            weight_decay = self.config['optimizer']['weight_decay']
        )
        """
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.trainer.max_epochs, 
            eta_min=self.config['optimizer']['eta_min']
        )
        
        return {
            'optimizer': self.optimizer, 
            'lr_scheduler': self.scheduler
        }