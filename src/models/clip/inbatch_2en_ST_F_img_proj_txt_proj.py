import numpy as np
import torch
import lightning as L
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from src.models.model_utils import optimizer_dict
from src.losses import loss_fn_registry


class CLIPModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.pretrained_clip_model_ckpt = self.config['clip_checkpoints'][self.config['model_type']]
        self.image_encoder_mode = self.config['image_encoder_mode']
        self.text_encoder_mode = self.config['text_encoder_mode']
        
        #Temperature parameters
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / self.config['loss_fn']['temperature']))
        if not self.config['loss_fn']['train_temperature']:
            self.logit_scale.requires_grad = False

        # Models
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_clip_model_ckpt, 
            local_files_only=True
            )
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_clip_model_ckpt, 
            local_files_only=True
            )
        self.image_vector_proj_mat = torch.nn.Linear(self.image_encoder.config.projection_dim, self.text_encoder.config.projection_dim)
        self.text_vector_proj_mat = torch.nn.Linear(self.text_encoder.config.projection_dim, self.text_encoder.config.projection_dim)
        # Define any extra model layers here
        
        if self.image_encoder_mode == 'freeze':
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        if self.text_encoder_mode == 'freeze':
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        self.loss_fn = loss_fn_registry[config['loss_fn']['name']]
        self.save_hyperparameters()
    
    
    def img_forward(self, batch):
        image = batch['image']
        image_embeds = self.image_encoder(**image).image_embeds
        image_embeds = self.image_vector_proj_mat(image_embeds)
        image_embeds = image_embeds / torch.linalg.vector_norm(image_embeds, ord=2, dim=1, keepdim=True)
        return {'image-embeds': image_embeds}
    
    
    def img_txt_forward(self, batch):
        query_image = batch['query-image']
        query_text = batch['query-text']
        
        query_image_embeds = self.image_encoder(**query_image).image_embeds
        query_image_embeds = self.image_vector_proj_mat(query_image_embeds)
        query_image_embeds = query_image_embeds / torch.linalg.vector_norm(query_image_embeds, ord=2, dim=1,keepdim=True)
        
        query_text_embeds = self.text_encoder(**query_text).text_embeds
        query_text_embeds = self.text_vector_proj_mat(query_text_embeds)
        query_text_embeds = query_text_embeds / torch.linalg.vector_norm(query_text_embeds, ord=2, dim=1, keepdim=True)
        
        target_hat_embeds = query_image_embeds + query_text_embeds
        target_hat_embeds = target_hat_embeds / torch.linalg.vector_norm(target_hat_embeds, ord=2, dim=1, keepdim=True)

        return{
            'target-hat-embeds': target_hat_embeds
        }
    
    
    def img_txt_img_forward(self, batch):
        query_image = batch['query-image']
        query_text = batch['query-text']
        target_image = batch['target-image']

        query_image_embeds = self.image_encoder(**query_image).image_embeds
        query_image_embeds = self.image_vector_proj_mat(query_image_embeds)
        query_image_embeds = query_image_embeds / torch.linalg.vector_norm(query_image_embeds, ord=2, dim=1,keepdim=True)
        
        query_text_embeds = self.text_encoder(**query_text).text_embeds
        query_text_embeds = self.text_vector_proj_mat(query_text_embeds)
        query_text_embeds = query_text_embeds / torch.linalg.vector_norm(query_text_embeds, ord=2, dim=1, keepdim=True)

        target_image_embeds = self.image_encoder(**target_image).image_embeds
        target_image_embeds = self.image_vector_proj_mat(target_image_embeds)
        target_image_embeds = target_image_embeds / torch.linalg.vector_norm(target_image_embeds, ord=2, dim=1,keepdim=True)

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

        target_hat_embeds = query_image_embeds + query_text_embeds
        target_hat_embeds = target_hat_embeds / torch.linalg.vector_norm(target_hat_embeds, ord=2, dim=1, keepdim=True)

        loss, avg_rank = self.loss_fn(target_hat_embeds, target_image_embeds, self.logit_scale)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('logit_scale', self.logit_scale.exp().detach().cpu().numpy().item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_avg_rank', avg_rank.detach().cpu().numpy().item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        ## Debug metrics here. Comment out when not needed
        target_hat_n_target_cosine_sim_mean = torch.mean(torch.nn.functional.cosine_similarity(target_hat_embeds, target_image_embeds, dim=1))
        self.log('train_target_hat_n_target_cosine_sim_mean', target_hat_n_target_cosine_sim_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch, batch_idx):
        outs = self.img_txt_img_forward(batch)

        query_image_embeds = outs['query-image-embeds']
        query_text_embeds = outs['query-text-embeds']
        target_image_embeds = outs['target-image-embeds']

        target_hat_embeds = query_image_embeds + query_text_embeds
        target_hat_embeds = target_hat_embeds / torch.linalg.vector_norm(target_hat_embeds, ord=2, dim=1, keepdim=True)

        loss, avg_rank = self.loss_fn(target_hat_embeds, target_image_embeds, self.logit_scale)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_avg_rank', avg_rank.detach().cpu().numpy().item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        ## Debug metrics here. Comment out when not needed
        target_hat_n_target_cosine_sim_mean = torch.mean(torch.nn.functional.cosine_similarity(target_hat_embeds, target_image_embeds, dim=1))
        self.log('val_target_hat_n_target_cosine_sim_mean', target_hat_n_target_cosine_sim_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
    
    def configure_optimizers(self):
        self.optimizer = optimizer_dict[self.config['optimizer']['name']](
            params = filter(lambda p: p.requires_grad, self.parameters()), 
            lr = self.config['optimizer']['lr'],
            weight_decay = self.config['optimizer']['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.trainer.max_epochs, 
            eta_min=self.config['optimizer']['eta_min']
        )
        
        return {
            'optimizer': self.optimizer, 
            'lr_scheduler': self.scheduler
        }