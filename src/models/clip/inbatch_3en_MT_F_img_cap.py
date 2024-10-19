import numpy as np
import torch
import lightning as L
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from src.models.model_utils import optimizer_dict
from src.losses import loss_fn_registry
from deepspeed.ops.adam import FusedAdam


class CLIPModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.pretrained_clip_model_ckpt = self.config['clip_checkpoints'][self.config['model_type']]
        self.image_encoder_mode = self.config['image_encoder_mode']
        self.text_encoder_mode_align = self.config['text_encoder_mode_align']
        self.text_encoder_mode_instruct = self.config['text_encoder_mode_instruct']
        
        #Temperature parameters
        self.logit_scale_align = torch.nn.Parameter(torch.ones([]) * np.log(1 / self.config['loss_fn']['temperature_align']))
        if not self.config['loss_fn']['train_temperature_align']:
            self.logit_scale_align.requires_grad = False
        
        self.logit_scale_instruct = torch.nn.Parameter(torch.ones([]) * np.log(1 / self.config['loss_fn']['temperature_instruct']))
        if not self.config['loss_fn']['train_temperature_instruct']:
            self.logit_scale_instruct.requires_grad = False

        # Models
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_clip_model_ckpt, 
            local_files_only=True
            )
        self.text_encoder_align = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_clip_model_ckpt, 
            local_files_only=True
            )
        self.text_encoder_instruct = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_clip_model_ckpt, 
            local_files_only=True
            )
        # Define any extra model layers here
        
        if self.image_encoder_mode == 'freeze':
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        if self.text_encoder_mode_align == 'freeze':
            for param in self.text_encoder_align.parameters():
                param.requires_grad = False
        if self.text_encoder_mode_instruct == 'freeze':
            for param in self.text_encoder_instruct.parameters():
                param.requires_grad = False
        
        self.loss_fn_align = loss_fn_registry[config['loss_fn']['align_loss_name']]
        self.loss_fn_instruct = loss_fn_registry[config['loss_fn']['instruct_loss_name']]
        
        self.automatic_optimization = False
        self.save_hyperparameters()
    
    
    def img_forward(self, batch):
        image = batch['image']
        image_embeds = self.image_encoder(**image).image_embeds
        image_embeds = image_embeds / torch.linalg.vector_norm(image_embeds, ord=2, dim=1, keepdim=True)
        return {'image-embeds': image_embeds}
    
    
    def img_txt_forward(self, batch):
        query_image = batch['query-image']
        query_text = batch['query-text']

        query_image_embeds = self.image_encoder(**query_image).image_embeds
        query_image_embeds = query_image_embeds / torch.linalg.vector_norm(query_image_embeds, ord=2, dim=1,keepdim=True)

        query_text_embeds = self.text_encoder(**query_text).text_embeds
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
        align_image = batch['align-image']
        align_text = batch['align-text']

        query_image_embeds = self.image_encoder(**query_image).image_embeds
        query_image_embeds = query_image_embeds / torch.linalg.vector_norm(query_image_embeds, ord=2, dim=1,keepdim=True)
        
        query_text_embeds = self.text_encoder_instruct(**query_text).text_embeds
        query_text_embeds = query_text_embeds / torch.linalg.vector_norm(query_text_embeds, ord=2, dim=1, keepdim=True)

        target_image_embeds = self.image_encoder(**target_image).image_embeds
        target_image_embeds = target_image_embeds / torch.linalg.vector_norm(target_image_embeds, ord=2, dim=1,keepdim=True)
        
        align_image_embeds = self.image_encoder(**align_image).image_embeds
        align_image_embeds = align_image_embeds / torch.linalg.vector_norm(align_image_embeds, ord=2, dim=1,keepdim=True)
        
        align_text_embeds = self.text_encoder_align(**align_text).text_embeds
        align_text_embeds = align_text_embeds / torch.linalg.vector_norm(align_text_embeds, ord=2, dim=1,keepdim=True)

        return {
            'query-image-embeds': query_image_embeds, 
            'query-text-embeds': query_text_embeds, 
            'target-image-embeds': target_image_embeds,
            'align-image-embeds': align_image_embeds,
            'align-text-embeds': align_text_embeds
        }
    
    
    def training_step(self, batch, batch_idx):
        outs = self.img_txt_img_forward(batch)

        query_image_embeds = outs['query-image-embeds']
        query_text_embeds = outs['query-text-embeds']
        target_image_embeds = outs['target-image-embeds']
        align_image_embeds = outs['align-image-embeds']
        align_text_embeds = outs['align-text-embeds']

        target_hat_embeds = query_image_embeds + query_text_embeds
        target_hat_embeds = target_hat_embeds / torch.linalg.vector_norm(target_hat_embeds, ord=2, dim=1, keepdim=True)

        loss_instruct, avg_rank = self.loss_fn_instruct(target_hat_embeds, target_image_embeds, self.logit_scale_instruct)
        loss_align = self.loss_fn_align(align_image_embeds, align_text_embeds, self.logit_scale_align)
        loss = (1/2)*(loss_instruct + loss_align)
        
        #optimizer_align, optimizer_instruct = self.optimizers()
        optimizer_align = self.optimizers()[0]
        optimizer_instruct = self.optimizers()[1]
        optimizer_align.zero_grad()
        optimizer_instruct.zero_grad()
        
        #self.manual_backward(loss_align, optimizer=optimizer_align)
        #self.manual_backward(loss_instruct, optimizer=optimizer_instruct)
        self.manual_backward(loss_align)
        self.manual_backward(loss_instruct)
        
        optimizer_align.step()
        optimizer_instruct.step()
        
        scheduler_align, scheduler_instruct = self.lr_schedulers()
        scheduler_align.step()
        scheduler_instruct.step()
        
        align_lr = scheduler_align.get_last_lr()[0]
        instruct_lr = scheduler_instruct.get_last_lr()[0]
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_align', loss_align, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_loss_instruct', loss_instruct, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        self.log('logit_scale_align', self.logit_scale_align.exp().detach().cpu().numpy().item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('logit_scale_instruct', self.logit_scale_instruct.exp().detach().cpu().numpy().item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        self.log('train_avg_rank', avg_rank.detach().cpu().numpy().item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('align_LR', align_lr, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('instruct_LR', instruct_lr, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        ## Debug metrics here. Comment out when not needed
        target_hat_n_target_cosine_sim_mean = torch.mean(torch.nn.functional.cosine_similarity(target_hat_embeds, target_image_embeds, dim=1))
        self.log('train_target_hat_n_target_cosine_sim_mean', target_hat_n_target_cosine_sim_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        align_image_n_text_cosine_sim_mean = torch.mean(torch.nn.functional.cosine_similarity(align_image_embeds, align_text_embeds, dim=1))
        self.log('train_align_image_n_text_cosine_sim_mean', align_image_n_text_cosine_sim_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    

    def validation_step(self, batch, batch_idx):
        outs = self.img_txt_img_forward(batch)

        query_image_embeds = outs['query-image-embeds']
        query_text_embeds = outs['query-text-embeds']
        target_image_embeds = outs['target-image-embeds']
        align_image_embeds = outs['align-image-embeds']
        align_text_embeds = outs['align-text-embeds']

        target_hat_embeds = query_image_embeds + query_text_embeds
        target_hat_embeds = target_hat_embeds / torch.linalg.vector_norm(target_hat_embeds, ord=2, dim=1, keepdim=True)

        loss_instruct, avg_rank = self.loss_fn_instruct(target_hat_embeds, target_image_embeds, self.logit_scale_instruct)
        loss_align = self.loss_fn_align(align_image_embeds, align_text_embeds, self.logit_scale_align)
        loss = (1/2)*(loss_instruct + loss_align)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss_align', loss_align, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_loss_instruct', loss_instruct, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        self.log('val_avg_rank', avg_rank.detach().cpu().numpy().item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        ## Debug metrics here. Comment out when not needed
        target_hat_n_target_cosine_sim_mean = torch.mean(torch.nn.functional.cosine_similarity(target_hat_embeds, target_image_embeds, dim=1))
        self.log('val_target_hat_n_target_cosine_sim_mean', target_hat_n_target_cosine_sim_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        align_image_n_text_cosine_sim_mean = torch.mean(torch.nn.functional.cosine_similarity(align_image_embeds, align_text_embeds, dim=1))
        self.log('val_align_image_n_text_cosine_sim_mean', align_image_n_text_cosine_sim_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
    
    def configure_optimizers(self):
        self.optimizer_align = optimizer_dict[self.config['optimizer']['align_optimizer']['name']](
            params = filter(lambda p: p.requires_grad, list(self.text_encoder_align.parameters()) + list(self.image_encoder.parameters())), 
            lr = self.config['optimizer']['align_optimizer']['lr'],
            weight_decay = self.config['optimizer']['align_optimizer']['weight_decay']
        )
        
        self.optimizer_instruct = optimizer_dict[self.config['optimizer']['instruct_optimizer']['name']](
            params = filter(lambda p: p.requires_grad, list(self.text_encoder_instruct.parameters()) + list(self.image_encoder.parameters())), 
            lr = self.config['optimizer']['instruct_optimizer']['lr'],
            weight_decay = self.config['optimizer']['instruct_optimizer']['weight_decay']
        )
        
        self.scheduler_align = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_align, 
            T_max=self.trainer.max_epochs, 
            eta_min=self.config['optimizer']['align_optimizer']['eta_min']
        )
        
        self.scheduler_instruct = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_instruct, 
            T_max=self.trainer.max_epochs, 
            eta_min=self.config['optimizer']['instruct_optimizer']['eta_min']
        )
        
        return (
            [self.optimizer_align, self.optimizer_instruct],
            [{'scheduler': self.scheduler_align, 'interval': self.config['optimizer']['align_optimizer']['interval'], 'frequency': self.config['optimizer']['align_optimizer']['frequency']}, 
             {'scheduler': self.scheduler_instruct, 'interval': self.config['optimizer']['instruct_optimizer']['interval'], 'frequency': self.config['optimizer']['instruct_optimizer']['frequency']}]
        )