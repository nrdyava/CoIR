import torch

def loss_with_temp_rank(embeds_1, embeds_2, logit_scale):
    logits_per_A = torch.mm(embeds_1, embeds_2.t())
    diagonal_elements = torch.diag(logits_per_A)
    avg_rank = torch.sum(logits_per_A >= diagonal_elements.unsqueeze(1), dim = 1).float().mean()
    logits_per_A = logits_per_A*torch.clamp(logit_scale.exp(), max=100)
    logits_per_B = torch.mm(embeds_2, embeds_1.t())*torch.clamp(logit_scale.exp(), max=100)
    
    labels = torch.arange(logits_per_A.size(0), device=logits_per_A.device)

    loss_A = torch.nn.functional.cross_entropy(logits_per_A, labels)
    loss_B = torch.nn.functional.cross_entropy(logits_per_B, labels)
    loss = (loss_A + loss_B) / 2.0
    return loss, avg_rank


def loss_with_temp(embeds_1, embeds_2, logit_scale):
    logits_per_A = torch.mm(embeds_1, embeds_2.t())*torch.clamp(logit_scale.exp(), max=100)
    logits_per_B = torch.mm(embeds_2, embeds_1.t())*torch.clamp(logit_scale.exp(), max=100)
    
    labels = torch.arange(logits_per_A.size(0), device=logits_per_A.device)

    loss_A = torch.nn.functional.cross_entropy(logits_per_A, labels)
    loss_B = torch.nn.functional.cross_entropy(logits_per_B, labels)
    loss = (loss_A + loss_B) / 2.0
    return loss


def asymetric_loss_with_temp(target_hat_embeds, target_image_embeds, logit_scale, gpu_id):
    logits = torch.mm(target_hat_embeds, target_image_embeds.t())
    
    bs = logits.size(0)
    #print("DEBUG: batch_size in loss: ", logits.size(1)) ## Comment this after debugging
    labels = (gpu_id * bs + torch.arange(bs)).to(logits.device)
    
    #print("logits size ", logits.size())
    diagonal_elements = torch.diag(logits[:, gpu_id * bs: (gpu_id+1) * bs])
    #print("diag_size ", diagonal_elements.size())
    avg_rank = torch.sum(logits >= diagonal_elements.unsqueeze(1), dim = 1).float().mean()
    
    _max_score, max_idxs = torch.max(logits, 1)
    acc = (max_idxs == labels).sum() / bs
    
    logits = logits*torch.clamp(logit_scale.exp(), max=100)
    
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    return loss, avg_rank, acc



def asymetric_loss_with_temp_QN1(target_hat_embeds, target_image_embeds, logit_scale, gpu_id):
    logits = torch.mm(target_hat_embeds, target_image_embeds.t())
    
    bs = logits.size(0)
    #print("DEBUG: batch_size in loss: ", logits.size(1)) ## Comment this after debugging
    labels = (gpu_id * bs + torch.arange(bs)).to(logits.device)
    
    fh = logits[:, :bs]
    sh = torch.diagonal(logits[:, bs:]).unsqueeze(1)
    logits = torch.cat([fh, sh], dim = 1)
    
    #print("logits size ", logits.size())
    diagonal_elements = torch.diag(logits[:, gpu_id * bs: (gpu_id+1) * bs])
    #print("diag_size ", diagonal_elements.size())
    avg_rank = torch.sum(logits >= diagonal_elements.unsqueeze(1), dim = 1).float().mean()
    
    _max_score, max_idxs = torch.max(logits, 1)
    acc = (max_idxs == labels).sum() / bs
    
    logits = logits*torch.clamp(logit_scale.exp(), max=100)
    
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    return loss, avg_rank, acc
    
    