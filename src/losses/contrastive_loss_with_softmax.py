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