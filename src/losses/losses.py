import torch


def vanilla_constrastive_cross_entrpy_loss(embeds_1, embeds_2, temperature, dotp_clip):
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'mean')
    dotp = torch.nn.functional.cosine_similarity(embeds_1, embeds_2, dim=1)*torch.exp(temperature)
    dotp = torch.clamp(dotp, max = dotp_clip)
    labels = torch.ones(dotp.shape[0]).to(embeds_1.device)
    return cross_entropy_loss(dotp, labels)