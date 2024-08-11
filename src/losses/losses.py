import torch


def vanilla_constrastive_cross_entrpy_loss(embeds_1, embeds_2, config):
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'mean')
    dotp = torch.nn.functional.cosine_similarity(embeds_1, embeds_2, dim=1)*torch.exp(config['loss_fn']['temperature'])
    dotp = torch.clamp(dotp, max = config['loss_fn']['dotp_clip'])
    labels = torch.ones(dotp.shape[0]).to(embeds_1.device)
    return cross_entropy_loss(dotp, labels)


def cosine_embedding_loss(embeds_1, embeds_2, config):
    return torch.nn.CosineEmbeddingLoss()(embeds_1, embeds_2, torch.ones(embeds_1.shape[0]).to(embeds_1.device))