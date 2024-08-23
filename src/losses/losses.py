import torch


def vanilla_constrastive_cross_entrpy_loss(embeds_1, embeds_2, config):
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'mean')
    dotp = torch.nn.functional.cosine_similarity(embeds_1, embeds_2, dim=1)*torch.exp(config['loss_fn']['temperature'])
    dotp = torch.clamp(dotp, max = config['loss_fn']['dotp_clip'])
    labels = torch.ones(dotp.shape[0]).to(embeds_1.device)
    loss = cross_entropy_loss(dotp, labels)
    return loss

def BCE_loss_of_dotps(embeds_1, embeds_2, config):
    bce_loss = torch.nn.BCELoss(reduction = 'mean')
    dotp = torch.nn.functional.cosine_similarity(embeds_1, embeds_2, dim=1)
    labels = torch.ones(dotp.shape[0]).to(embeds_1.device)
    return bce_loss(dotp, labels)

def BCE_loss_of_dotps_ns(embeds_1, embeds_2, embeds_3, config):
    bce_loss = torch.nn.BCELoss(reduction = 'mean')
    dotp_1 = torch.nn.functional.cosine_similarity(embeds_1, embeds_2, dim=1)
    dotp_2 = torch.nn.functional.cosine_similarity(embeds_1, embeds_3, dim=1)
    labels_1 = torch.ones(dotp_1.shape[0]).to(embeds_1.device)
    labels_2 = torch.zeros(dotp_2.shape[0]).to(embeds_1.device)
    return (bce_loss(dotp_1, labels_1) + bce_loss(dotp_2, labels_2))/2.0


def cosine_embedding_loss(embeds_1, embeds_2, config):
    return torch.nn.CosineEmbeddingLoss()(embeds_1, embeds_2, torch.ones(embeds_1.shape[0]).to(embeds_1.device))