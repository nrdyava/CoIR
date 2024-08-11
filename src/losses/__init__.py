from src.losses.losses import vanilla_constrastive_cross_entrpy_loss, cosine_embedding_loss

loss_fn_registry = {
    'vanilla_constrastive_cross_entrpy_loss': vanilla_constrastive_cross_entrpy_loss,
    'cosine_embedding_loss': cosine_embedding_loss
}