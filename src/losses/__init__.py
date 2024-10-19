from src.losses import (
    contrastive_loss_with_softmax
)

loss_fn_registry = {
    'contrastive_loss_with_softmax_temp_rank': contrastive_loss_with_softmax.loss_with_temp_rank,
    'contrastive_loss_with_softmax_temp': contrastive_loss_with_softmax.loss_with_temp
}