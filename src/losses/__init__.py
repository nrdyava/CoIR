from src.losses import (
    contrastive_loss_with_softmax
)

loss_fn_registry = {
    'contrastive_loss_with_softmax_temp_rank': contrastive_loss_with_softmax.loss_with_temp_rank,
    'contrastive_loss_with_softmax_temp': contrastive_loss_with_softmax.loss_with_temp,
    'contrastive_asymetric_loss_with_softmax_temp': contrastive_loss_with_softmax.asymetric_loss_with_temp,
    'contrastive_asymetric_loss_with_softmax_temp_QN1': contrastive_loss_with_softmax.asymetric_loss_with_temp_QN1
}