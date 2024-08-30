from src.losses.losses import (
    BCE_loss_of_dotps,
    vanilla_constrastive_cross_entrpy_loss,
    cosine_embedding_loss,
    BCE_loss_of_dotps_ns,
    symmetric_loss_without_temp,
    symmetric_loss_with_temp
)

loss_fn_registry = {
    'vanilla_constrastive_cross_entrpy_loss': vanilla_constrastive_cross_entrpy_loss,
    'cosine_embedding_loss': cosine_embedding_loss,
    'BCE_loss_of_dotps': BCE_loss_of_dotps,
    'BCE_loss_of_dotps_ns': BCE_loss_of_dotps_ns,
    'symmetric_loss_without_temp':symmetric_loss_without_temp,
    'symmetric_loss_with_temp':symmetric_loss_with_temp
}