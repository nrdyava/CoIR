from src.models.clip.clip import CLIPModel
from src.models.clip.clip_ns import CLIPModelNS
from src.models.clip.clip_inbatch import CLIPModelINBATCH
from src.models.clip.clip_inbatch_mt import CLIPModelINBATCH_MT

model_registry = {
    'clip': CLIPModel,
    'clip_ns': CLIPModelNS,
    'clip_inbatch': CLIPModelINBATCH,
    'clip_inbatch_mt': CLIPModelINBATCH_MT
}