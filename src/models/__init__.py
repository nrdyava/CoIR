from src.models.clip.clip import CLIPModel
from src.models.clip.clip_ns import CLIPModelNS

model_registry = {
    'clip': CLIPModel,
    'clip_ns': CLIPModelNS
}