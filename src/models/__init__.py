from src.models.clip.inbatch_2en_ST_F import CLIPModel as clip_inbatch_2en_ST_F
from src.models.clip.inbatch_2en_ST_F_WA import CLIPModel as clip_inbatch_2en_ST_F_WA
from src.models.clip.inbatch_2en_ST_F_img_proj import CLIPModel as clip_inbatch_2en_ST_F_img_proj

model_module_registry = {
    'clip_inbatch_2en_ST_F': clip_inbatch_2en_ST_F, 
    'clip_inbatch_2en_ST_F_WA': clip_inbatch_2en_ST_F_WA,
    'clip_inbatch_2en_ST_F_img_proj': clip_inbatch_2en_ST_F_img_proj
}