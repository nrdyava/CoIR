from src.models.clip.inbatch_2en_ST_F import CLIPModel as clip_inbatch_2en_ST_F
from src.models.clip.inbatch_2en_ST_F_WA import CLIPModel as clip_inbatch_2en_ST_F_WA
from src.models.clip.inbatch_2en_ST_F_img_proj import CLIPModel as clip_inbatch_2en_ST_F_img_proj
from src.models.clip.inbatch_2en_ST_F_img_proj_txt_proj import CLIPModel as clip_inbatch_2en_ST_F_img_proj_txt_proj
from src.models.clip.inbatch_2en_ST_F_T_HAT_Proj import CLIPModel as clip_inbatch_2en_ST_F_T_HAT_Proj
from src.models.clip.inbatch_3en_MT_F_img_cap import CLIPModel as clip_inbatch_3en_MT_F_img_cap
from src.models.clip.inbatch_2en_ST_F_ON_GT import CLIPModel as clip_inbatch_2en_ST_F_ON_GT

model_module_registry = {
    'clip_inbatch_2en_ST_F': clip_inbatch_2en_ST_F, 
    'clip_inbatch_2en_ST_F_WA': clip_inbatch_2en_ST_F_WA,
    'clip_inbatch_2en_ST_F_img_proj': clip_inbatch_2en_ST_F_img_proj,
    'clip_inbatch_2en_ST_F_img_proj_txt_proj': clip_inbatch_2en_ST_F_img_proj_txt_proj,
    'clip_inbatch_2en_ST_F_T_HAT_Proj': clip_inbatch_2en_ST_F_T_HAT_Proj,
    'clip_inbatch_3en_MT_F_img_cap': clip_inbatch_3en_MT_F_img_cap,
    'clip_inbatch_2en_ST_F_ON_GT': clip_inbatch_2en_ST_F_ON_GT
}