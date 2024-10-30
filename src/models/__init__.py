from src.models.clip.inbatch_2en_ST_F import CLIPModel as clip_inbatch_2en_ST_F
from src.models.clip.inbatch_2en_ST_F_WA import CLIPModel as clip_inbatch_2en_ST_F_WA
from src.models.clip.inbatch_2en_ST_F_img_proj import CLIPModel as clip_inbatch_2en_ST_F_img_proj
from src.models.clip.inbatch_2en_ST_F_img_proj_txt_proj import CLIPModel as clip_inbatch_2en_ST_F_img_proj_txt_proj
from src.models.clip.inbatch_2en_ST_F_T_HAT_Proj import CLIPModel as clip_inbatch_2en_ST_F_T_HAT_Proj
from src.models.clip.inbatch_3en_MT_F_img_cap import CLIPModel as clip_inbatch_3en_MT_F_img_cap
from src.models.clip.inbatch_2en_ST_F_ON_GT import CLIPModel as clip_inbatch_2en_ST_F_ON_GT
from src.models.flava.inbatch_2en_ST_F_ON_GT import FLAVA_Model as flava_inbatch_2en_ST_F_ON_GT
from src.models.clip.inbatch_2en_ST_F_ON_GT_QN import CLIPModel as clip_inbatch_2en_ST_F_ON_GT_QN
from src.models.clip.inbatch_2en_ST_F_ON_GT_QN_WA import CLIPModel as clip_inbatch_2en_ST_F_ON_GT_QN_WA
from src.models.clip.inbatch_2en_MT_FR1_ON_GT_QN import CLIPModel as clip_inbatch_2en_MT_FR1_ON_GT_QN

model_module_registry = {
    'clip_inbatch_2en_ST_F': clip_inbatch_2en_ST_F, 
    'clip_inbatch_2en_ST_F_WA': clip_inbatch_2en_ST_F_WA,
    'clip_inbatch_2en_ST_F_img_proj': clip_inbatch_2en_ST_F_img_proj,
    'clip_inbatch_2en_ST_F_img_proj_txt_proj': clip_inbatch_2en_ST_F_img_proj_txt_proj,
    'clip_inbatch_2en_ST_F_T_HAT_Proj': clip_inbatch_2en_ST_F_T_HAT_Proj,
    'clip_inbatch_3en_MT_F_img_cap': clip_inbatch_3en_MT_F_img_cap,
    'clip_inbatch_2en_ST_F_ON_GT': clip_inbatch_2en_ST_F_ON_GT,
    'flava_inbatch_2en_ST_F_ON_GT': flava_inbatch_2en_ST_F_ON_GT,
    'clip_inbatch_2en_ST_F_ON_GT_QN': clip_inbatch_2en_ST_F_ON_GT_QN,
    'clip_inbatch_2en_ST_F_ON_GT_QN_WA': clip_inbatch_2en_ST_F_ON_GT_QN_WA,
    'clip_inbatch_2en_MT_FR1_ON_GT_QN': clip_inbatch_2en_MT_FR1_ON_GT_QN
}