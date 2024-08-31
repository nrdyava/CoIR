import os
import torch
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
from tqdm import tqdm

lasco_path = '/local/vondrick/nd2794/CoIR/data/LaSCo'
output_path = os.path.join('/local/vondrick/nd2794/CoIR/data', 'LaSCo_img_tensors')

image_processor = AutoProcessor.from_pretrained('/local/vondrick/nd2794/pretrained_models/clip/clip-vit-base-patch32')

os.mkdir(output_path)
os.makedirs(os.path.join(output_path, 'coco', 'train2014'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'coco', 'val2014'), exist_ok=True)

train_imgs_list = [f for f in os.listdir(os.path.join(lasco_path, 'coco', 'train2014')) if f.endswith('.jpg')]
val_imgs_list = [f for f in os.listdir(os.path.join(lasco_path, 'coco', 'val2014')) if f.endswith('.jpg')]

train_imgs_out_path = os.path.join(output_path, 'coco', 'train2014')
for img in tqdm(train_imgs_list):
    img_path = os.path.join(lasco_path, 'coco', 'train2014', img)
    tensor_path = os.path.join(train_imgs_out_path, img.replace('.jpg', '.pt'))
    
    image = Image.open(img_path)
    img_tensors = image_processor(images = image, return_tensors='pt')
    torch.save(img_tensors, tensor_path)

val_imgs_out_path = os.path.join(output_path, 'coco', 'val2014')
for img in tqdm(val_imgs_list):
    img_path = os.path.join(lasco_path, 'coco', 'val2014', img)
    tensor_path = os.path.join(val_imgs_out_path, img.replace('.jpg', '.pt'))
    
    image = Image.open(img_path)
    img_tensors = image_processor(images = image, return_tensors='pt')
    torch.save(img_tensors, tensor_path)









