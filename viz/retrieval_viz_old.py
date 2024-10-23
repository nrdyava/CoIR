import os
import json
import streamlit as st
from PIL import Image


input_file = '/proj/vondrick4/naveen/coir-ret-results/CLIP-ViT-B-32/outputs-lasco-val.json'
lasco_data_path = '/local/vondrick/naveen/coir-data/LaSCo'
samples = json.load(open(input_file, 'r'))


def get_image_path(image_id):
    return os.path.join(lasco_data_path, 'coco', 'val2014', 'COCO_val2014_' + str(image_id).zfill(12) + '.jpg')

def load_image(image_path):
    img = Image.open(image_path)
    return img

def get_label(current_sample, rank):
    ret_id = current_sample['top_50_ret_cands'][rank-1]
    t_id = current_sample['target-image-id']
    q_id = current_sample['query-image-id']
    ans = f'Rank-{rank}'
    
    if ret_id == t_id:
        ans += ' (Target)'
    if ret_id == q_id:
        ans += ' (Query)'
    
    return ans
    

if 'sample_idx' not in st.session_state:
    st.session_state.sample_idx = 0
    

def next_sample():
    if st.session_state.sample_idx < len(samples) - 1:
        st.session_state.sample_idx += 1
        
def previous_sample():
    if st.session_state.sample_idx > 0:
        st.session_state.sample_idx -= 1
        
  
current_sample = samples[st.session_state.sample_idx]
source_image = load_image(get_image_path(current_sample['query-image-id']))
target_image = load_image(get_image_path(current_sample['target-image-id']))
query_text = current_sample['query-text-raw']

#source_image = Image.open(os.path.join(lasco_data_path, 'coco', current_sample['query-image']))
#target_image = Image.open(os.path.join(lasco_data_path, 'coco', current_sample['target-image']))
#query_text = current_sample['query-text']


# Display buttons to switch between samples
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.button("Previous", on_click=previous_sample)
with col3:
    st.button("Next", on_click=next_sample)
    

# Display Source Image, Composed Query, and Target Image
st.write("## LaSCo Dataset Visualization")
col1, col2, col3 = st.columns(3)


with col1:
    st.image(source_image, caption="Source Image", use_column_width=True)

with col2:
    st.write("**Query Text:**")
    st.write(query_text)

with col3:
    st.image(target_image, caption="Target Image", use_column_width=True)
    

# Display Source Image, Composed Query, and Target Image
st.write("## LaSCo Retrieval Visualization")
ret1, ret2, ret3, ret4, ret5 = st.columns(5)
ret6, ret7, ret8, ret9, ret10 = st.columns(5)

with ret1:
    st.image(load_image(get_image_path(current_sample['top_50_ret_cands'][0])), caption=get_label(current_sample, 1), use_column_width=True)

with ret2:
    st.image(load_image(get_image_path(current_sample['top_50_ret_cands'][1])), caption=get_label(current_sample, 2), use_column_width=True)

with ret3:
    st.image(load_image(get_image_path(current_sample['top_50_ret_cands'][2])), caption=get_label(current_sample, 3), use_column_width=True)
    
with ret4:
    st.image(load_image(get_image_path(current_sample['top_50_ret_cands'][3])), caption=get_label(current_sample, 4), use_column_width=True)

with ret5:
    st.image(load_image(get_image_path(current_sample['top_50_ret_cands'][4])), caption=get_label(current_sample, 5), use_column_width=True)
    
with ret6:
    st.image(load_image(get_image_path(current_sample['top_50_ret_cands'][5])), caption=get_label(current_sample, 6), use_column_width=True)

with ret7:
    st.image(load_image(get_image_path(current_sample['top_50_ret_cands'][6])), caption=get_label(current_sample, 7), use_column_width=True)

with ret8:
    st.image(load_image(get_image_path(current_sample['top_50_ret_cands'][7])), caption=get_label(current_sample, 8), use_column_width=True)
    
with ret9:
    st.image(load_image(get_image_path(current_sample['top_50_ret_cands'][8])), caption=get_label(current_sample, 9), use_column_width=True)

with ret10:
    st.image(load_image(get_image_path(current_sample['top_50_ret_cands'][9])), caption=get_label(current_sample, 10), use_column_width=True)
    



