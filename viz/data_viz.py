import os
import json
import streamlit as st
from PIL import Image

def load_image(image_path):
    img = Image.open(image_path)
    return img

input_file = 'lasco_train_viz.json'
lasco_data_path = '/local/vondrick/nd2794/CoIR/data/LaSCo'
samples = json.load(open(input_file, 'r'))


if 'sample_idx' not in st.session_state:
    st.session_state.sample_idx = 0
    

def next_sample():
    if st.session_state.sample_idx < len(samples) - 1:
        st.session_state.sample_idx += 1
        
def previous_sample():
    if st.session_state.sample_idx > 0:
        st.session_state.sample_idx -= 1
        
  
current_sample = samples[st.session_state.sample_idx]
source_image = Image.open(os.path.join(lasco_data_path, 'coco', current_sample['query-image']))
target_image = Image.open(os.path.join(lasco_data_path, 'coco', current_sample['target-image']))
query_text = current_sample['query-text']


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
    st.write("**Composed Query**")
    st.write(query_text)

with col3:
    st.image(target_image, caption="Target Image", use_column_width=True)
    



