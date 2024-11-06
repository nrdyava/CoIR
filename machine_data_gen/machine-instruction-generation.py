#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


from vllm import LLM, SamplingParams
import tqdm as notebook_tqdm
import json
import os
import re
import copy
from huggingface_hub import login

with open('/home/nd2794/keys.json', 'r') as file:
    keys = json.load(file)

cache_dir = '/proj/vondrick4/naveen/HF_CACHE_DIR'
hf_token = keys["HF_READ_TOKEN"]
login(token=hf_token)

os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['PYTORCH_TRANSFORMERS_CACHE'] = cache_dir
os.environ['PYTORCH_PRETRAINED_BERT_CACHE'] = cache_dir
os.environ['HF_HUB_CACHE'] = cache_dir


# ## Define Utility Functions

# In[2]:


def get_instruction(source_event, target_event):
    common_instruct = """Two images (Source & Target) are described by their captions.
Your have generate a query text for composed image retrieval task that describes how the Source image should be modified to get the Target image.
Each image may multiple captions which are listed in a new line. You must read all the captions for each image to fully understand the image.
Only output the query text."""

    source_instruct = """SOURCE IMAGE CAPTIONS:\n""" + "\n".join(f"- {item}" for item in source_event)

    target_instruct = """TARGET IMAGE CAPTIONS:\n""" + "\n".join(f"- {item}" for item in target_event)

    return common_instruct + '\n\n' + source_instruct + '\n\n' + target_instruct


# In[3]:


ex_source_cap = [
    'A person in blue and orange shirt riding a bicycle.',
    'A bicyclist riding in front of speed blurred foliage. ',
    'A cyclist with a helmet bicycles quickly on a road.',
    'The man is riding his bike down the street. '
]

ex_target_cap = [
    'A man is riding a motorcycle down the street',
    'A man riding a motorcycle down a street next to a restaurant.',
    'Helmeted motorcyclist riding on roadway in populated setting. ',
    'A motorcycle going down a street very fast.'
]

ex_query = "The person is riding a motorcycle"


# In[4]:


conversation_template = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": get_instruction(ex_source_cap, ex_target_cap)
    },
    {
        "role": "assistant",
        "content": ex_query 
    }
]


# ## Load Model

# In[5]:


model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
llm = LLM(
    model=model_name, 
    tensor_parallel_size=1, 
    download_dir=cache_dir,
    dtype='bfloat16'
)


# In[7]:


sampling_params = SamplingParams(
    #temperature=1.0,
    max_tokens=200,
)


# ## Load the data

# In[44]:


split = 'val'
if split == 'train':
    data = json.load(open('/proj/vondrick4/naveen/coir-data/LaSCo/metadata/lasco_train_indexed_MT(img_cap).json', 'r'))
    imgs_and_caps = json.load(open('/proj/vondrick4/naveen/coir-data/annotations/images_and_caps_train2014_processed_dict.json', 'r'))
elif split == 'val':
    data = json.load(open('/proj/vondrick4/naveen/coir-data/LaSCo/metadata/lasco_val_indexed_MT(img_cap).json', 'r'))
    imgs_and_caps= json.load(open('/proj/vondrick4/naveen/coir-data/annotations/images_and_caps_val2014_processed_dict.json', 'r'))
    for key in imgs_and_caps.keys():
        imgs_and_caps[key] = [imgs_and_caps[key]]


# ## Process the data

# In[45]:


conversations = []
tracker = {}
i = 0

for sample in data:
    tracker[sample['id']] = {}
    
    conversation_forward = copy.deepcopy(conversation_template)
    conversation_forward.append(
        {
            "role": "user",
            "content": get_instruction(imgs_and_caps[str(sample["coir"]["query-image-id"])], imgs_and_caps[str(sample["coir"]["target-image-id"])])
        }
    )
    
    conversation_reverse = copy.deepcopy(conversation_template)
    conversation_reverse.append(
        {
            "role": "user",
            "content": get_instruction(imgs_and_caps[str(sample["coir"]["target-image-id"])], imgs_and_caps[str(sample["coir"]["query-image-id"])])
        }
    )
    
    conversations.append(conversation_forward)
    tracker[sample['id']]['forward'] = i
    i+=1
    
    conversations.append(conversation_reverse)
    tracker[sample['id']]['reverse'] = i
    i+=1


# ## Run vLLM

# In[46]:


outputs = llm.chat(
    messages=conversations,
    sampling_params=sampling_params,
    use_tqdm=True
)


# ## Process & Write Outputs

# In[47]:


processed_outputs = []
for out in outputs:
    processed_outputs.append(out.outputs[0].text)


# In[48]:


for sample in data:
    sample["coir"]['query-text-forward-mg'] = processed_outputs[tracker[sample['id']]['forward']]
    sample["coir"]['query-text-reverse-mg'] = processed_outputs[tracker[sample['id']]['reverse']]


# In[49]:


with open('/proj/vondrick4/naveen/coir-data/LaSCo/metadata/lasco_{}_indexed_MT(img_cap)_MG_CV14.json'.format(split), 'w') as json_file:
    json.dump(data, json_file, indent=4)

