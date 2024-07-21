#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
from tabulate import tabulate


# In[ ]:





# In[2]:


root_dir = os.path.dirname(os.getcwd())
print("root directory: {}".format(root_dir))


# In[ ]:





# In[4]:


with open(os.path.join(root_dir, "datasets/LaSCo/lasco_train.json"), 'r') as file:
    lasco_train = json.load(file)
with open(os.path.join(root_dir, "datasets/LaSCo/lasco_val.json"), 'r') as file:
    lasco_val = json.load(file)
with open(os.path.join(root_dir, "datasets/LaSCo/lasco_train_corpus.json"), 'r') as file:
    lasco_train_corpus = json.load(file)
with open(os.path.join(root_dir, "datasets/LaSCo/lasco_val_corpus.json"), 'r') as file:
    lasco_val_corpus = json.load(file)


# In[5]:


coco_train2014_list = set(os.listdir(os.path.join(root_dir, "datasets/LaSCo/coco/train2014")))
coco_val2014_list = set(os.listdir(os.path.join(root_dir, "datasets/LaSCo/coco/val2014")))


# In[ ]:





# In[6]:


print("\n"+"-"*100+"\n")
table = [
    ["lasco_train", len(lasco_train)],
    ["lasco_val", len(lasco_val)],
    ["lasco_train_corpus", len(lasco_train_corpus)],
    ["lasco_val_corpus", len(lasco_val_corpus)]
]
print("Number of records in LaSCo dataset by split:")
print(tabulate(table, tablefmt="simple_grid"))
print("\n"+"-"*100+"\n")


# In[7]:


print("\n"+"-"*100+"\n")
table = [
    ["coco_train2014", len(coco_train2014_list)],
    ["coco_val2014", len(coco_val2014_list)]
]
print("Number of records in COCO dataset by split:")
print(tabulate(table, tablefmt="simple_grid"))
print("\n"+"-"*100+"\n")


# In[ ]:





# In[8]:


for record in lasco_train:
    if (record['query-image'][1].split('/')[1] in coco_train2014_list and record['target-image'][1].split('/')[1] in coco_train2014_list):
        continue
    else:
        print("Missing Record qid: {}".format(record['qid']))


# In[9]:


for record in lasco_val:
    if (record['query-image'][1].split('/')[1] in coco_val2014_list and record['target-image'][1].split('/')[1] in coco_val2014_list):
        continue
    else:
        print("Missing Record qid: {}".format(record['qid']))


# In[10]:


for record in list(lasco_train_corpus.values()):
    if (record.split('/')[1] in coco_train2014_list):
        continue
    else:
        print("Missing record: {}".format(record))


# In[11]:


for record in list(lasco_val_corpus):
    if (record['path'].split('/')[2] in coco_val2014_list):
        continue
    else:
        print("Missing record: {}".format(record['id']))


# In[ ]:





# In[12]:


print("\n"+"-"*100+"\n")
print("All tests concluded!")
print("\n"+"-"*100+"\n")

