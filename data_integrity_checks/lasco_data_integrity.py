"""
Checks:

1. Number of Triplets - Train & Validation data [Done]
2. Corpus Size - Train & Validation Data [Done]
3. Check for existence of fields for all samples in lasco_train.json & lasco_val.json - ["qid", "query-image", "query-text", "target-image"] [Done]
4. Query images and target images check for existence in coco - for lasco_train.json and lasco_val.json [Done]
5. Check for existence of corpus images in coco - Train &  Val corpus [Done]
6. Check for existence of target images in train and val corpus respectively [Done]
"""



import os
import json
from tabulate import tabulate




cwd = os.getcwd()
print("Current directory: {}".format(cwd))
lasco_dir = os.path.join(os.path.dirname(cwd), "datasets", "LaSCo")
print("LaSCo dataset directory: {}".format(lasco_dir))




with open(os.path.join(lasco_dir, "lasco_train.json"), 'r') as file:
    lasco_train = json.load(file)

with open(os.path.join(lasco_dir, "lasco_val.json"), 'r') as file:
    lasco_val = json.load(file)

with open(os.path.join(lasco_dir, "lasco_train_corpus.json"), 'r') as file:
    lasco_train_corpus = json.load(file)

with open(os.path.join(lasco_dir, "lasco_val_corpus.json"), 'r') as file:
    lasco_val_corpus = json.load(file)




## Test - 1
print("\n"+"-"*100+"\n")
print("Checking number of triplets:")
table = [
    ["lasco-train-triplets", len(lasco_train)],
    ["lasco-val-triplets", len(lasco_val)],
    ["total-triplets (train+val)", len(lasco_train) + len(lasco_val)]
]
print(tabulate(table, tablefmt="simple_grid"))
print("\n"+"-"*100+"\n")



## Test - 2
print("\n"+"-"*100+"\n")
print("Checking corpus size:")
table = [
    ["lasco-train-corpus-size", len(lasco_train_corpus)],
    ["lasco-val-corpus-size", len(lasco_val_corpus)]
]
print(tabulate(table, tablefmt="simple_grid"))
print("\n"+"-"*100+"\n")



## Test - 3
print("\n"+"-"*100+"\n")
print("Checking for consistency of data fields: ")

f2c = ['qid', 'query-image', 'query-text', 'target-image']
print("Fields to check: {}".format(f2c))

print("=> Checking lasco_train.json")
for sample in lasco_train:
    temp = list(sample.keys())
    if 'qid' in temp and 'query-image' in temp and 'query-text' in temp and 'target-image' in temp:
        continue
    else:
        print("Data fields consistency check failed for :\n{}".format(sample))
print("   lasco_train.json is consistent")

print("=> Checking lasco_val.json")
for sample in lasco_val:
    temp = list(sample.keys())
    if 'qid' in temp and 'query-image' in temp and 'query-text' in temp and 'target-image' in temp:
        continue
    else:
        print("Data fields consistency check failed for :\n{}".format(sample))
print("   lasco_val.json is consistent")
print("\n"+"-"*100+"\n")




coco_train2014_list = set(os.listdir(os.path.join(lasco_dir, "coco/train2014")))
coco_val2014_list = set(os.listdir(os.path.join(lasco_dir, "coco/val2014")))



## Test - 4
print("\n"+"-"*100+"\n")
print("=> Checking if lasco train - (query and target) images are in coco/train2014:")
temp = 0
for record in lasco_train:
    if (record['query-image'][1].split('/')[1] in coco_train2014_list and record['target-image'][1].split('/')[1] in coco_train2014_list):
        continue
    else:
        print("   Missing Record qid: {}".format(record['qid']))
        temp = 1
if (temp == 0):
    print("   Check is successful\n")
else:
    print("   Check failed\n")



print("=> Checking if lasco val - (query and target) images are in coco/val2014:")
temp = 0
for record in lasco_val:
    if (record['query-image'][1].split('/')[1] in coco_val2014_list and record['target-image'][1].split('/')[1] in coco_val2014_list):
        continue
    else:
        print("Missing Record qid: {}".format(record['qid']))
if (temp == 0):
    print("   Check is successful")
else:
    print("   Check failed")



print("\n"+"-"*100+"\n")



## Test - 5
print("\n"+"-"*100+"\n")
print("=> Checking if lasco train corpus - images are in coco/train2014:")
temp = 0
for record in list(map(lambda x: x.split('/')[-1], list(lasco_train_corpus.values()))):
    if record in coco_train2014_list:
        continue
    else:
        print("   Missing Record: {}".format(record))
        temp = 1
        
if (temp == 0):
    print("   Check is successful\n")
else:
    print("   Check failed\n")



print("=> Checking if lasco val corpus - images are in coco/val2014:")
temp = 0
for record in list(map(lambda x: x['path'].split('/')[-1], lasco_val_corpus)):
    if record in coco_val2014_list:
        continue
    else:
        print("Missing Record: {}".format(record['qid']))
        temp = 1
        
if (temp == 0):
    print("   Check is successful")
else:
    print("   Check failed")



print("\n"+"-"*100+"\n")





lasco_train_corpus_imgs_list = set(list(map(lambda x: x.split('/')[-1], list(lasco_train_corpus.values()))))
lasco_val_corpus_imgs_list = set(list(map(lambda x: x['path'].split('/')[-1], lasco_val_corpus)))


## Test - 6
print("\n"+"-"*100+"\n")
print("=> Checking if lasco train-target-images are in lasco-train-corpus:")
temp = 0
for sample in list(map(lambda x: x['target-image'][1].split('/')[-1], lasco_train)):
    if sample in lasco_train_corpus_imgs_list:
        continue
    else:
        temp = 1
        print("   Missing Record: {}".format(record))

if (temp == 0):
    print("   Check is successful\n")
else:
    print("   Check failed\n")

print("=> Checking if lasco val-target-images are in lasco-val-corpus:")
temp = 0
for sample in list(map(lambda x: x['target-image'][1].split('/')[-1], lasco_val)):
    if sample in lasco_val_corpus_imgs_list:
        continue
    else:
        temp = 1
        print("   Missing Record: {}".format(record))

if (temp == 0):
    print("   Check is successful")
else:
    print("   Check failed")

print("\n"+"-"*100+"\n")