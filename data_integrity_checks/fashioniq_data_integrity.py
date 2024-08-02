"""
Checks:

Categories: [Dresses, Shirts, Top&Tees]
Dataset splits: [Train, Val, Test]

1. Basis stats (To compare with the numbers menitoned in the dataset paper) [Done]
    a. Number of images for each Category X Dataset splits
    b. Number of queries given in each Category X Dataset Split
    c. Number of images in images folder 
2. Data fields consistency checks [Done]
    a. Train, Val & Test data splits 
3. Triangular Consistency checks at Category X Dataset Split level [Done]
    a. Query (candidate, target) images in images
    b. Query (candidate, target) images in image_splits
    c. image_splits images in images folder

"""


import os
import json
from tabulate import tabulate



cwd = os.getcwd()
print("Current directory: {}".format(cwd))
fiq_dir = os.path.join(os.path.dirname(cwd), "datasets", "FashionIQ")
print("FashionIQ dataset directory: {}".format(fiq_dir))



images_dir_list = os.listdir(os.path.join(fiq_dir, "images"))
images_dir_list = list(map(lambda x: x.split('.')[0], images_dir_list))
images_dir_set = set(images_dir_list)

with open(os.path.join(fiq_dir, "captions", "cap.dress.test.json"), 'r') as file:
    cap_dress_test = json.load(file)
with open(os.path.join(fiq_dir, "captions", "cap.dress.train.json"), 'r') as file:
    cap_dress_train = json.load(file)
with open(os.path.join(fiq_dir, "captions", "cap.dress.val.json"), 'r') as file:
    cap_dress_val = json.load(file)
with open(os.path.join(fiq_dir, "captions", "cap.shirt.test.json"), 'r') as file:
    cap_shirt_test = json.load(file)
with open(os.path.join(fiq_dir, "captions", "cap.shirt.train.json"), 'r') as file:
    cap_shirt_train = json.load(file)
with open(os.path.join(fiq_dir, "captions", "cap.shirt.val.json"), 'r') as file:
    cap_shirt_val = json.load(file)
with open(os.path.join(fiq_dir, "captions", "cap.toptee.test.json"), 'r') as file:
    cap_toptee_test = json.load(file)
with open(os.path.join(fiq_dir, "captions", "cap.toptee.train.json"), 'r') as file:
    cap_toptee_train = json.load(file)
with open(os.path.join(fiq_dir, "captions", "cap.toptee.val.json"), 'r') as file:
    cap_toptee_val = json.load(file)

with open(os.path.join(fiq_dir, "image_splits", "split.dress.test.json"), 'r') as file:
    split_dress_test = json.load(file)
with open(os.path.join(fiq_dir, "image_splits", "split.dress.train.json"), 'r') as file:
    split_dress_train = json.load(file)
with open(os.path.join(fiq_dir, "image_splits", "split.dress.val.json"), 'r') as file:
    split_dress_val = json.load(file)
with open(os.path.join(fiq_dir, "image_splits", "split.shirt.test.json"), 'r') as file:
    split_shirt_test = json.load(file)
with open(os.path.join(fiq_dir, "image_splits", "split.shirt.train.json"), 'r') as file:
    split_shirt_train = json.load(file)
with open(os.path.join(fiq_dir, "image_splits", "split.shirt.val.json"), 'r') as file:
    split_shirt_val = json.load(file)
with open(os.path.join(fiq_dir, "image_splits", "split.toptee.test.json"), 'r') as file:
    split_toptee_test = json.load(file)
with open(os.path.join(fiq_dir, "image_splits", "split.toptee.train.json"), 'r') as file:
    split_toptee_train = json.load(file)
with open(os.path.join(fiq_dir, "image_splits", "split.toptee.val.json"), 'r') as file:
    split_toptee_val = json.load(file)



#Test - 1
print("\n"+"-"*100+"\n")
print("Number of images in the images folder: {}".format(len(images_dir_list)))
print("\n"+"-"*100+"\n")



print("\n"+"-"*100+"\n")
print("Checking number of images (Category X Split):")
print("=> Dresses:")
table = [
    ["Train", len(split_dress_train)],
    ["Val", len(split_dress_val)],
    ["Test", len(split_dress_test)],
    ["Total", len(split_dress_train)+len(split_dress_val)+len(split_dress_test)]
]
print(tabulate(table, tablefmt="simple_grid"))

print("=> Shirts:")
table = [
    ["Train", len(split_shirt_train)],
    ["Val", len(split_shirt_val)],
    ["Test", len(split_shirt_test)],
    ["Total", len(split_shirt_train)+len(split_shirt_val)+len(split_shirt_test)]
]
print(tabulate(table, tablefmt="simple_grid"))

print("=> Tops&Tees:")
table = [
    ["Train", len(split_toptee_train)],
    ["Val", len(split_toptee_val)],
    ["Test", len(split_toptee_test)],
    ["Total", len(split_toptee_train)+len(split_toptee_val)+len(split_toptee_test)]
]
print(tabulate(table, tablefmt="simple_grid"))

print("\n"+"-"*100+"\n")



print("\n"+"-"*100+"\n")
print("Checking number of Queries (Category X Split):")
print("=> Dresses:")
table = [
    ["Train", len(cap_dress_train)],
    ["Val", len(cap_dress_val)],
    ["Test", len(cap_dress_test)],
    ["Total", len(cap_dress_train)+len(cap_dress_val)+len(cap_dress_test)]
]
print(tabulate(table, tablefmt="simple_grid"))

print("=> Shirts:")
table = [
    ["Train", len(cap_shirt_train)],
    ["Val", len(cap_shirt_val)],
    ["Test", len(cap_shirt_test)],
    ["Total", len(cap_shirt_train)+len(cap_shirt_val)+len(cap_shirt_test)]
]
print(tabulate(table, tablefmt="simple_grid"))

print("=> Tops&Tees:")
table = [
    ["Train", len(cap_toptee_train)],
    ["Val", len(cap_toptee_val)],
    ["Test", len(cap_toptee_test)],
    ["Total", len(cap_toptee_train)+len(cap_toptee_val)+len(cap_toptee_test)]
]
print(tabulate(table, tablefmt="simple_grid"))

print("\n"+"-"*100+"\n")


#Test - 2
print("\n"+"-"*100+"\n")
print("Checking data fields Consitency:")


print("=> Dresses:")

print("  +Train:")
temp = 0
for sample in cap_dress_train:
    li = list(sample.keys())
    if 'candidate' in li and 'target' in li and 'captions' in li:
        continue
    else:
        temp = 1
        print("      Consistency Violated:\n {}".format(record))
if temp == 0:
    print("      Data fields are consistent")
else:
    print("      Data fields are inconsistent")

print("  +Val:")
temp = 0
for sample in cap_dress_val:
    li = list(sample.keys())
    if 'candidate' in li and 'target' in li and 'captions' in li:
        continue
    else:
        temp = 1
        print("      Consistency Violated:\n {}".format(record))
if temp == 0:
    print("      Data fields are consistent")
else:
    print("      Data fields are inconsistent")

print("  +Test:")
temp = 0
for sample in cap_dress_test:
    li = list(sample.keys())
    if 'candidate' in li and 'captions' in li:
        continue
    else:
        temp = 1
        print("      Consistency Violated:\n {}".format(record))
if temp == 0:
    print("      Data fields are consistent")
else:
    print("      Data fields are inconsistent")
    


print("=> Shirt:")

print("  +Train:")
temp = 0
for sample in cap_shirt_train:
    li = list(sample.keys())
    if 'candidate' in li and 'target' in li and 'captions' in li:
        continue
    else:
        temp = 1
        print("      Consistency Violated:\n {}".format(record))
if temp == 0:
    print("      Data fields are consistent")
else:
    print("      Data fields are inconsistent")

print("  +Val:")
temp = 0
for sample in cap_shirt_val:
    li = list(sample.keys())
    if 'candidate' in li and 'target' in li and 'captions' in li:
        continue
    else:
        temp = 1
        print("      Consistency Violated:\n {}".format(record))
if temp == 0:
    print("      Data fields are consistent")
else:
    print("      Data fields are inconsistent")

print("  +Test:")
temp = 0
for sample in cap_shirt_test:
    li = list(sample.keys())
    if 'candidate' in li and 'captions' in li:
        continue
    else:
        temp = 1
        print("      Consistency Violated:\n {}".format(record))
if temp == 0:
    print("      Data fields are consistent")
else:
    print("      Data fields are inconsistent")

print("=> Tops&Tees:")

print("  +Train:")
temp = 0
for sample in cap_toptee_train:
    li = list(sample.keys())
    if 'candidate' in li and 'target' in li and 'captions' in li:
        continue
    else:
        temp = 1
        print("      Consistency Violated:\n {}".format(record))
if temp == 0:
    print("      Data fields are consistent")
else:
    print("      Data fields are inconsistent")

print("  +Val:")
temp = 0
for sample in cap_toptee_val:
    li = list(sample.keys())
    if 'candidate' in li and 'target' in li and 'captions' in li:
        continue
    else:
        temp = 1
        print("      Consistency Violated:\n {}".format(record))
if temp == 0:
    print("      Data fields are consistent")
else:
    print("      Data fields are inconsistent")

print("  +Test:")
temp = 0
for sample in cap_toptee_test:
    li = list(sample.keys())
    if 'candidate' in li and 'captions' in li:
        continue
    else:
        temp = 1
        print("      Consistency Violated:\n {}".format(record))
if temp == 0:
    print("      Data fields are consistent")
else:
    print("      Data fields are inconsistent")

print("\n"+"-"*100+"\n")



#Test - 3
print("\n"+"-"*100+"\n")
print("=> Checking Triangular Consistency - 1:")
img_chk_list = []
img_chk_list.extend(list(map(lambda x: x['target'], cap_dress_train)))
img_chk_list.extend(list(map(lambda x: x['candidate'], cap_dress_train)))
img_chk_list.extend(list(map(lambda x: x['target'], cap_dress_val)))
img_chk_list.extend(list(map(lambda x: x['candidate'], cap_dress_val)))
img_chk_list.extend(list(map(lambda x: x['candidate'], cap_dress_test)))

img_chk_list.extend(list(map(lambda x: x['target'], cap_shirt_train)))
img_chk_list.extend(list(map(lambda x: x['candidate'], cap_shirt_train)))
img_chk_list.extend(list(map(lambda x: x['target'], cap_shirt_val)))
img_chk_list.extend(list(map(lambda x: x['candidate'], cap_shirt_val)))
img_chk_list.extend(list(map(lambda x: x['candidate'], cap_shirt_test)))

img_chk_list.extend(list(map(lambda x: x['target'], cap_toptee_train)))
img_chk_list.extend(list(map(lambda x: x['candidate'], cap_toptee_train)))
img_chk_list.extend(list(map(lambda x: x['target'], cap_toptee_val)))
img_chk_list.extend(list(map(lambda x: x['candidate'], cap_toptee_val)))
img_chk_list.extend(list(map(lambda x: x['candidate'], cap_toptee_test)))

temp = 0
for record in img_chk_list:
    if record in images_dir_set:
        continue
    else:
        temp = 1
        break

if temp == 1:
    print("   Failed Triangular Consistency - 1:")
else:
    print("   Passed Triangular Consistency - 1:")
print("\n"+"-"*100+"\n")




print("\n"+"-"*100+"\n")
print("=> Checking Triangular Consistency - 2:")
img_chk_list = []
img_chk_list.extend(split_dress_train)
img_chk_list.extend(split_dress_val)
img_chk_list.extend(split_dress_test)

img_chk_list.extend(split_shirt_train)
img_chk_list.extend(split_shirt_val)
img_chk_list.extend(split_shirt_test)

img_chk_list.extend(split_toptee_train)
img_chk_list.extend(split_toptee_val)
img_chk_list.extend(split_toptee_test)

temp = 0
for record in img_chk_list:
    if record in images_dir_set:
        continue
    else:
        temp = 1
        break

if temp == 1:
    print("   Failed Triangular Consistency - 2:")
else:
    print("   Passed Triangular Consistency - 2:")
print("\n"+"-"*100+"\n")




print("\n"+"-"*100+"\n")
print("=> Checking Triangular Consistency - 3:")
temp = 0
for sample in list(map(lambda x: x['target'], cap_dress_train)):
    if sample in split_dress_train:
        continue
    else:
        temp = 1
for sample in list(map(lambda x: x['target'], cap_dress_val)):
    if sample in split_dress_val:
        continue
    else:
        temp = 1

for sample in list(map(lambda x: x['target'], cap_shirt_train)):
    if sample in split_shirt_train:
        continue
    else:
        temp = 1
for sample in list(map(lambda x: x['target'], cap_shirt_val)):
    if sample in split_shirt_val:
        continue
    else:
        temp = 1

for sample in list(map(lambda x: x['target'], cap_toptee_train)):
    if sample in split_toptee_train:
        continue
    else:
        temp = 1
for sample in list(map(lambda x: x['target'], cap_toptee_val)):
    if sample in split_toptee_val:
        continue
    else:
        temp = 1

if temp == 0:
    print("   Passed Triangular Consistency - 3:")
else:
    print("   Failed Triangular Consistency - 3:")

print("\n"+"-"*100+"\n")
