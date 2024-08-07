#!/bin/bash
cd datasets
echo "Downloading LaSCO dataset main files from \"https://github.com/levymsn/LaSCo?tab=readme-ov-file#lasco-dataset\""

mkdir LaSCo
cd LaSCo

curl -O https://raw.githubusercontent.com/levymsn/LaSCo/main/downloads/lasco_train.json
curl -O https://raw.githubusercontent.com/levymsn/LaSCo/main/downloads/lasco_train_corpus.json
curl -O https://raw.githubusercontent.com/levymsn/LaSCo/main/downloads/lasco_val.json
curl -O https://raw.githubusercontent.com/levymsn/LaSCo/main/downloads/lasco_val_corpus.json

mkdir coco
cd coco

echo "Downloading COCO images from \"https://cocodataset.org/#home\""

curl -O http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip

curl -O http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

rm train2014.zip
rm val2014.zip

echo "LaSCo dataset download completed"

