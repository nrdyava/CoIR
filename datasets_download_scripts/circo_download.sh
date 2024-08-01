#!/bin/bash

echo "Donwloading CIRCO dataset main files from \"https://github.com/miccunifi/CIRCO\""

git clone https://github.com/miccunifi/CIRCO.git

cd CIRCO
mkdir COCO2017_unlabeled
cd COCO2017_unlabeled

echo "Downloading COCO data from \"https://cocodataset.org/#download\""

curl -O http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
unzip image_info_unlabeled2017.zip

curl -O http://images.cocodataset.org/zips/unlabeled2017.zip
unzip unlabeled2017.zip

rm image_info_unlabeled2017.zip
rm unlabeled2017.zip

echo "CIRCO dataset download completed"



