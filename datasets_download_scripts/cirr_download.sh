#!/bin/bash
cd datasets
echo "Downloading CIRR main files from \"https://github.com/Cuberick-Orion/CIRR\""

git clone -b cirr_dataset https://github.com/Cuberick-Orion/CIRR.git CIRR
cd CIRR

mkdir img_raw
cd img_raw

echo "Downloading raw images from NLVR2 Dataset from \"https://lil.nlp.cornell.edu/resources/NLVR2/\""

curl -O https://lil.nlp.cornell.edu/resources/NLVR2/train_img.zip
curl -O https://lil.nlp.cornell.edu/resources/NLVR2/dev_img.zip
curl -O https://lil.nlp.cornell.edu/resources/NLVR2/test1_img.zip
curl -O https://lil.nlp.cornell.edu/resources/NLVR2/test2.zip

unzip train_img.zip
unzip dev_img.zip
unzip test1_img.zip
unzip test2.zip

rm train_img.zip
rm dev_img.zip
rm test1_img.zip
rm test2.zip

mv images/train .
rm -rf images

echo "CIRR dataset download completed"
