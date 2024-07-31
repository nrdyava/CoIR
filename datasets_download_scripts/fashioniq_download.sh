#!/bin/bash
echo "Downloading FashionIQ Dataset from \"https://huggingface.co/datasets/Plachta/FashionIQ\""
wget https://huggingface.co/datasets/Plachta/FashionIQ/resolve/main/fashionIQ_dataset.rar?download=true -O FashionIQ.rar
unrar x FashionIQ.rar
mv fashionIQ_dataset FashionIQ
rm FashionIQ.rar
