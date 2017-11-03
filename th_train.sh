#!/bin/bash

# export PATH=/homes/grail/qihucn/anaconda2/lib:$PATH
TH=/usr/local/packages/torch/install/bin/th
CUDA_VISIBLE_DEVICES=0 $TH /projects/grail/qihucn/ResNet/main.lua -batchSize 2 -dataset cv -imHeight 384 -imWidth 480 -modelType 2 -lr 0.0001 -d 10 -de 300 -optimizer adam -maxEpoch 100
