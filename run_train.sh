#!/bin/bash

lr=0.25
ngram=7
epoch=60
dim=240
python train.py \
    --lr $lr \
    --ngram $ngram \
    --epoch $epoch \
    --dim $dim \
    > ../../log/toxic_en/log_lr${lr}_ngram${ngram}_epoch${epoch}_dim${dim}_$(date +%Y_%m_%d_%H_%M_%S).txt
