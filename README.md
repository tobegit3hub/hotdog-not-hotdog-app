# Hotdog Not Hotdog App

## Introduction

Hot dog or not hot dog application with machine learning.

## Usage

Train with [pre-trained checkpoint](https://github.com/tensorflow/models/tree/master/research/slim).

```
./train.py --model="inception_v3" --load_checkpoint=True  --model_trainable_scopes="InceptionV3/Logits,InceptionV3/AuxLogits"
```

Train with DNN model.

```
./train.py
```
