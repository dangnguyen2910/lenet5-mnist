#!/bin/bash
mkdir data
curl -L -o data/mnist_dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset
unzip data/mnist_dataset.zip
