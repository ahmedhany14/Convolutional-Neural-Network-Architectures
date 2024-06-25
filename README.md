<br />
<p align="center">
  <h3 align="center">  Convolutional Neural Network Architectures </h3>
</p>

This repository contains a collection of popular CNN architectures implemented in Python using various deep learning frameworks.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Architectures](#architectures)

## Introduction

Convolutional Neural Networks (CNNs) are a class of deep learning models that have shown remarkable performance in various computer vision tasks such as image classification, object detection, and segmentation. This repository aims to provide clean, well-documented, and efficient implementations of popular CNN architectures to facilitate learning and experimentation.

## Architectures

The repository currently includes implementations of the following CNN architectures:

- [LeNet-5](https://medium.com/@siddheshb008/lenet-5-architecture-explained-3b559cb2d52b)
- [AlexNet](https://paravisionlab.co.in/alexnet-architecture/)
- [VGGNet](https://medium.com/@siddheshb008/vgg-net-architecture-explained-71179310050f)
- [ResNet](https://medium.com/@siddheshb008/resnet-architecture-explained-47309ea9283d)
- [GoogLeNet (Inception)](https://viso.ai/deep-learning/googlenet-explained-the-inception-model-that-won-imagenet/)

### 1. LeNet-5

- The network has 5 layers
- The input to this model is a 32 * 32 grayscale image hence the number of channels is one.
- First convolution layers, with the filter size 5 * 5 and we have 6 such filters. As a result, we get a feature map of size 28 * 28 * 6
- Second convolution layers, with sixteen filters of size 5 * 5. Again the feature map changed it is 10 * 10 * 16. we get a feature map of size 5 * 5 * 16.
- After that we Flatten the result, to be ready for the 3rd layer
- The 3rd layer, is a FC layer with 120 neurons
- The 4th layer, is a FC layer with 84 neurons
- The 5th layer, is a outout layer with 10 neurons
- Architecture Details:

| Layer (type)        | Output Shape      | Param # |
| ------------------- | ----------------- | ------- |
| conv2d_4 (Conv2D)   | (None, 24, 24, 6) | 156     |
| average_pooling2d_4 | (None, 12, 12, 6) | 0       |
| (AveragePooling2D)  |                   |         |
| conv2d_5 (Conv2D)   | (None, 8, 8, 15)  | 2,265   |
| average_pooling2d_5 | (None, 4, 4, 15)  | 0       |
| (AveragePooling2D)  |                   |         |
| flatten_2 (Flatten) | (None, 240)       | 0       |
| dense_6 (Dense)     | (None, 120)       | 28,920  |
| dense_7 (Dense)     | (None, 84)        | 10,164  |
| dense_8 (Dense)     | (None, 10)        | 850     |
