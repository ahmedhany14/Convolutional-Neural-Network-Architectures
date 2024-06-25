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

- The input to this model is a 32 \* 32 grayscale image hence the number of channels is one.

- First convolution layers, with the filter size 5 _ 5 and we have 6 such filters. As a result, we get a feature map of size 28 _ 28 \* 6

- Second convolution layers, with sixteen filters of size 5 _ 5. Again the feature map changed it is 10 _ 10 _ 16. we get a feature map of size 5 _ 5 \* 16.

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

### 2. AlexNet

Alexnet is a deep architecture, which works with RGB images, and The input to this model is the images of size 227 * 227 * 3

- The first convolution layer with 96 filters of size 11 * 11 with stride 4. The activation function used in this layer is relu. The output feature map is 55 * 55 * 96.

- Now we have the first Maxpooling layer, of size 3 * 3 and stride 2. Then we get the resulting feature map with the size 27 * 27 * 96.

- the second convolution operation. This time the filter size is reduced to 5 _ 5 and we have 256 such filters. The stride is 1 and padding 2. The activation function used is again relu. Now the output size we get is 27 * 27 * 256.

- Again we applied a max-pooling layer of size 3 _ 3 with stride 2. The resulting feature map is of shape 13 * 13 * 256

- Third convolution operation with 384 filters of size 3 * 3 stride 1 and also padding 1. Again the activation function used is relu. The output feature map is of shape 13 * 13 * 384.

- Fourth convolution operation with 384 filters of size 3 * 3. The stride along with the padding is 1. On top of that activation function used is relu. Now the output size remains unchanged i.e 13 _ 13 * 384.

- The final convolution layer of size 3 _ 3 with 256 such filters. The stride and padding are set to one also the activation function is relu. The resulting feature map is of shape 13 * 13 * 256.

- We apply the third max-pooling layer of size 3 * 3 and stride 2. Resulting in the feature map of the shape 6 * 6 * 256.

- After this, we have our first dropout layer. The drop-out rate is set to be 0.5.

- Then we have the first fully connected layer with a relu activation function. The size of the output is 4096. Next comes another dropout layer with the dropout rate fixed at 0.5.

- This followed by a second fully connected layer with 4096 neurons and relu activation.

- Finally, we have the last fully connected layer or output layer with 1000 neurons as we have 10000 classes in the data set. The activation function used at this layer is Softmax.

- Architecture Details:

| Layer (type)                | Output Shape       | Param #      |
|-----------------------------|--------------------|--------------|
| conv2d (Conv2D)             | (None, 67, 67, 96) |      34,944  |
| max_pooling2d (MaxPooling2D)| (None, 33, 33, 96) |           0  |
| conv2d_1 (Conv2D)           | (None, 33, 33, 256)|     614,656  |
| max_pooling2d_1 (MaxPooling2D)| (None, 16, 16, 256)|         0  |
| conv2d_2 (Conv2D)           | (None, 16, 16, 384)|     885,120  |
| conv2d_3 (Conv2D)           | (None, 16, 16, 384)|   1,327,488  |
| conv2d_4 (Conv2D)           | (None, 16, 16, 256)|     884,992  |
| max_pooling2d_2 (MaxPooling2D)| (None, 7, 7, 256)|          0  |
| dropout (Dropout)           | (None, 7, 7, 256)  |           0  |
| flatten (Flatten)           | (None, 12544)      |           0  |
| dense (Dense)               | (None, 9216)       | 115,614,720  |
| dense_1 (Dense)             | (None, 4096)       |  37,752,832  |
| dense_2 (Dense)             | (None, 4096)       |  16,781,312  |
| dense_3 (Dense)             | (None, 1000)       |   4,097,000  |

  Why AlexNet so important?

  Because of Deep Architecture, Innovative Techniques like (ReLU,Dropout, Data Augmentation).
