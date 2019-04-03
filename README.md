# dcgan.js
[![Build Status](https://travis-ci.org/DavidCai1993/dcgan.js.svg?branch=master)](https://travis-ci.org/DavidCai1993/dcgan.js)

Node.js implementation of Deep Convolutional Generative Adversarial Networks

## Training Example

![example.gif](./example/example.gif)

## Usage

### Train

```js
node dcgan.js train --epoch <epoch> --batchSize <batchSize> [--gpu]
// Example: node dcgan.js train --epoch 100 --batchSize 128 --gpu
```
