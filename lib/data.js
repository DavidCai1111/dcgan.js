'use strict'
const path = require('path')
const fs = require('fs')
const assert = require('assert')
const tf = require('@tensorflow/tfjs')

const TRAIN_DATA_BUFFER = fs.readFileSync(path.join(__dirname, '../data/train-images-idx3-ubyte'))
const TRAIN_LABEL_BUFFER = fs.readFileSync(path.join(__dirname, '../data/train-labels-idx1-ubyte'))
const TEST_DATA_BUFFER = fs.readFileSync(path.join(__dirname, '../data/t10k-images-idx3-ubyte'))
const TEST_LABEL_BUFFER = fs.readFileSync(path.join(__dirname, '../data/t10k-labels-idx1-ubyte'))

const IMAGE_HEADER_MAGIC_NUM = 2051
const IMAGE_HEADER_BYTES = 16
const IMAGE_HEIGHT = 28
const IMAGE_WIDTH = 28
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH
const LABEL_HEADER_MAGIC_NUM = 2049
const LABEL_HEADER_BYTES = 8
const LABEL_RECORD_BYTE = 1
const LABEL_FLAT_SIZE = 10

function loadHeaderValues (buffer, headerLength) {
  const headerValues = []
  for (let i = 0; i < headerLength / 4; i++) {
    headerValues[i] = buffer.readUInt32BE(i * 4)
  }

  return headerValues
}

function loadImagesFromBuffer (buffer) {
  const headerValues = loadHeaderValues(buffer, IMAGE_HEADER_BYTES)

  assert.strictEqual(headerValues[0], IMAGE_HEADER_MAGIC_NUM)
  assert.strictEqual(headerValues[2], IMAGE_HEIGHT)
  assert.strictEqual(headerValues[3], IMAGE_WIDTH)

  const images = []

  let index = IMAGE_HEADER_BYTES

  while (index < buffer.byteLength) {
    const array = new Float32Array(IMAGE_FLAT_SIZE)

    for (let i = 0; i < IMAGE_FLAT_SIZE; i++) {
      array[i] = buffer.readUInt8(index++) / 255
    }
    images.push(array)
  }

  assert.strictEqual(images.length, headerValues[1])

  return images
}

function loadLabelsFromBuffer (buffer) {
  const headerValues = loadHeaderValues(buffer, LABEL_HEADER_BYTES)

  assert.strictEqual(headerValues[0], LABEL_HEADER_MAGIC_NUM)

  const labels = []

  let index = LABEL_HEADER_BYTES

  while (index < buffer.byteLength) {
    const array = new Int32Array(LABEL_RECORD_BYTE)

    for (let i = 0; i < LABEL_RECORD_BYTE; i++) {
      array[i] = buffer.readUInt8(index++)
    }
    labels.push(array)
  }

  assert.strictEqual(labels.length, headerValues[1])

  return labels
}

class MnistDataSet {
  constructor () {
    this.dataset = [
      loadImagesFromBuffer(TRAIN_DATA_BUFFER),
      loadLabelsFromBuffer(TRAIN_LABEL_BUFFER),
      loadImagesFromBuffer(TEST_DATA_BUFFER),
      loadLabelsFromBuffer(TEST_LABEL_BUFFER)
    ]

    this.trainSize = this.dataset[0].length
    this.testSize = this.dataset[2].length
    this.trainBatchIndex = 0
    this.testBatchIndex = 0
  }

  getTrainData () {
    return this._getData(true)
  }

  getTestData () {
    return this._getData(false)
  }

  _getData (isTrainingData = true) {
    let imagesIndex, labelsIndex

    if (isTrainingData) {
      imagesIndex = 0
      labelsIndex = 1
    } else {
      imagesIndex = 2
      labelsIndex = 3
    }

    const size = this.dataset[imagesIndex].length

    tf.util.assert(
      this.dataset[labelsIndex].length === size,
      `Mismatch in the number of images (${size}) and ` +
            `the number of labels (${this.dataset[labelsIndex].length})`)

    const imagesShape = [size, IMAGE_HEIGHT, IMAGE_WIDTH, 1]
    const images = new Float32Array(tf.util.sizeFromShape(imagesShape))
    const labels = new Int32Array(tf.util.sizeFromShape([size, 1]))

    let imageOffset = 0
    let labelOffset = 0
    for (let i = 0; i < size; ++i) {
      images.set(this.dataset[imagesIndex][i], imageOffset)
      labels.set(this.dataset[labelsIndex][i], labelOffset)
      imageOffset += IMAGE_FLAT_SIZE
      labelOffset += 1
    }

    return {
      images: tf.tensor4d(images, imagesShape),
      labels: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
    }
  }
}

module.exports = new MnistDataSet()
