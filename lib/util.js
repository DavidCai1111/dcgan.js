'use strict'
const tf = require('@tensorflow/tfjs')
const Jimp = require('jimp')

function combineImages (images) {
  const num = images.shape[0]
  const width = Math.sqrt(num)
  const height = Math.ceil(num / width)

  const shape = [images.shape[1], images.shape[2], images.shape[3]]

  const image = tf.zeros([height * shape[0], width * shape[1]], images.dtype)

  // TODO: Combine the image
  return image
}

function saveImage (path, tensor) {
  const tensorArray = Array.from(tensor.dataSync())

  let i = 0

  // eslint-disable-next-line no-new
  new Jimp(28, 28, function (err, image) {
    if (err) throw err

    image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
      this.bitmap.data[idx + 0] = tensorArray[i++]
      this.bitmap.data[idx + 1] = tensorArray[i++]
      this.bitmap.data[idx + 2] = tensorArray[i++]
      this.bitmap.data[idx + 3] = 255
    })

    image.write(path)
  })
}

module.exports = {
  combineImages,
  saveImage
}
