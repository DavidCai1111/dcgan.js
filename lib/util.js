'use strict'
const tf = require('@tensorflow/tfjs')
const Jimp = require('jimp')

function combineImages (images) {
  const num = images.shape[0]
  const width = Math.floor(Math.sqrt(num))
  const height = Math.ceil(num / width)

  const shape = [images.shape[1], images.shape[2]]

  const image = tf.zeros([height * shape[0], width * shape[1]], images.dtype).arraySync()

  const imagesArray = images.arraySync()

  for (let k = 0; k < imagesArray.length; k++) {
    const i = Math.floor(k / width)
    const j = k % width

    for (let l = i * shape[0]; l < (i + 1) * shape[0]; l++) {
      for (let m = j * shape[1]; m < (j + 1) * shape[1]; m++) {
        image[l][m] = imagesArray[k][l - (i * shape[0])][m - (j * shape[1])][0]
      }
    }
  }

  return tf.tensor(image)
}

async function saveImage (path, tensor) {
  const shape = tensor.shape
  const tensorArray = tensor.as1D().arraySync()

  let i = 0

  await new Promise(function (resolve, reject) {
    // eslint-disable-next-line no-new
    new Jimp(shape[1], shape[0], function (err, image) {
      if (err) throw reject(err)
      image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
        this.bitmap.data[idx + 0] = tensorArray[i] * 255
        this.bitmap.data[idx + 1] = tensorArray[i] * 255
        this.bitmap.data[idx + 2] = tensorArray[i] * 255
        this.bitmap.data[idx + 3] = 255
        i++
      })

      image.writeAsync(path)
        .then(resolve)
        .catch(reject)
    })
  })
}

module.exports = {
  combineImages,
  saveImage
}
