'use strict'
const Jimp = require('jimp')

function combineImages (images) {

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
