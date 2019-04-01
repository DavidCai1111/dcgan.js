'use strict'
const path = require('path')
const tf = require('@tensorflow/tfjs')
const util = require('./util')

function getGeneratorModel () {
  const model = tf.sequential()

  model.add(tf.layers.dense({ inputDim: 100, units: 1024 }))
  model.add(tf.layers.activation({ activation: 'tanh' }))
  model.add(tf.layers.dense({ units: 128 * 7 * 7 }))
  model.add(tf.layers.batchNormalization())
  model.add(tf.layers.activation({ activation: 'tanh' }))
  model.add(tf.layers.reshape({ targetShape: [7, 7, 128] }))
  model.add(tf.layers.upSampling2d({ size: [2, 2] }))
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: [5, 5], padding: 'same' }))
  model.add(tf.layers.activation({ activation: 'tanh' }))
  model.add(tf.layers.upSampling2d({ size: [2, 2] }))
  model.add(tf.layers.conv2d({ filters: 1, kernelSize: [5, 5], padding: 'same' }))
  model.add(tf.layers.activation({ activation: 'tanh' }))

  return model
}

function getDiscriminatorModel () {
  const model = tf.sequential()

  model.add(tf.layers.conv2d({ filters: 64, kernelSize: [5, 5], padding: 'same', inputShape: [28, 28, 1] }))
  model.add(tf.layers.activation({ activation: 'tanh' }))
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }))
  model.add(tf.layers.conv2d({ filters: 128, kernelSize: [5, 5] }))
  model.add(tf.layers.activation({ activation: 'tanh' }))
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }))
  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({ units: 1024 }))
  model.add(tf.layers.activation({ activation: 'tanh' }))
  model.add(tf.layers.dense({ units: 1 }))
  model.add(tf.layers.activation({ activation: 'sigmoid' }))

  return model
}

function getGeneratorWithDiscriminator (generator, discriminator) {
  const model = tf.sequential()

  model.add(generator)

  generator.trainable = false

  model.add(discriminator)

  return model
}

async function generate (batchSize, nice = false) {
  const generator = await tf.loadLayersModel(
    `file://${path.join(__dirname, '../assets/generator-model/model.json')}`
  )

  let image = null

  if (!nice) {
    const noise = tf.randomUniform([batchSize, 100], -1, 1)
    const generatedImages = generator.predict(noise, { verbose: true })
    image = util.combineImages(generatedImages)
  } else {
    const discriminator = await tf.loadLayersModel(
      `file://${path.join(__dirname, '../assets/discriminator-model/model.json')}`
    )
    const noise = tf.randomUniform([batchSize * 20, 100], -1, 1)
    const generatedImages = generator.predict(noise, { verbose: true })
    const predictedResult = discriminator.predict(generatedImages, { verbose: true })

    const index = tf.range(0, batchSize * 20)
    index.reshape([batchSize * 20, 1])
    const predictedWithIndex = predictedResult.arraySync().concat(index.arraySync())

    predictedWithIndex.sort((a, b) => {
      if (a[0] < b[0]) return -1
      else return 1
    })

    let niceImages = tf.zeros(
      [batchSize, generatedImages.shape[1], generatedImages.shape[2]],
      'float32'
    )

    niceImages = niceImages.arraySync()

    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(predictedWithIndex[i][1])

      for (let j = 0; j < niceImages.shape[1]; j++) {
        for (let k = 0; k < niceImages.shape[2]; k++) {
          niceImages[i][j][k][0] = generatedImages[idx][j][k][0]
        }
      }
    }

    image = util.combineImages(tf.tensor(niceImages))
  }

  image = tf.add(127.5, tf.mul(image, 127.5))

  await util.saveImage(path.join(__dirname, '../assets/generated-image.png'), image)
}

module.exports = {
  getGeneratorModel,
  getDiscriminatorModel,
  getGeneratorWithDiscriminator,
  generate
}
