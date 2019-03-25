'use strict'
const path = require('path')
const tf = require('@tensorflow/tfjs')
const model = require('./model')
const data = require('./data')
const util = require('./util')

async function train (batchSize, epoch = 100) {
  const generator = model.getGeneratorModel()
  const discriminator = model.getDiscriminatorModel()
  const discriminatorOnGenerator = model.getGeneratorWithDiscriminator(generator, discriminator)

  const trainData = data.getTrainData()
  const testData = data.getTestData()

  const xTrain = trainData.images
  const yTrain = trainData.labels

  const xTest = testData.images
  const yTest = testData.labels

  generator.compile({
    loss: 'binaryCrossentropy',
    optimizer: 'SGD'
  })

  discriminatorOnGenerator.compile({
    loss: 'binaryCrossentropy',
    optimizer: tf.train.momentum(0.0005, 0.9, true)
  })

  discriminator.trainable = true

  discriminator.compile({
    loss: 'binaryCrossentropy',
    optimizer: tf.train.momentum(0.0005, 0.9, true)
  })

  for (let i = 0; i < epoch; i++) {
    console.log(`[dcgan.js] Epoch: ${i + 1}`)
    console.log(`[dcgan.js] Number of batches: ${xTrain.shape[0] / batchSize}`)

    for (let j = 0; j < (xTrain.shape[0] / batchSize); j++) {
      const noise = tf.randomUniform([batchSize, 100], -1, 1)
      const imageBatch = tf.slice(xTrain, j * batchSize, batchSize)
      const generatedImages = generator.predict(noise, { verbose: false })

      if (j % 20 === 0) {
        let image = util.combineImages(generatedImages)
        image = tf.add(tf.scalar(127.5), tf.mul(image, tf.scalar(127.5)))

        util.saveImage(
          path.join(__dirname, `../assets/${epoch}_${j}_${new Date().toISOString()}.png`),
          image
        )
      }

      const x = tf.concat([imageBatch, generatedImages])
      const y = tf.concat([tf.tensor(new Array(batchSize).fill(1)), tf.tensor(new Array(batchSize).fill(0))])

      const discriminatorLoss = discriminator.trainOnBatch(x, y)

      console.log(`[dcgan.js] discriminatorLoss-${j}: ${discriminatorLoss}`)
    }
  }
}

module.exports = {
  train
}
