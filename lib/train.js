'use strict'
const model = require('./model')

async function train () {
  const generator = model.getGeneratorModel()
  const discriminator = model.getDiscriminatorModel()
  const discriminatorOnGenerator = model.getGeneratorWithDiscriminator(generator, discriminator)
}

module.exports = {
  train
}
