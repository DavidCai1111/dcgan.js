'use strict'
require('@tensorflow/tfjs-node')
const { train } = require('./lib/train')

;(async function () {
  await train(128)
})().catch(console.error)
