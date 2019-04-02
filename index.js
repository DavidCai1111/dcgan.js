'use strict'
require('@tensorflow/tfjs-node-gpu')
const { train } = require('./lib/train')

;(async function () {
  await train(512, 1000)
})().catch(console.error)
