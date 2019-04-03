'use strict'
const transfer = require('commander')
const pkg = require('./package')

transfer.version(pkg.version)

transfer
  .command('train')
  .description('train the model')
  .option('-e, --epoch', 'Epoch')
  .option('-b, --batchSize', 'Batch Size')
  .option('-g, --gpu')
  .action(function (opts) {
    ;(async function () {
      const { epoch, batchSize, gpu } = opts

      if (gpu) require('@tensorflow/tfjs-node-gpu')
      else require('@tensorflow/tfjs-node')

      await require('./lib/train').train(batchSize, epoch)
    })().catch(console.error)
  })

transfer.parse(process.argv)
