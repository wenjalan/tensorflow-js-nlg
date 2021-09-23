const tf = require('@tensorflow/tfjs-node');
const fs = require('fs/promises');
const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout,
});
const Data = require('./data2');
const MODEL_DIRECTORY = 'file://./model/';
const BUNDLE_DIRECTORY = './model/bundle.json';

async function start() {
    const [trainXs, trainYs, wordToIdMap, idToWordMap, vocabSize] = await Data.load();

    const model = createModel(trainXs);
    console.log('Model Created ===');
    model.summary();

    console.log('Beginning training...');
    await model.fit(trainXs, trainYs, {
        epochs: 5,
        batchSize: 100,
    });
    console.log('Training complete!');

    // save the model
    const saveResult = await model.save(MODEL_DIRECTORY);
    console.log('Saved model.')

    // save model bundles
    const bundles = {
        wordToIdMap: wordToIdMap,
        idToWordMap: idToWordMap,
        vocabSize: vocabSize,
    };
    await fs.writeFile(BUNDLE_DIRECTORY, JSON.stringify(bundles));
    console.log('Saved bundles.');
}

function createModel(inputTensor) {
    const model = tf.sequential();

    // add batch normalization
    model.add(tf.layers.batchNormalization({
        inputShape: [inputTensor.shape[1], inputTensor.shape[2]],
        units: 256,
    }));

    // add a lstm layer
    model.add(tf.layers.lstm({
        units: 256,
        returnSequences: true,
    }));

    // flatten
    model.add(tf.layers.flatten());

    // add an output layer
    model.add(tf.layers.dense({
        units: inputTensor.shape[2],
        activation: 'softmax',
    }));

    // print summary
    model.summary();

    // training
    const optimizer = tf.train.rmsprop(0.01);

    // compile
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    // return the model
    return model;
}

start();
