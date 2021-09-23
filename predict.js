const tf = require('@tensorflow/tfjs-node');
const fs = require('fs/promises')
const Data = require("./data2");

const MODEL_DIRECTORY = 'file://./model/model.json';
const BUNDLE_DIRECTORY = './model/bundle.json';

async function start() {
    // load model from disk
    const model = await tf.loadLayersModel(MODEL_DIRECTORY);
    console.log('Loaded model:');
    model.summary();

    // load bundle from disk
    const bundleData = await fs.readFile(BUNDLE_DIRECTORY, 'utf-8');
    const bundle = JSON.parse(bundleData);

    // predict test
    // const seed = 'fox in box knox on box socks in mr';
    // const input = Data.sentenceToInput(seed, bundle.wordToIdMap, bundle.vocabSize);
    // const pred = await model.predict(input);
    // pred.print();

    // console.log('PAUSE');

    // have it write some seuss
    const n = 5;
    const seqLen = 10;
    const originalSeed = 'fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox fox';
    const TEMPERATURE = 0.5;
    let seedSentence = originalSeed;
    const predictions = [];
    let lastPred = undefined;
    for (let i = 0; i < n; i++) {
        // create seed input
        const seedInput = Data.sentenceToInput(seedSentence, bundle.wordToIdMap, bundle.vocabSize);

        // get a prediction
        let predRaw = await model.predict(seedInput);
        // predRaw.print();

        // we're only concerned with the last column of the prediction
        const nextWordPred = tf.slice(predRaw, [0, seqLen - 2], -1);
        const flatNextWord = nextWordPred.reshape([-1]);
        // flatNextWord.print();
        predRaw = flatNextWord;

        // await verbosePrediction(predRaw, bundle.idToWordMap);
        // print differences between predictions
        if (lastPred != undefined) {
            const sqDiff = await tf.squaredDifference(lastPred, predRaw).data();
            const sum = sqDiff.reduce((a, b) => a + b);
            console.log(i + ':sqDiffSum:', sum);
        }
        lastPred = predRaw;

        // sample an answer
        const sampleIndex = sample(tf.squeeze(predRaw), TEMPERATURE);
        // const predDecoded = await Data.labelToWord(predRaw, bundle.idToWordMap);
        const predDecoded = bundle.idToWordMap[sampleIndex];
        predictions.push(predDecoded);
        seedSentence = seedSentence.split(/\s/).slice(1).join(' ') + ' ' + predDecoded;
    }
    // print out the seuss
    console.log('seed:', originalSeed);
    console.log('continuation:', predictions.join(' '));
}

async function verbosePrediction(predRaw, idToWordMap) {
    const elements = await predRaw.data();
    // const map = {};
    for (let i = 0; i < idToWordMap.length; i++) {
        console.log(idToWordMap[i] + "=" + elements[i]);
        // map[idToWordMap[i]] = elements[i];
    }
    // console.log(map.sort());
    console.log('=========================\n\n\n')
}

// src: https://github.com/tensorflow/tfjs-examples/blob/master/lstm-text-generation/model.js
function sample(probs, temperature) {
    return tf.tidy(() => {
        const logits = tf.div(tf.log(probs), Math.max(temperature, 1e-6));
        const isNormalized = false;
        // `logits` is for a multinomial distribution, scaled by the temperature.
        // We randomly draw a sample from the distribution.
        return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
    });
}

start();
