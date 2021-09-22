const tf = require('@tensorflow/tfjs-node');
const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout,
});
const Data = require('./data');

async function start() {
    const [trainXs, trainYs, idsToWords, wordsToIds, vocabSize] = await Data.load();
    const model = createModel(trainXs);

    await model.fit(trainXs, trainYs, {
        epochs: 20,
        batchSize: 100,
    });

    // predict something
    const testSentence = 'fox in box in socks in knox on box';
    const testIds = toIds(testSentence, wordsToIds);
    const testInputBuffer = tf.buffer([1, 9, vocabSize]);
    for (let i = 0; i < testIds.length; i++) {
        testInputBuffer.set(1, 1, i, testIds[i]);
    }
    const prediction = model.predict(testInputBuffer.toTensor());
    prediction.print();
    const predIndex = prediction.argMax().argMax();
    predIndex.array().then((arr) => {
            console.log(arr);
            console.log(idsToWords[arr]);
    });
}

function toIds(str, wordsToIds) {
    const tokens = str.split(' ');
    const ids = [];
    for (let i = 0; i < tokens.length; i++) {
        ids[i] = wordsToIds[tokens[i]];
    }
    return ids;
}

function createModel(inputTensor) {
    const model = tf.sequential();

    // add a lstm layer
    model.add(tf.layers.lstm({
        inputShape: [inputTensor.shape[1], inputTensor.shape[2]],
        units: 32,
        returnSequences: true,
    }));

    // flatten
    model.add(tf.layers.flatten());

    // add an output layer
    model.add(tf.layers.dense({
        units: 245,
        activation: 'softmax',
    }));

    // print summary
    model.summary();

    // compile
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    // return the model
    return model;
}

start();