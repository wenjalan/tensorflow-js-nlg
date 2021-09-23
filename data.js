const fs = require('fs').promises;
const tf = require('@tensorflow/tfjs-node');

const FILE_PATH = 'foxinsocks.txt';
const SEQUENCE_LENGTH = 10;

module.exports = class Data {
    constructor() {

    }

    // returns training inputs, training labels, an id to word map, and a word to id map
    static async load() {
        // echo to console
        console.log('Loading from file', FILE_PATH);

        // read from file system
        const data = (await fs.readFile(FILE_PATH, 'utf-8')).toLowerCase().replace(/[,.]/g, '');

        // get size and vocab
        const cleanedData = data.split(/\s+/);
        const vocab = new Set(cleanedData);

        // print some info
        console.log('> length:', cleanedData.length);
        console.log('> vocab size:', vocab.size);

        // map each word in the vocab to a number, and vice versa
        const [idsToWords, wordsToIds] = Data.getWordMaps(vocab);

        // print some info
        const sampleSize = 5;
        console.log('> id to word map first 5:');
        let sampleWords = [];
        for (let i = 0; i < sampleSize; i++) {
            console.log(i + '=' + idsToWords[i]);
            sampleWords[i] = idsToWords[i];
        }
        console.log('> word to id map first 5:');
        for (let i = 0; i < sampleSize; i++) {
            const word = sampleWords[i];
            console.log(word + '=' + wordsToIds[word]);
        }

        // encode the data into a series of ids
        const encodedData = [];
        cleanedData.forEach(word => {
            encodedData.push(wordsToIds[word]);
        });

        // print
        console.log('> encoded data:', encodedData);

        // create inputs and labels
        // input will be 9 words, output will be the 10th word following those 9 words
        const inputs = [];
        const labels = [];
        let n = 0;
        for (let i = 0; i < encodedData.length - SEQUENCE_LENGTH; i++) {
            // create input sequence
            for (let j = i; j < i + (SEQUENCE_LENGTH - 1); j++) {
                inputs.push(encodedData[j]);
            }

            // create output sequence
            const outputSequence = [encodedData[i + (SEQUENCE_LENGTH - 1)]];
            labels.push(outputSequence);
            n++;
        }

        // convert examples to tensors
        // populate a 3D tensor
        // trainXs: [n, input sample length, vocab size]
        // trainYs: [n, vocab size]
        const trainXsBuffer = tf.buffer([n, SEQUENCE_LENGTH - 1, vocab.size]);
        const trainYsBuffer = tf.buffer([n, vocab.size]);
        // for each example
        for (let i = 0; i < n; i++) {
            // get the next sequence
            const sequence = inputs[i];

            // for each timestep
            for (let j = 0; j < SEQUENCE_LENGTH - 1; j++) {
                trainXsBuffer.set(1, i, j, sequence[j]);
            }

            // set the element at [i, word id] to 1
            trainYsBuffer.set(1, i, labels[i]);
        }

        // turn into tensors
        const trainXs = trainXsBuffer.toTensor();
        const trainYs = trainYsBuffer.toTensor();
        trainXs.print();
        trainYs.print();

        // return it all
        return [trainXs, trainYs, idsToWords, wordsToIds, vocab.size];
    }

    // creates two maps of words to ids and ids to words
    static getWordMaps(vocab) {
        const idsToWords = [];
        const wordsToIds = {};
        let index = 0;
        vocab.forEach(word => {
            idsToWords[index] = word;
            wordsToIds[word] = index;
            index++;
        });
        return [idsToWords, wordsToIds];
    }

    // converts a string of words into a list of ids
    static sentenceToIds(str, wordsToIds) {
        const words = str.split(/\s+/);
        const ids = [];
        words.forEach((word) => {
            ids.push(wordsToIds[word]);
        });
        return ids;
    }

    // converts a list of ids into a sentence
    static idsToSentence(ids, idsToWords) {
        let str = '';
        ids.forEach((id) => {
            str += idsToWords[id] + ' ';
        });
        return str.trim();
    }

}
