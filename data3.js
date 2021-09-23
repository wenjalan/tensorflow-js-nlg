const fs = require('fs').promises;
const tf = require('@tensorflow/tfjs-node');

const FILE_PATH = 'foxinsocks.txt';
const SEQUENCE_LENGTH = 10;

module.exports = class Data {

    // returns training inputs, training labels
    static async load() {
        // echo
        console.log('Loading from file', FILE_PATH);

        // read the data from the file and get all unique words
        const data = (await fs.readFile(FILE_PATH, 'utf-8')).toLowerCase().replace(/[,.]/g, '');
        const cleanedData = data.split(/\s+/);
        const vocab = new Set(cleanedData);
        console.log('> length:', cleanedData.length);
        console.log('> vocab size:', vocab.size);

        // split data into examples
        // todo: create labels that are sequences, not single words
        // see https://datascience.stackexchange.com/questions/22047/time-series-forecasting-with-rnnstateful-lstm-produces-constant-values
        const examples = new Map();
        for (let i = 0; i < cleanedData.length - SEQUENCE_LENGTH; i++) {
            // create an input sequence containing [i, i + seqLen - 2] words
            const inputSequence = [];
            for (let j = 0; j < SEQUENCE_LENGTH - 2; j++) {
                const word = cleanedData[i + j];
                inputSequence.push(word);
            }

            // create an output sequence containing [i + 1, i + seqLen - 1] words
            const outputSequence = [];
            for (let j = 1; j < SEQUENCE_LENGTH - 1; j++) {
                const word = cleanedData[i + j];
                outputSequence.push(word);
            }

            // add to examples
            examples.set(inputSequence, outputSequence);
        }

        // encode all vocab to integers
        const [idToWord, wordToId] = Data.getWordMaps(vocab);

        // encode all examples to integers
        const encodedExamples = new Map();
        for (const [input, label] of examples) {
            const encodedInput = this.wordsToIds(input, wordToId);
            const encodedLabel = this.wordsToIds(label, wordToId);
            encodedExamples.set(encodedInput, encodedLabel);
        }

        // convert the examples into tensors
        const numExamples = encodedExamples.size;
        const sampleLen = SEQUENCE_LENGTH - 1;
        const vocabSetSize = vocab.size;
        const trainXsBuffer = tf.buffer([numExamples, sampleLen, vocabSetSize]);
        const trainYsBuffer = tf.buffer([numExamples, sampleLen, vocabSetSize]);
        let i = 0;
        for (const [input, label] of encodedExamples) {
            // trainXs
            for (let j = 0; j < sampleLen; j++) {
                const code = input[j];
                trainXsBuffer.set(1, i, j, code);
            }

            // trainYs
            for (let j = 0; j < sampleLen; j++) {
                const code = label[j];
                trainYsBuffer.set(1, i, j, code);
            }

            // inc
            i++;
        }

        // return the tensors
        const trainXs = trainXsBuffer.toTensor();
        const trainYs = trainYsBuffer.toTensor();
        // trainXs.print();
        // trainYs.print();
        return [trainXs, trainYs, wordToId, idToWord, vocabSetSize];
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

    // converts a list of words to their ids
    static wordsToIds(words, wordToIdMap) {
        const ids = [];
        words.forEach((word) => {
            ids.push(wordToIdMap[word]);
        });
        return ids;
    }

    // converts a sentence into an input tensor
    // shape: [1, sequenceLen, vocabSize]
    static sentenceToInput(str, wordToIdMap, vocabSize) {
        const words = str.split(/\s+/);
        const ids = this.wordsToIds(words, wordToIdMap);
        const sequenceLen = ids.length;
        const inputBuffer = tf.buffer([1, sequenceLen, vocabSize]);
        for (let i = 0; i < sequenceLen; i++) {
            const code = ids[i];
            inputBuffer.set(1, 1, i, code);
        }
        return inputBuffer.toTensor();
    }

    // converts an output tensor to a word
    static async labelToWord(label, idToWordMap) {
        const elements = await label.data();
        const id = elements.indexOf(Math.max.apply(Math, elements));
        const word = idToWordMap[id];
        return word;
    }
}
