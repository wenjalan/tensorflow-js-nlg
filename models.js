const tf = require('@tensorflow/tfjs-node')

/**
 * A Model is a function with learnable parameters that maps some input to some output.
 * In the case of Tensorflow, that means an input Tensor to an output Tensor.
 */

// Sequential Models
// The kind of the model you envision: several layers, one after the other, that starts with a single
// input layer and terminates in a single output layer, with a number of hidden layers between. Information
// flows from the input layer, through the hidden layers one at a time, then to the output layer.

// Create a new Sequential Model
const m1 = tf.sequential();

// Add Layers to the model using add()
m1.add(
    // Create a new Dense layer with tf.layers
    tf.layers.dense(
        {
            inputShape: [1], // the input will be a Tensor with shape [1], only applies to first layer
            units: 10, // this layer will contain 10 neurons
        }
    )
)

// add a hidden layer for shits and giggles
m1.add(tf.layers.dense({
    units: 100,
}));

// add an output layer
m1.add(tf.layers.dense({
    units: 10,
}));

// print some information about the model
m1.summary();