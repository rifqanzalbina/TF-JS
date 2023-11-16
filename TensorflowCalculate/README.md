# Tensorflow

Tensorflow js is a powerful librarrry machine learning with easy to learn and easy to know what is machine learning, so in this code example , i make that 'system' can calculate 2x - 1 by himself without inpout from user or human, interesting right? so let's me explain the code

## The Code

## Import Tensorflow JS

```javaScript
const tf = require("@tensorflow/tfjs");
```

This line imports the Tensorflow JS library, which is required to define and train machine learning models in JavaScript.

## Define and Run the Main Funtion

```javaScript
async function run() {
    // ... rest of the function
}
run();
```

Here, an asynchronous function run is defined and immediately invoked. Asynchronous functions are used because some TensorFlow.js operations, like training the model (fit method), return promises that need to be awaited.

## Create a Sequential Model

```javaScript
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
```

- 'tf.sequential()' creates a sequential model. This is a type of model that is composed of layers in sequence; each layer has exactly one input tensor and one output tensor.
- 'model.add()' adds a layer to the model. In this case, it's a dense (fully connected) layer.
- 'tf.layers.dense({units: 1, inputShape: [1]})' creates a dense layer with 1 unit (neuron) and an input shape of [1], meaning it expects to receive 1-dimensional data.

## Compile the Model

```javaScript
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
```

- 'model.compile()' prepares the model for training.
- 'loss: 'meanSquaredError'' sets the loss function to mean squared error, a common choice for regression tasks.
- optimizer: 'sgd' uses stochastic gradient descent as the optimization algorithm to minimize the loss function.

## Generate Synthetic Data For Training

```javaScript
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]); 
```

- 'tf.tensor2d()' creates 2-dimensional tensors from arrays.
- 'xs' and 'ys' are the input ('x') and output ('y') data for training. In this example, the relationship is 'y = 2x - 1'.

## Train The Model

```javaScript
await model.fit(xs, ys, {epochs: 50});
```

- 'model.fit()' trains the model with the given input ('xs') and output ('ys') data.
- Training runs for a specified number of iterations ('epochs'). Here, it's set to 50.

## Make Predictions 

```javaScript
model.predict(tf.tensor2d([5], [1, 1])).print();
```

- 'model.predict()' is used to make predictions with the trained model.
- It predicts the output for 'x = 5'. Given the training data, the expected output would be close to '2 * 5 - 1 = 9'.

So That's All i hove you guys can know what code is this :D


