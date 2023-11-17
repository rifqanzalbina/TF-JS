# Tensorflow

Tensorflow is a strong library to learn machine learning because so many option that you can do with it, make model, prediction, calculate and many thing, so in this code example i have code that machine can predict the output from (y = x^2 + 2x + 1)  this formula, and first of all  again i'm tell this is all machine do, not from user / human tto give a good output

## The Code

## Import Tensorflow.js 

```javaScript
const tf = require('@tensorflow/tfjs');
```

- This line imports the TensorFlow.js library, enabling the use of its functions and classes. TensorFlow.js is a JavaScript library for training and deploying machine learning models.

## Define the Polynomial Regression Function

```javaScript
async function polynomialRegression() {
    // ... rest of the function
}
```

- We define an asynchronous function, 'polynomialRegression', because some operations in TensorFlow.js are asynchronous (like model training) and return Promises.

## Create the model

```javaScript
const model = tf.sequential();
model.add(tf.layers.dense({units: 4, inputShape: [1], activation: 'relu'}));
model.add(tf.layers.dense({units: 1}));
```

- 'tf.sequential()' initializes a new sequential model. A sequential model is a linear stack of layers.
- 'model.add()' adds layers to the model.
- The first layer added is a dense (fully connected) layer with 4 units (neurons) and uses the ReLU activation function. The 'inputShape' is set to '[1]', indicating that the input to this layer is 1-dimensional.
- The second layer is another dense layer with 1 unit. This layer is the output layer and will output the predicted value.

## Compile the model

## Compile The model

```javaScript
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
```

- 'model.compile()' prepares the model for training.
- The loss function is set to 'meanSquaredError', which is commonly used for regression problems. It measures the average of the squares of the errors—that is, the average squared difference between the predicted values and the actual values.
- The optimizer 'sgd' stands for stochastic gradient descent, an algorithm that's used to minimize the loss function.

## Generate Synthetic Training Data

```javaScript
const xs = tf.tensor1d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
const ys = tf.tensor1d(xs.dataSync().map(x => x**2 + 2*x + 1));
```

- 'tf.tensor1d()' creates a one-dimensional tensor from an array of numbers.
- 'xs' contains the input values.
- 'ys' contains the output values, calculated using a quadratic equation ('y = x^2 + 2x + 1'). This simulates a real-world scenario where we have known input-output pairs for training.

## Train the Model

```javaScript
await model.fit(xs.reshape([10, 1]), ys.reshape([10, 1]), {epochs: 200});
```

- 'model.fit()' is used to train the model.
- 'xs.reshape([10, 1])' and 'ys.reshape([10, 1])' convert the input and output tensors into a shape that TensorFlow.js can work with for training. Here, it reshapes them to be 2D tensors with 10 rows and 1 column.
- The model is trained over 200 epochs. An epoch is one cycle through the full training dataset.

## Use the model Prediction

```javaScript
const output = model.predict(tf.tensor2d([10], [1, 1]));
output.print();
```

- After training, 'model.predict()' is used to make predictions based on new data that the model hasn't seen before.
- Here, it predicts the output for 'x = 10'.
- 'output.print()' prints the predicted value to the console.

## Execute the Function

```javaScript
polynomialRegression();
```

- Finally, the 'polynomialRegression' function is called to execute all the above steps.

## Summary
This script demonstrates how to set up a simple neural network for polynomial regression using TensorFlow.js. It covers creating a model, adding layers, compiling the model, generating synthetic data for training, training the model, and making predictions with the trained model. The script is a complete example of using TensorFlow.js for a basic machine learning task.
