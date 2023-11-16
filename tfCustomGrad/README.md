# Tensorflow

tensorflow Js is a powerful library that make in javascript that can make a model machine learning easy to use and easy to make model
<br>
so in this example repository i have some code that can result x^3 which is that machine learning by itself not code it from user / human , interesting right?
<br>
and this is explanation of the code which is simply you can try it too :D

## Import Tensorflow.js

```javaScript
const tf = require('@tensorflow/tfjs');
```

- This is import Tensorflow js libary and set variabel to 'tf'

## Make Operation Quantum

```javaScript
const customOp = tf.customGrad((a, save) => {
  save([a]);

  return {
    value: a.pow(tf.scalar(3, 'int32')),
    gradFunc: (dy, saved) => [dy.mul(saved[0].abs())]
  };
});
```

- 'tf.customGrad' is used to define custom operations and the gradients (children) of those operations.

- This function receives input 'a' and uses 'save' to save it.

- 'value' is the output of this custom operation. In this case, 'a.pow(tf.scalar(3, 'int32'))' computes the value of 'a' cubed.

- 'gradFunc' defines how the gradient of this operation is calculated. This takes the outer derivative ('dy') and the saved value ('saved'), then multiplies them. In this case, the gradient is calculated as 'dy' times the absolute value of 'a' (saved[0].abs()).

## Make a Tensor and Calculate Gradients

```javaScript
const a = tf.tensor1d([0, -1, 2, -2, 8]);
const da = tf.grad(a => customOp(a));

console.log(`f'(a):`);
da(a).print();
```

- Creating a one-dimensional tensor 'a'.

- 'tf.grad' is used to create a function ('da') that calculates the gradient of a custom operation against 'a'.

- 'da(a).print()' displays the gradient of a after it has been applied to 'a' custom operation.

## Calculate Value Custom Operation

```javaScript
console.log(`f(a):`);
customOp(a).print();
```

- Menghitung dan menampilkan hasil operasi kustom 'customOp(a)'.

## Summary
This code demonstrates the creation and use of a custom mathematical operation in TensorFlow.js, where it calculates the cubed value of the input, and also defines how the derivative (gradient) of the operation is calculated. Then, the code applies these operations to a tensor and displays the results and gradients of the operations. This is an example of a more general concept in machine learning and neural network model development, where gradient calculation is key in the training process.
