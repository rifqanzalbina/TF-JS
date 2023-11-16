# tensorflow

tensorflow Js is a powerful library that make in javascript that can make a model machine learning easy to use and easy to make model
<br>
so in this example repository i have some code that can result x^3 which is that machine learning by itself not code it from user / human , interesting right?
<br>
and this is explanation of the code which is simply you can try it too :D

## Import Tensorflow.js

```bash
const tf = require('@tensorflow/tfjs');
```

- This is import Tensorflow js libary and set variabel to 'tf'

## Make Operation Quantum

```bash
const customOp = tf.customGrad((a, save) => {
  save([a]);

  return {
    value: a.pow(tf.scalar(3, 'int32')),
    gradFunc: (dy, saved) => [dy.mul(saved[0].abs())]
  };
});
```
