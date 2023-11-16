const tf = require('@tensorflow/tfjs');

const customOp = tf.customGrad((a, save) => {
  save([a]);

  return {
    value: a.pow(tf.scalar(3, 'int32')),
    gradFunc: (dy, saved) => [dy.mul(saved[0].abs())]
  };
});

const a = tf.tensor1d([0, -1, 2, -2, 8]);

const da = tf.grad(a => customOp(a));

console.log(`f'(a):`);
da(a).print();

// And Making X^3 
console.log(`f(a):`);
customOp(a).print();



