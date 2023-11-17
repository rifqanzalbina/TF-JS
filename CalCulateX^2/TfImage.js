const tf = require('@tensorflow/tfjs');

async function polynomialRegression(){
    const model = tf.sequential();
    
    model.add(tf.layers.dense({units : 4, inputShape : [1], activation : 'relu'}));
    model.add(tf.layers.dense({units : 1}));

    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xs = tf.tensor1d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const ys = tf.tensor1d(xs.dataSync().map(x => x**2 + 2*x + 1));

    await model.fit(xs.reshape([10, 1]), ys.reshape([10, 1]), {epochs : 200});

    const output = model.predict(tf.tensor2d([10], [1, 1]));
    output.print();
}

polynomialRegression();