const tf = require('@tensorflow/tfjs');

async function classifyImage(imageTensor) {
    // Load the pre-trained MobileNet model
    const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

    // Preprocess the image to match MobileNet's input format
    const processedImage = tf.tidy(() => {
        // Resize to 224 x 224
        let resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
        // Normalize the image from [0, 255] to [-1, 1].
        return resized.toFloat().sub(tf.scalar(127.5)).div(tf.scalar(127.5)).reshape([1, 224, 224, 3]);
    });

    // Make a prediction
    const prediction = model.predict(processedImage);

    // Convert logits to probabilities and class names
    prediction.softmax().print();

    // Dispose the tensors
    tf.dispose([imageTensor, processedImage, prediction]);
}

// Example: Load an image tensor (Replace this with actual image tensor)
const imageTensor = tf.zeros([224, 224, 3]); 

// Run the classification
classifyImage(imageTensor);
