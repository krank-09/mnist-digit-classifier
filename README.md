MNIST handwritten digit classification using TensorFlow / Keras.

This script builds, trains, saves, reloads, evaluates, and visualizes a simple
feed-forward neural network for classifying digits from the MNIST dataset.

What the script does:
- Disables the default SSL certificate verification behavior so MNIST can be
    downloaded in environments where certificate checks may fail.
- Imports TensorFlow, Keras components, NumPy, Matplotlib, and the MNIST
    dataset loader.
- Loads the MNIST dataset, which contains 28x28 grayscale images of digits
    (0 through 9).
- Normalizes pixel values from the range 0–255 to 0–1 for better training
    stability.
- Reshapes each image from a 28x28 matrix into a 784-element vector, since the
    model is a dense (fully connected) neural network.
- Builds a Sequential model with:
    - an explicit Input layer for 784 features,
    - a Dense hidden layer with ReLU activation,
    - a second Dense hidden layer with ReLU activation,
    - a Dense output layer with softmax activation for 10-class prediction.
- Compiles the model with:
    - Adam optimizer for adaptive gradient-based learning,
    - sparse_categorical_crossentropy loss because labels are integer-encoded,
    - accuracy as the evaluation metric.
- Checks whether a previously trained model file exists:
    - If the file is present, the model is loaded and training is skipped.
    - If the file is absent, the model is trained for 5 epochs with a batch size
        of 32 and 10% of the training data reserved for validation.
- Saves the trained model in Keras format for future reuse.
- Plots and saves training curves for accuracy and loss.
- Evaluates the model on the test dataset and prints test loss and accuracy.
- Generates predictions for the first 5 test images and visualizes them with
    their predicted and actual labels.

Commands / APIs used:
- ssl._create_default_https_context / ssl._create_unverified_context:
    Used to bypass SSL verification during dataset download.
- mnist.load_data():
    Downloads and returns the training and test MNIST splits.
- x / 255.0:
    Normalizes image pixel values.
- reshape(-1, 784):
    Flattens images into vectors suitable for Dense layers.
- Sequential([...]):
    Creates a stack of layers for a straightforward neural network.
- Input(shape=(784,)):
    Defines the expected input size.
- Dense(..., activation='relu'):
    Adds fully connected hidden layers with ReLU nonlinearity.
- Dense(..., activation='softmax'):
    Produces class probabilities across 10 digit classes.
- model.compile(...):
    Configures the training process.
- model.fit(...):
    Trains the model on the training data.
- model.save(...):
    Writes the trained model to disk.
- tf.keras.models.load_model(...):
    Loads a saved model from disk.
- model.evaluate(...):
    Computes final loss and accuracy on the test set.
- model.predict(...):
    Produces class probabilities for unseen samples.
- np.argmax(...):
    Converts probability vectors into predicted digit labels.
- matplotlib.pyplot:
    Used to plot training history and sample predictions.

Outputs produced by the script:
- mnist_model.keras: Saved model file
- training_history.png: Plot of training/validation accuracy and loss
- predictions.png: Visualization of sample predictions

Overall purpose:
The script demonstrates a complete machine learning workflow for digit
classification, including data preprocessing, model training, persistence,
evaluation, and result visualization.
