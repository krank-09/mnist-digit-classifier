
"""
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
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import numpy as np

# ─── 1. LOAD DATA ───────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Data loaded:", x_train.shape)

# ─── 2. PREPROCESS ──────────────────────────────────────────
x_train = x_train / 255.0
x_test  = x_test  / 255.0

x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# ─── 3. BUILD MODEL ─────────────────────────────────────────
model = Sequential([
    Input(shape=(784,)),           # Fixed: no more UserWarning
    Dense(128, activation='relu'),
    Dense(64,  activation='relu'),
    Dense(10,  activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── 4. TRAIN OR LOAD ───────────────────────────────────────
import os

if os.path.exists('mnist_model.keras'):
    print("\nLoading saved model — skipping training...")
    model = tf.keras.models.load_model('mnist_model.keras')
else:
    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1
    )
    model.save('mnist_model.keras')
    print("Model saved!")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# ─── 5. EVALUATE ────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss:     {test_loss:.4f}")

# ─── 6. VISUALIZE PREDICTIONS ───────────────────────────────
predictions = model.predict(x_test[:5])

plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    predicted = np.argmax(predictions[i])
    actual    = y_test[i]
    color     = 'green' if predicted == actual else 'red'
    plt.title(f"Pred: {predicted}\nActual: {actual}", color=color)
    plt.axis('off')

plt.suptitle('Model Predictions (Green=Correct, Red=Wrong)')
plt.tight_layout()
plt.savefig('predictions.png')
plt.show()