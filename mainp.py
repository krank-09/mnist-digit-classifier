"""MNIST digit classifier: a small dense network in TensorFlow/Keras.

Trains for 5 epochs (or loads mnist_model.keras if it exists), evaluates on the
test set and plots a few predictions.
"""
import os

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