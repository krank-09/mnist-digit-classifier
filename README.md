# MNIST Digit Classifier

A small neural network in TensorFlow/Keras that recognises handwritten digits (MNIST). It was my first neural network: I've since moved to PyTorch for my other projects.

- Input: 28x28 grayscale images, scaled to 0–1 and flattened to 784 values
- Model: Dense(128, ReLU), Dense(64, ReLU), Dense(10, softmax)
- Trained for 5 epochs with Adam, 10% of the training set held out for validation
- Saves the model after training and loads it on later runs instead of retraining
- Plots the accuracy/loss curves and shows predictions on a few test images

## Run

```bash
pip install tensorflow numpy matplotlib
python mainp.py
```

On macOS, if downloading MNIST fails with a certificate error, run `Install Certificates.command` from your Python folder in Applications.
