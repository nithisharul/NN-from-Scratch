# MNIST Neural Network from Scratch

A feedforward neural network built using only NumPy, trained on the MNIST handwritten digit dataset. No PyTorch, no TensorFlow.

## Architecture
```
Input (784) → Hidden Layer (10, ReLU) → Output (10, Softmax)
```

- Input: 784 neurons (28×28 flattened pixels)
- Hidden Layer: 10 neurons, ReLU activation
- Output: 10 neurons (digits 0–9), Softmax activation

## Files
```
notebook.ipynb   # main notebook with all code and experiments
```

## How it works

### Forward Pass
```
Z1 = W1 @ X + b1
A1 = ReLU(Z1)
Z2 = W2 @ A1 + b2
A2 = softmax(Z2)
```

### Backward Pass (Backpropagation)
```
dZ2 = A2 - Y
dW2 = dZ2 @ A1.T / N
dZ1 = W2.T @ dZ2 * ReLU'(Z1)
dW1 = dZ1 @ X.T / N
```

### Weight Update (Gradient Descent)
```
W = W - lr * dW
b = b - lr * db
```

## Failure Modes

### Failure 1: Bad Learning Rate
- Too large (lr=10) → loss explodes to NaN, overshoots minimum every step
- Too small (lr=0.00001) → loss barely moves, weights frozen
- Fix: Adam optimizer, adapts step size per weight based on gradient history

### Failure 2: Bad Weight Initialization
- Zeros → symmetry problem, all neurons identical, network never learns
- Large random (×10) → activations explode, loss diverges
- Fix: small random init (×0.01), weights centered around 0

### Failure 3: Vanishing Gradients
- Sigmoid activation → gradient squished by 0.25 every layer, early weights never update
- Fix: ReLU activation → gradient passes through unchanged

## Usage
```python
# train
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=500, alpha=0.1)

# predict
Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_test)
predictions = get_predictions(A2)
accuracy = get_accuracy(predictions, Y_test)
```

## Requirements
```
numpy
matplotlib
```
