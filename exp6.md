# Experiment 6

## Aim
Implement image classification on MNIST dataset using a CNN model with fully connected layers.

## Theory
Convolutional Neural Networks (CNNs) are deep learning architectures specifically designed for processing grid-like data such as images. Key components include:

1. **Convolutional Layers**: Extract spatial features using learnable filters that slide over the input image, detecting edges, textures, and patterns.

2. **Pooling Layers**: Reduce spatial dimensions while retaining important features, making the network more computationally efficient and robust to small translations.

3. **Fully Connected Layers**: After feature extraction, these layers perform high-level reasoning and classification by combining features learned from convolutional layers.

4. **MNIST Dataset**: A benchmark dataset containing 70,000 grayscale images of handwritten digits (0-9), each 28×28 pixels.

The CNN architecture typically follows: Input → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Dense → Output

This hierarchical feature learning makes CNNs highly effective for image classification tasks.


## Code: 
```
import tensorflow as tf
from tensorflow import keras

# Load MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., None] / 255.0, x_test[..., None] / 255.0

# Build CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)

# Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")
```

## Output:

```
Epoch 1/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.9245 - loss: 0.2525 - val_accuracy: 0.9815 - val_loss: 0.0655
Epoch 2/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9777 - loss: 0.0731 - val_accuracy: 0.9858 - val_loss: 0.0511
Epoch 3/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9846 - loss: 0.0504 - val_accuracy: 0.9888 - val_loss: 0.0396
Epoch 4/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9874 - loss: 0.0404 - val_accuracy: 0.9882 - val_loss: 0.0401
Epoch 5/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9898 - loss: 0.0329 - val_accuracy: 0.9878 - val_loss: 0.0377

Test Accuracy: 98.87%
```