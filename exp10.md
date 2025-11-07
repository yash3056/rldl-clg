# CBS - 2

## Aim
Implement deep learning methodologies on multimodal dataset.

## Theory
Multimodal learning involves processing and integrating information from multiple modalities (e.g., text, images, audio) to perform tasks that require understanding across different data types.

1. **Multimodal Data**: Data that comes from multiple sources or representations:
   - **Visual**: Images, videos
   - **Textual**: Captions, descriptions, labels
   - **Audio**: Speech, sounds
   - **Numerical**: Metadata, features

2. **Multimodal Fusion Strategies**:
   - **Early Fusion**: Combine raw features from different modalities at input level
   - **Late Fusion**: Process each modality separately and combine predictions
   - **Hybrid Fusion**: Combine at multiple levels of the network

3. **Architecture Design**:
   - Separate branches for each modality (e.g., CNN for images, RNN for text)
   - Feature extraction layers specific to each data type
   - Fusion layer to combine learned representations
   - Shared layers for final prediction

4. **Applications**:
   - Image captioning (vision + language)
   - Visual question answering
   - Video understanding with audio
   - Multimedia recommendation systems
   - Emotion recognition from audio-visual data

5. **Challenges**:
   - Alignment of different modalities
   - Handling missing modalities
   - Balancing contribution from each modality
   - Different sampling rates and feature dimensions

Multimodal learning enables richer representations and often achieves better performance than unimodal approaches by leveraging complementary information from different sources.

## Code:

```
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load MNIST for image data
(x_train_img, y_train), (x_test_img, y_test) = keras.datasets.mnist.load_data()
x_train_img = x_train_img[..., None] / 255.0
x_test_img = x_test_img[..., None] / 255.0

# Create synthetic text modality (one-hot encoded class descriptions)
def create_text_features(labels, noise_level=0.1):
    text_features = np.eye(10)[labels]
    text_features += np.random.randn(*text_features.shape) * noise_level
    return text_features.astype(np.float32)

x_train_text = create_text_features(y_train)
x_test_text = create_text_features(y_test)

# Build multimodal model with two input branches
# Image branch (CNN)
img_input = keras.Input(shape=(28, 28, 1), name='image_input')
img_features = keras.layers.Conv2D(32, 3, activation='relu')(img_input)
img_features = keras.layers.MaxPooling2D(2)(img_features)
img_features = keras.layers.Conv2D(64, 3, activation='relu')(img_features)
img_features = keras.layers.GlobalAveragePooling2D()(img_features)
img_features = keras.layers.Dense(64, activation='relu')(img_features)

# Text branch (Dense)
text_input = keras.Input(shape=(10,), name='text_input')
text_features = keras.layers.Dense(32, activation='relu')(text_input)
text_features = keras.layers.Dense(64, activation='relu')(text_features)

# Fusion layer - concatenate both modalities
merged = keras.layers.concatenate([img_features, text_features])
merged = keras.layers.Dense(128, activation='relu')(merged)
merged = keras.layers.Dropout(0.3)(merged)
output = keras.layers.Dense(10, activation='softmax')(merged)

# Create and compile multimodal model
model = keras.Model(inputs=[img_input, text_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Multimodal Model Architecture:")
model.summary()

# Train multimodal model
print("\nTraining Multimodal Model:")
model.fit(
    [x_train_img, x_train_text], y_train,
    epochs=5, batch_size=128, validation_split=0.1, verbose=1
)

# Evaluate
loss, acc = model.evaluate([x_test_img, x_test_text], y_test, verbose=0)
print(f"\nMultimodal Test Accuracy: {acc*100:.2f}%")

# Compare with unimodal (image only)
print("\nTraining Image-Only Model for comparison:")
img_only_model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
img_only_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
img_only_model.fit(x_train_img, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=0)
_, acc_img = img_only_model.evaluate(x_test_img, y_test, verbose=0)

print(f"\nCOMPARISON:")
print(f"Image-Only Model: {acc_img*100:.2f}%")
print(f"Multimodal Model: {acc*100:.2f}%")
print(f"Improvement: {(acc-acc_img)*100:.2f}%")

```

## Output:

```
Multimodal Model Architecture:
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ image_input (InputLayer)      │ (None, 28, 28, 1)         │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv2d (Conv2D)               │ (None, 26, 26, 32)        │             320 │ image_input[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ max_pooling2d (MaxPooling2D)  │ (None, 13, 13, 32)        │               0 │ conv2d[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv2d_1 (Conv2D)             │ (None, 11, 11, 64)        │          18,496 │ max_pooling2d[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ text_input (InputLayer)       │ (None, 10)                │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ global_average_pooling2d      │ (None, 64)                │               0 │ conv2d_1[0][0]             │
│ (GlobalAveragePooling2D)      │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_1 (Dense)               │ (None, 32)                │             352 │ text_input[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 64)                │           4,160 │ global_average_pooling2d[… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_2 (Dense)               │ (None, 64)                │           2,112 │ dense_1[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ concatenate (Concatenate)     │ (None, 128)               │               0 │ dense[0][0], dense_2[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_3 (Dense)               │ (None, 128)               │          16,512 │ concatenate[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout (Dropout)             │ (None, 128)               │               0 │ dense_3[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_4 (Dense)               │ (None, 10)                │           1,290 │ dropout[0][0]              │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 43,242 (168.91 KB)
 Trainable params: 43,242 (168.91 KB)
 Non-trainable params: 0 (0.00 B)

Training Multimodal Model:
Epoch 1/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.9710 - loss: 0.2329 - val_accuracy: 1.0000 - val_loss: 5.1156e-04
Epoch 2/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 1.0000 - loss: 9.6791e-04 - val_accuracy: 1.0000 - val_loss: 8.1525e-05
Epoch 3/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 1.0000 - loss: 3.4862e-04 - val_accuracy: 1.0000 - val_loss: 3.2963e-05
Epoch 4/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 1.0000 - loss: 1.6860e-04 - val_accuracy: 1.0000 - val_loss: 1.3007e-05
Epoch 5/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 1.0000 - loss: 1.2648e-04 - val_accuracy: 1.0000 - val_loss: 8.8949e-06

Multimodal Test Accuracy: 100.00%

Training Image-Only Model for comparison:
/home/aza/workspace/rldl-clg/.venv/lib/python3.12/site-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)

COMPARISON:
Image-Only Model: 90.37%
Multimodal Model: 100.00%
Improvement: 9.63%
```
