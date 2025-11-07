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
