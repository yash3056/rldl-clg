# Experiment 7

## Aim
Train a sentiment analysis model on IMDB dataset using RNN layers with LSTM/GRU.

## Theory
Recurrent Neural Networks (RNNs) are designed to process sequential data by maintaining an internal state (memory) that captures information from previous time steps. Key concepts:

1. **RNN Architecture**: Unlike feedforward networks, RNNs have loops that allow information to persist, making them suitable for sequences like text, time series, and speech.

2. **LSTM (Long Short-Term Memory)**: An advanced RNN variant that solves the vanishing gradient problem through:
   - **Forget Gate**: Decides what information to discard
   - **Input Gate**: Determines what new information to store
   - **Output Gate**: Controls what information to output
   - **Cell State**: Carries information across long sequences

3. **GRU (Gated Recurrent Unit)**: A simplified version of LSTM with fewer parameters, combining forget and input gates into an update gate, making it faster while maintaining similar performance.

4. **Sentiment Analysis**: The task of determining emotional tone (positive/negative) from text. The IMDB dataset contains 50,000 movie reviews labeled for sentiment.

5. **Embedding Layer**: Converts words into dense vector representations, capturing semantic relationships.

For text classification, the architecture follows: Input → Embedding → LSTM/GRU → Dense → Output

## Code:

```
import tensorflow as tf
from tensorflow import keras

# Load IMDB dataset
vocab_size, max_len = 10000, 250
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

# Build LSTM model
model_lstm = keras.Sequential([
    keras.layers.Embedding(vocab_size, 128, input_length=max_len),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dense(1, activation='sigmoid')
])

# Build GRU model
model_gru = keras.Sequential([
    keras.layers.Embedding(vocab_size, 128, input_length=max_len),
    keras.layers.GRU(64, return_sequences=False),
    keras.layers.Dense(1, activation='sigmoid')
])

# Train LSTM
print("Training LSTM Model:")
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.2, verbose=1)
loss_lstm, acc_lstm = model_lstm.evaluate(x_test, y_test, verbose=0)
print(f"LSTM Test Accuracy: {acc_lstm*100:.2f}%")

# Train GRU
print("\nTraining GRU Model:")
model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_gru.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.2, verbose=1)
loss_gru, acc_gru = model_gru.evaluate(x_test, y_test, verbose=0)
print(f"GRU Test Accuracy: {acc_gru*100:.2f}%")

```

## Output:

```
Training LSTM Model:
Epoch 1/3
157/157 ━━━━━━━━━━━━━━━━━━━━ 9s 52ms/step - accuracy: 0.7693 - loss: 0.4756 - val_accuracy: 0.7908 - val_loss: 0.4649
Epoch 2/3
157/157 ━━━━━━━━━━━━━━━━━━━━ 8s 51ms/step - accuracy: 0.8939 - loss: 0.2732 - val_accuracy: 0.8662 - val_loss: 0.3134
Epoch 3/3
157/157 ━━━━━━━━━━━━━━━━━━━━ 8s 50ms/step - accuracy: 0.9232 - loss: 0.2080 - val_accuracy: 0.8672 - val_loss: 0.3229
LSTM Test Accuracy: 86.88%

Training GRU Model:
Epoch 1/3
157/157 ━━━━━━━━━━━━━━━━━━━━ 10s 59ms/step - accuracy: 0.7612 - loss: 0.4717 - val_accuracy: 0.8404 - val_loss: 0.3668
Epoch 2/3
157/157 ━━━━━━━━━━━━━━━━━━━━ 9s 57ms/step - accuracy: 0.8977 - loss: 0.2614 - val_accuracy: 0.8748 - val_loss: 0.3209
Epoch 3/3
157/157 ━━━━━━━━━━━━━━━━━━━━ 9s 57ms/step - accuracy: 0.9282 - loss: 0.1928 - val_accuracy: 0.8602 - val_loss: 0.3873
GRU Test Accuracy: 85.93%
```