# CBS - 1

## Aim
Implement Bi-GRU and Bi-LSTM on image or text dataset for prediction.

## Theory
Bidirectional RNNs process sequences in both forward and backward directions, capturing context from both past and future time steps. Key concepts:

1. **Bidirectional Architecture**: Consists of two separate recurrent layers:
   - **Forward Layer**: Processes sequence from start to end
   - **Backward Layer**: Processes sequence from end to start
   - Outputs from both layers are concatenated to form the final representation

2. **Bi-LSTM (Bidirectional Long Short-Term Memory)**:
   - Combines forward and backward LSTM layers
   - Captures long-range dependencies in both directions
   - Particularly effective for tasks where context from both sides matters (e.g., named entity recognition, POS tagging)

3. **Bi-GRU (Bidirectional Gated Recurrent Unit)**:
   - Similar to Bi-LSTM but with simpler gating mechanism
   - Fewer parameters, faster training
   - Often achieves comparable performance to Bi-LSTM

4. **Advantages**:
   - Complete context awareness: Each time step has information from both past and future
   - Better feature representation for sequence classification
   - Improved accuracy for many NLP tasks

5. **Applications**:
   - Sentiment analysis
   - Text classification
   - Machine translation
   - Speech recognition

The bidirectional approach is especially powerful when the entire sequence is available at prediction time, as it can leverage complete contextual information.

## Code:

```
import tensorflow as tf
from tensorflow import keras

# Load IMDB dataset for text classification
vocab_size, max_len = 10000, 200
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

# Bi-LSTM model
print("Building Bi-LSTM Model:")
model_bilstm = keras.Sequential([
    keras.layers.Embedding(vocab_size, 128, input_length=max_len),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_bilstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nTraining Bi-LSTM:")
model_bilstm.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.2, verbose=1)
loss_bilstm, acc_bilstm = model_bilstm.evaluate(x_test, y_test, verbose=0)
print(f"Bi-LSTM Test Accuracy: {acc_bilstm*100:.2f}%")

# Bi-GRU model
print("\n" + "="*50)
print("Building Bi-GRU Model:")
model_bigru = keras.Sequential([
    keras.layers.Embedding(vocab_size, 128, input_length=max_len),
    keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=False)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_bigru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nTraining Bi-GRU:")
model_bigru.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.2, verbose=1)
loss_bigru, acc_bigru = model_bigru.evaluate(x_test, y_test, verbose=0)
print(f"Bi-GRU Test Accuracy: {acc_bigru*100:.2f}%")

# Compare results
print("\n" + "="*50)
print("COMPARISON:")
print(f"Bi-LSTM: {acc_bilstm*100:.2f}%")
print(f"Bi-GRU:  {acc_bigru*100:.2f}%")

```

## Output:

```

Training Bi-LSTM:
Epoch 1/3
157/157 ━━━━━━━━━━━━━━━━━━━━ 10s 56ms/step - accuracy: 0.7776 - loss: 0.4477 - val_accuracy: 0.8686 - val_loss: 0.3095
Epoch 2/3
157/157 ━━━━━━━━━━━━━━━━━━━━ 9s 55ms/step - accuracy: 0.9112 - loss: 0.2300 - val_accuracy: 0.8748 - val_loss: 0.2965
Epoch 3/3
157/157 ━━━━━━━━━━━━━━━━━━━━ 9s 56ms/step - accuracy: 0.9360 - loss: 0.1718 - val_accuracy: 0.8706 - val_loss: 0.3429
Bi-LSTM Test Accuracy: 86.03%

==================================================
Building Bi-GRU Model:

Training Bi-GRU:
Epoch 1/3
157/157 ━━━━━━━━━━━━━━━━━━━━ 11s 62ms/step - accuracy: 0.7350 - loss: 0.4953 - val_accuracy: 0.8452 - val_loss: 0.3548
Epoch 2/3
157/157 ━━━━━━━━━━━━━━━━━━━━ 10s 63ms/step - accuracy: 0.8895 - loss: 0.2764 - val_accuracy: 0.8606 - val_loss: 0.3239
Epoch 3/3
157/157 ━━━━━━━━━━━━━━━━━━━━ 10s 65ms/step - accuracy: 0.9258 - loss: 0.1975 - val_accuracy: 0.8730 - val_loss: 0.3192
Bi-GRU Test Accuracy: 86.38%

==================================================
COMPARISON:
Bi-LSTM: 86.03%
Bi-GRU:  86.38%
```
