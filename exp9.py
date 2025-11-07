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
