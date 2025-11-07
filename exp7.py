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
