import tensorflow as tf
from tensorflow import keras
import numpy as np
import re

# Load and parse movie lines
lines = {}
with open('cornell_data/cornell movie-dialogs corpus/movie_lines.txt', 'r', encoding='iso-8859-1') as f:
    for line in f:
        parts = line.split(' +++$+++ ')
        if len(parts) == 5:
            lines[parts[0]] = parts[4].strip()

# Load conversations
conversations = []
with open('cornell_data/cornell movie-dialogs corpus/movie_conversations.txt', 'r', encoding='iso-8859-1') as f:
    for line in f:
        parts = line.split(' +++$+++ ')
        if len(parts) == 4:
            conv_ids = eval(parts[3].strip())
            conversations.append([lines.get(id, '') for id in conv_ids if id in lines])

# Prepare text pairs (question-answer)
pairs = []
for conv in conversations:
    for i in range(len(conv)-1):
        pairs.append([conv[i], conv[i+1]])

# Clean and limit dataset
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z?.!,¿]", " ", text)
    return text.strip()

pairs = [[clean_text(q), clean_text(a)] for q, a in pairs[:10000]]
questions, answers = zip(*pairs)

# Tokenization
tokenizer = keras.preprocessing.text.Tokenizer(num_words=8000, oov_token='<OOV>')
tokenizer.fit_on_texts(questions + answers)
q_seq = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(questions), maxlen=20, padding='post')
a_seq = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(answers), maxlen=20, padding='post')

# Simple text classification model (predict if answer length > 10 words)
labels = np.array([1 if len(a.split()) > 10 else 0 for a in answers])

# Build model
model = keras.Sequential([
    keras.layers.Embedding(8000, 64, input_length=20),
    keras.layers.LSTM(32),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(q_seq, labels, epochs=5, batch_size=64, validation_split=0.2, verbose=1)

loss, acc = model.evaluate(q_seq[:1000], labels[:1000], verbose=0)
print(f"\nText Classification Accuracy: {acc*100:.2f}%")
print(f"Sample conversations: {len(conversations)}")
print(f"Total question-answer pairs: {len(pairs)}")
