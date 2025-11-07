# Experiment 8

## Aim
Apply Deep Learning Models in the field of Natural Language Processing using the Cornell Movie Dialogs Corpus.

## Theory
Natural Language Processing (NLP) involves using computational techniques to analyze and understand human language. Deep learning has revolutionized NLP through:

1. **Word Embeddings**: Dense vector representations of words that capture semantic relationships. Words with similar meanings have similar vectors in the embedding space.

2. **Sequence-to-Sequence Models**: Architecture for mapping one sequence to another, useful for tasks like machine translation, summarization, and dialogue generation.

3. **Text Preprocessing**: Essential steps include:
   - Tokenization: Breaking text into words/tokens
   - Cleaning: Removing special characters, lowercasing
   - Vocabulary building: Creating word-to-index mappings
   - Padding: Making sequences uniform length

4. **Text Classification**: Categorizing text into predefined classes using features learned by neural networks.

5. **Cornell Movie Dialogs Corpus**: A dataset containing conversations from movie scripts, useful for dialogue analysis and understanding conversational patterns.

Deep learning models can automatically learn hierarchical representations from raw text, eliminating the need for manual feature engineering. Common architectures include RNNs, LSTMs, GRUs, and more recently, Transformers.

## Code:

```
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

```

## Output:

```
Epoch 1/5
125/125 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.6338 - loss: 0.6595 - val_accuracy: 0.5855 - val_loss: 0.6911
Epoch 2/5
125/125 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6352 - loss: 0.6541 - val_accuracy: 0.5855 - val_loss: 0.6805
Epoch 3/5
125/125 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6620 - loss: 0.6107 - val_accuracy: 0.5805 - val_loss: 0.7343
Epoch 4/5
125/125 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.7274 - loss: 0.5227 - val_accuracy: 0.5715 - val_loss: 0.7907
Epoch 5/5
125/125 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.7750 - loss: 0.4471 - val_accuracy: 0.5620 - val_loss: 0.8962

Text Classification Accuracy: 82.80%
Sample conversations: 83097
Total question-answer pairs: 10000

```


