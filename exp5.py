import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

def download_and_prepare_cornell_dataset():
    """Download and prepare Cornell Movie Dialogs dataset"""
    import urllib.request
    import zipfile
    
    print("Downloading Cornell Movie Dialogs Corpus...")
    
    url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    zip_path = "cornell_movie_dialogs.zip"
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Dataset downloaded successfully!")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("cornell_data")
        
        lines_file = "cornell_data/cornell movie-dialogs corpus/movie_lines.txt"
        conversations_file = "cornell_data/cornell movie-dialogs corpus/movie_conversations.txt"
        
        lines = {}
        print("Parsing movie lines...")
        with open(lines_file, 'r', encoding='iso-8859-1') as f:
            for line in f:
                parts = line.strip().split(' +++$+++ ')
                if len(parts) >= 5:
                    line_id = parts[0]
                    text = parts[4]
                    lines[line_id] = text
        
        conversations = []
        print("Parsing conversations...")
        with open(conversations_file, 'r', encoding='iso-8859-1') as f:
            for line in f:
                parts = line.strip().split(' +++$+++ ')
                if len(parts) >= 4:
                    line_ids = eval(parts[3])  # List of line IDs
                    
                    for i in range(len(line_ids) - 1):
                        question_id = line_ids[i]
                        answer_id = line_ids[i + 1]
                        
                        if question_id in lines and answer_id in lines:
                            question = lines[question_id].strip()
                            answer = lines[answer_id].strip()
                            
                            if 3 <= len(question.split()) <= 15 and 3 <= len(answer.split()) <= 15:
                                conversations.append((question, answer))
        
        os.remove(zip_path)
        
        print(f"Successfully loaded {len(conversations)} conversation pairs from Cornell dataset")
        return conversations[:1000]
        
    except Exception as e:
        print(f"Error downloading Cornell dataset: {e}")
        print("Falling back to sample conversational data...")

def preprocess_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

conversations = download_and_prepare_cornell_dataset()
questions = [preprocess_text(q) for q, a in conversations]
answers = [preprocess_text(a) for q, a in conversations]
answers = ['<start> ' + answer + ' <end>' for answer in answers]

print("Tokenizing text...")
tokenizer = Tokenizer(filters='', lower=True)
tokenizer.fit_on_texts(questions + answers)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")
question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

max_len = max(max(len(seq) for seq in question_sequences), 
              max(len(seq) for seq in answer_sequences))
max_len = min(max_len, 30)

X = pad_sequences(question_sequences, maxlen=max_len, padding='post')
y = pad_sequences(answer_sequences, maxlen=max_len, padding='post')

print(f"Sequence length: {max_len}")
print(f"Training samples: {len(X)}")

def create_seq2seq_model():
    encoder_inputs = Input(shape=(max_len,), name='encoder_inputs')
    encoder_embedding = Embedding(vocab_size, 64, mask_zero=True)(encoder_inputs)
    encoder_lstm = Bidirectional(LSTM(128, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
    state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(max_len,), name='decoder_inputs')
    decoder_embedding = Embedding(vocab_size, 64, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

print("Creating model...")
model = create_seq2seq_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model summary:")
model.summary()

decoder_input_data = np.zeros_like(y)
decoder_input_data[:, 1:] = y[:, :-1]
decoder_target_data = np.expand_dims(y, -1)

print("Training the chatbot...")
history = model.fit(
    [X, decoder_input_data], 
    decoder_target_data,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

def generate_response(question):
    question = preprocess_text(question)
    question_seq = tokenizer.texts_to_sequences([question])
    question_padded = pad_sequences(question_seq, maxlen=max_len, padding='post')
    target_seq = np.zeros((1, max_len))
    if '<start>' in tokenizer.word_index:
        target_seq[0, 0] = tokenizer.word_index['<start>']
    response = []
    for i in range(max_len - 1):
        prediction = model.predict([question_padded, target_seq], verbose=0)
        predicted_id = np.argmax(prediction[0, i, :])
        
        if predicted_id == 0:
            break
        if predicted_id in tokenizer.index_word:
            word = tokenizer.index_word[predicted_id]
            if word == '<end>':
                break
            if word != '<start>':
                response.append(word)
        
        if i < max_len - 1:
            target_seq[0, i + 1] = predicted_id
    
    return ' '.join(response)

print("\n" + "="*50)
print("CHATBOT READY! Testing with movie-related questions:")
print("="*50)

test_questions = [
    "what do you think about this movie",
    "how was this film",
    "tell me about this movie",
    "what is your opinion",
    "how do you rate this"
]

for question in test_questions:
    response = generate_response(question)
    print(f"\nHuman: {question}")
    print(f"Bot: {response}")

print(f"\nTraining completed! The chatbot was trained on {len(conversations)} movie review conversations.")
print("You can now ask movie-related questions!")
