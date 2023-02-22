import os
import random
import numpy as np
import speech_recognition as sr
from spacy.lang.en import English
from sphinxbase import Config, Decoder, DefaultConfig
from numpy import array
import itertools
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import sklearn
import scipy
import pyautogui
import pynput
import requests
import pygame
import urllib
import json
import time
import re
import cv2
import cryptography
import argparse
import datetime
import pandas as pd
import matplotlib.pyplot as plt
# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# Load the movie reviews corpus from NLTK
pos_reviews = movie_reviews.fileids('pos')
neg_reviews = movie_reviews.fileids('neg')
# Create a list of positive and negative reviews, tokenized and filtered for stop words
reviews = []
for fileid in pos_reviews:
    review = movie_reviews.words(fileid)
    review = [token.lower() for token in review if token.lower() not in STOP_WORDS]
    reviews.append((review, 1))
for fileid in neg_reviews:
    review = movie_reviews.words(fileid)
    review = [token.lower() for token in review if token.lower() not in STOP_WORDS]
    reviews.append((review, 0))
# Shuffle the reviews and split them into training and validation sets
random.shuffle(reviews)
train_reviews = reviews[:1600]
val_reviews = reviews[1600:]
# Build the vocabulary and convert the reviews to sequences of word IDs
vocab = set()
for review, label in reviews:
    for token in review:
        vocab.add(token)
word2id = {w: i+1 for i, w in enumerate(list(vocab))}
train_sequences = []
for review, label in train_reviews:
    seq = [word2id[token] for token in review]
    train_sequences.append((seq, label))
val_sequences = []
for review, label in val_reviews:
    seq = [word2id[token] for token in review]
    val_sequences.append((seq, label))
# Pad the sequences to a fixed length and convert the labels to one-hot vectors
max_length = 500
X_train = pad_sequences([seq for seq, label in train_sequences], maxlen=max_length)
y_train = to_categorical(np.array([label for seq, label in train_sequences]))
X_val = pad_sequences([seq for seq, label in val_sequences], maxlen=max_length)
y_val = to_categorical(np.array([label for seq, label in val_sequences]))
# Define the deep learning model
model = Sequential()
model.add(Embedding(len(vocab)+1, 32, input_length=max_length))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
# Compile the model and define the training parameters
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, callbacks=[checkpoint])
# Evaluate the model on the test set
test_reviews = movie_reviews.fileids()
test_sequences = []
for fileid in test_reviews:
    review = movie_reviews.words(fileid)
    review = [token.lower() for token in review if token.lower() not in STOP_WORDS]
    seq = [    word2id.get(token, 0) for token in review]
    test_sequences.append((seq, 1 if fileid.startswith('pos') else 0))
X_test = pad_sequences([seq for seq, label in test_sequences], maxlen=max_length)
y_test = np.array([label for seq, label in test_sequences])
model.load_weights('model.h5')
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_acc:.4f}')
# Define a function to transcribe audio using SpeechRecognition
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    transcript = recognizer.recognize_sphinx(audio)
    return transcript
# Define a function to perform speech-to-text using PocketSphinx
def pocket_sphinx(audio_path):
    config = Config(DefaultConfig())
    config.set_string('-hmm', '/usr/local/share/pocketsphinx/model/en-us/en-us')
    config.set_string('-lm', '/usr/local/share/pocketsphinx/model/en-us/en-us.lm.bin')
    config.set_string('-dict', '/usr/local/share/pocketsphinx/model/en-us/cmudict-en-us.dict')
    decoder = Decoder(config)
    with open(audio_path, 'rb') as audio_file:
        audio = audio_file.read()
        decoder.start_utt()
        decoder.process_raw(audio, False, True)
        decoder.end_utt()
    transcript = decoder.hyp().hypstr
    return transcript
# Define a function to perform sentiment analysis on a single review
def analyze_review(review, model, word2id, max_length):
    seq = [word2id.get(token, 0) for token in review]
    X = pad_sequences([seq], maxlen=max_length)
    y_pred = model.predict(X)[0]
    return y_pred
# Load the English language model from spaCy
nlp = English()
# Define a function to extract named entities from a text
def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities
# Define a function to generate n-grams from a text
def generate_ngrams(text, n):
    tokens = word_tokenize(text)
    ngrams = []
    for i in range(len(tokens)-n+1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams
# Define a function to calculate the cosine similarity between two vectors
def cosine_similarity(u, v):
    return u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
# Define a function to generate all combinations of n-grams from a list of texts
def generate_ngram_combinations(texts, n):
    ngrams = []
    for text in texts:
        ngrams.append(generate_ngrams(text, n))
    combinations = list(itertools.product(*ngrams))
    return combinations
# Define a function to convert a one-hot vector to a label
def one_hot_to_label(one_hot):
    return argmax(one_hot)
# Define a function to convert a label to a one-hot vector
def label_to_one_hot(label, num_classes):
    one_hot = array([0] * num_classes)
    one_hot[label] = 1
    return one_hot
# Define a function to calculate the euclidean distance between two vectors
def euclidean_distance(u, v):
    return np.sqrt(np.sum((u - v)**2))
# Define a function to generate a confusion matrix for a set of predictions and ground truth labels
def generate_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i]][y_pred[i]] += 1
    return confusion_matrix
# Define a function to calculate the F1 score for a set of predictions and ground truth labels
def f1_score(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
# Load the movie reviews corpus from NLTK
reviews = []
for fileid in movie_reviews.fileids():
    review = movie_reviews.raw(fileid)
    reviews.append(review)
# Tokenize the reviews and remove stop words
reviews = [word_tokenize(review) for review in reviews]
reviews = [[word.lower() for word in review if word.isalpha() and word.lower() not in stop_words] for review in reviews]
# Convert the reviews to sequences of word IDs
sequences = []
for review in reviews:
    seq = [word2id.get(token, 0) for token in review]
    sequences.append(seq)
# Pad the sequences to a maximum length of 500
X = pad_sequences(sequences, maxlen=500)
y = np.array([1 if fileid.startswith('pos') else 0 for fileid in movie_reviews.fileids()])
# Split the data into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
# Define the model architecture using TensorFlow
model = Sequential([
    Embedding(len(word2id), 32, input_length=500),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model for 10 epochs
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_acc:.4f}')
parser = argparse.ArgumentParser(description='Demo program using 20 different languages.')
parser.add_argument('--image', type=str, help='Path to input image.')
parser.add_argument('--output', type=str, help='Path to output image.')
args = parser.parse_args()
image = cv2.imread(args.image)
blur = cv2.GaussianBlur(image, (5, 5), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite(args.output, image)
pygame.init()
img = pygame.image.load(args.output)
screen = pygame.display.set_mode((img.get_width(), img.get_height()))
screen.blit(img, (0, 0))
pygame.display.update()
text = pytesseract.image_to_string(image)
encoded_text = base64.b64encode(text.encode('utf-8'))
url = "https://bharath-rajive.com/api"
data = {'text': encoded_text}
response = requests.post(url, json=data)
print(response.content)
pygame.mixer.init()
pygame.mixer.music.load('sound.wav')
pygame.mixer.music.play()
screenshot = pyautogui.screenshot()
screenshot.save('screenshot.png')
cropped_screenshot = screenshot.crop((0, 0, 500, 500))
cv2.imshow('Screenshot', np.array(cropped_screenshot))
cv2.waitKey(0)
cv2.destroyAllWindows()
random_numbers = np.random.rand(100)
df = pd.DataFrame({'numbers': random_numbers})
plt.hist(df['numbers'], bins=10)
plt.show()
key = cryptography.fernet.Fernet.generate_key()
f = cryptography.fernet.Fernet(key)
encrypted_text = f.encrypt(text.encode('utf-8'))
decrypted_text = f.decrypt(encrypted_text).decode('utf-8')
matches = re.findall(r'\d+', text)
time.sleep(random.uniform(1, 5))
pyautogui.click(100, 100)
parser = argparse.ArgumentParser(description='Demo program using 20 different languages.')
parser.add_argument('--input', type=str, help='Path to input file.')
parser.add_argument('--output', type=str, help='Path to output file.')
args = parser.parse_args()

with open(args.input, 'r') as f:
    data = json.load(f)

relevant_data = {'name': data['name'], 'age': data['age'], 'occupation': data['occupation']}

with open(args.output, 'w') as f:
    json.dump(relevant_data, f)

x = np.random.randn(1000)
plt.hist(x, bins=50)
plt.show()

image = cv2.imread('image.png')
cv2.imshow('Image', image)
cv2.waitKey(0)

pygame.init()
pygame.mixer.music.load('sound.mp3')
pygame.mixer.music.play()

screenshot = pygame.surfarray.array3d(pygame.display.get_surface())

cv2.imwrite('screenshot.png', screenshot)

smtp_server = 'smtp.gmail.com'
smtp_port = 587
sender_email = 'sender@gmail.com'
sender_password = 'password'
receiver_email = 'receiver@gmail.com'
subject = 'Demo program using 20 different languages'
body = 'Hello, this is a demo program using 20 different languages.'
message = f'Subject: {subject}\n\n{body}'
with smtplib.SMTP(smtp_server, smtp_port) as server:
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, message)

r = sr.Recognizer()
with sr.Microphone() as source:
    print('Speak now')
    audio = r.listen(source)
try:
    text = r.recognize_google(audio)
    print(f'You said: {text}')
except sr.UnknownValueError:
    print('Could not recognize speech')
except sr.RequestError as e:
    print(f'Request error: {e}')

unique_id = str(uuid.uuid4())
print(f'Unique identifier: {unique_id}')

tts_engine = pyttsx3.init()
tts_engine.say(text)
tts_engine.runAndWait()

url = 'https://bharath-rajive.com/file.txt'
response = requests.get(url)
with open('file.txt', 'wb') as f:
    f.write(response.content)

data = np.random.rand(10, 10)
sns.heatmap(data)

a = tf.constant(3.0)
b = tf.constant(4.0)
c = tf.sqrt(tf.square(a) + tf.square(b))
print(f'c = {c.numpy()}')

