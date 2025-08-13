import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# --- Ensure NLTK data is available ---
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    # Some newer NLTK versions require punkt_tab
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass
    for pkg in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)

_ensure_nltk()

lemmatizer = WordNetLemmatizer()

# --- Load intents ---
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# --- Build vocabulary and classes ---
words = []
classes = []
documents = []
ignore_chars = {"?", "!", ".", ","}

for intent in intents.get("intents", []):
    tag = intent.get("tag", "")
    for pattern in intent.get("patterns", []):
        # tokenize
        tokens = nltk.word_tokenize(pattern)
        tokens = [t.lower() for t in tokens]
        words.extend(tokens)
        documents.append((tokens, tag))
        if tag not in classes:
            classes.append(tag)

# lemmatize + remove punctuation marks
words = [lemmatizer.lemmatize(w) for w in words if w not in ignore_chars]
words = sorted(set(words))
classes = sorted(set(classes))

# save vocabulary/classes for inference later
with open("words.pkl", "wb") as f:
    pickle.dump(words, f)
with open("classes.pkl", "wb") as f:
    pickle.dump(classes, f)

# --- Build training data ---
training = []
output_empty = [0] * len(classes)

for token_list, tag in documents:
    # normalize the pattern tokens to align with 'words' processing
    token_list = [lemmatizer.lemmatize(w.lower()) for w in token_list if w not in ignore_chars]

    # bag of words
    bag = [1 if w in token_list else 0 for w in words]

    # one-hot label
    output_row = output_empty.copy()
    output_row[classes.index(tag)] = 1

    training.append([bag, output_row])

# shuffle for training
random.shuffle(training)

# split into X (bags) and Y (labels) and convert to arrays
train_x = np.array([t[0] for t in training], dtype=np.float32)  # shape: (N, len(words))
train_y = np.array([t[1] for t in training], dtype=np.float32)  # shape: (N, len(classes))

# --- Build and train the model ---
model = Sequential()
model.add(Dense(128, input_shape=(len(words),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation="softmax"))

# NOTE: TF 2.14's SGD doesn't support 'weight_decay' kwarg; use this:
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model to current folder (no extra subfolder needed)
model.save("chatbot_model.keras")

print("Training complete. Saved model: chatbot_model.keras, words.pkl, classes.pkl")
