import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# Define ignore characters for text cleaning
ignore_chars = ['?', '!', '.', ',']

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words if word not in ignore_chars]

    return sentence_words
def bag_of_words(sentence):
    words = pickle.load(open("model/words.pkl", "rb"))

    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    classes = pickle.load(open("model/classes.pkl", "rb"))
    model = load_model("model/chatbot_model.keras")
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []        

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def get_response(intents_list):
    intents_json = json.load(open("model/intents.json"))
    
    # If no intents were predicted, return fallback response
    if not intents_list:
        return "I'm sorry, I couldn't understand your question. Please visit our website at digitalcommerce360.pk for more information and support."
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    
    # If no matching intent found, return fallback response
    return "I'm sorry, I don't have a specific answer for that question. Please visit our website at digitalcommerce360.pk for comprehensive information about digital commerce in Pakistan."