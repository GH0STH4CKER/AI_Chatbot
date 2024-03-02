import json
import random
import pickle
import numpy as np
from colorama import Fore,init
init()
grn = Fore.GREEN
blu = Fore. BLUE

import logging
import os

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents2.json').read())

words = pickle.load(open(r'C:\Users\Asus\Documents\Python_Projects\AI_chatbot\words1.pkl', 'rb'))
classes = pickle.load(open(r'C:\Users\Asus\Documents\Python_Projects\AI_chatbot\classes1.pkl', 'rb'))
model = load_model(r'C:\Users\Asus\Documents\Python_Projects\AI_chatbot\chatbotmodel2.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]),verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text)
    res = get_response(ints, intents)
    return res


# Test the chatbot
while True:
    print(grn+'')
    message = input("You: ")

    if message.lower() in ['quit','exit','close']:
        break
    elif  any(str(a) + symbol + str(b) in message for a in [1,2,3,4,5,6,7,8,9,0] for b in [1,2,3,4,5,6,7,8,9,0] for symbol in ['*','/','-','+','**']):
        index = []
        for i in range(len(message)):
            if message[i].isnumeric():
                index.append(i)
        try:
            ans = eval(message[index[0]:index[-1]+1])
        except Exception as e:
            print(blu+'Bot: Error ! ',e)
        else:
            print(blu+'Bot: Answer is ',ans)
    else:
        print(blu+"Bot:", chatbot_response(message))
