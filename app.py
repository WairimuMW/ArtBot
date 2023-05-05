from flask import Flask, render_template, request, jsonify
from model import *
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import random


# processing user input
def bag_of_words(s, words):
    # create an empty bag of words list
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    # generating the bag of words
    for sent in s_words:
        for i, w in enumerate(words):
            if w == sent:
                bag[i] = 1
    
    return np.array(bag)

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/get')
def chat():
    while True:
        inp = request.args.get('msg')
        inp = inp.lower()
        
        # passing the processed user input to the model
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results) # index of the largest value in our list
        tag = labels[results_index]
        
        if results[results_index] >= 0.7:    # 70% probability
            # getting a response from the tag
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
            
            response = random.choice(responses)
        
        else:
            response = "Sorry, I didn't quite get that."
    
        return str(response)

# run the flask app
if __name__ == "__main__":
	app.run()

