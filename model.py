#imports
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
nltk.download('punkt')
import numpy as np
import tensorflow as tf
import tflearn
import pickle
import json

# reading the intents.json file
with open("intents.json") as file:
    data = json.load(file)

# check if a file of the preprocessed data exists
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

# do this if it doesn't exist
except:
    # initializing empty lists
    words = []  # list of unique words
    labels = [] # list of unique tags
    docs_x = [] # list of patterns
    docs_y = [] # list of tags corresponding to the patterns in docs_x

    # loading the data
    for intent in data["intents"]:  # loop through the intents
        for pattern in intent["patterns"]:  # loop through the patterns
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            # the tag that corresponds to the pattern
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # stemming and vectorization
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))    # remove any duplicate elements and convert back into a list
    labels = sorted(labels)

    # creating a bag of words - one-hot encoding
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        
        # if the word exists we put a 1 if it doesn't we put a 0
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    #converting training data into numpy arrays
    training = np.array(training)
    output = np.array(output)

    #Saving preprocessed data
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)
        

# Building the model using tflearn

# getting rid of previous settings
tf.compat.v1.reset_default_graph()

# define the input shape that I'm expecting for the model
net = tflearn.input_data(shape = [None, len(training[0])])

# 2 fully connected hidden layers with 8 neurons
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)

# output layer
# softmax function gives a probability to each neuron
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")

net = tflearn.regression(net)

# to train the model
# DNN is a type of neural network
model = tflearn.DNN(net)

# load the model if it already exists
try:
    model.load("model.tflearn")

except:
    # training
    # epochs are the number of times the model will go through the same data
    model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)

    # save the model
    model.save("model.tflearn")
