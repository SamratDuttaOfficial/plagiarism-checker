import nltk
from nltk.stem.lancaster import LancasterStemmer #used to stem the words.
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("file.json") as file:
    dataset = json.load(file)
#Opens the python Dictionary saved in the JSON file.

list_words = []
labels = []
docs_x = [] #List of all the question_patterns.
docs_y = [] #List of all the tags for specific Texts.

for intent in dataset["intents"]:
    for pattern in intent["texts"]: #Stems the words. Finds the root of the word. Removes extra characters and symbols to find the root word. 
        split_words = nltk.word_tokenize(pattern) #Tokenizes the words. Breakes the words in the places of spaces and returns a list of all the words in it.
        list_words.extend(split_words) #Using instead of looping and adding one word at a time in the list. It just extends the list untill all the words are in it.
        docs_x.append(split_words) #Adding the pattern of words inside docs_x list.
        docs_y.append(intent["tag"]) #For each pattern, it says what Tag it is a part of.

    if intent["tag"] not in labels:
        labels.append(intent["tag"]) #Adds all the tags in the labels list.

list_words = [stemmer.stem(w.lower()) for w in list_words if w != "?"] #Lower cases all the words in the list_words list. 
list_words = sorted(list(set(list_words))) #Makes a set of the words to remove duplicate. This gives us the actual vocabulary size of the intent. Then converts it back to list and sorts it. 

labels = sorted(labels) #Sorts the labels where the tags are stored.

training = [] #contains the bag of words.
output = [] #The output list to choose the right tag for the output.

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = [] #Makes a list of all the words and tells if those words are occuring to find a pattern.
    #We are representing each sentence with a list the length of the amount of words in our models vocabulary. 
    #Each position in the list will represent a word from our vocabulary. 
    #If the position in the list is a 1 then that will mean that the word exists in our sentence, if it is a 0 then the word is nor present. 

    split_words = [stemmer.stem(w.lower()) for w in doc] #Stems all the words inside docs_x list.

    for w in list_words:
        if w in split_words:
            bag.append(1) #we are putting 1 in the bag of words list for the word (already present in the vocabulary list_words) present in the pattern and 0 if it is not.
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1 #We are looking in the labels list to find were the tag is in that list. We are making that value 1 in the output row. 
    #We will create output lists which are the length of the amount of labels/tags we have in our dataset. 
    #Each position in the list will represent one distinct label/tag, a 1 in any of those positions will show which label/tag is represented.

    training.append(bag) #We are putting the bag of words in the training list. 
    output.append(output_row) #We are putting the output_row list in the output list. 


training = numpy.array(training) #we will convert our training data and output to numpy arrays.
output = numpy.array(output)
tensorflow.reset_default_graph() #Resetting all the previous stuffs in the graph.

net = tflearn.input_data(shape=[None, len(training[0])]) #This finds the input shape that we are expecting for the model. Each training input is gonna be of the same length, so, 0.
net = tflearn.fully_connected(net, 8) #Hidden layer with 15 neurons.
net = tflearn.fully_connected(net, 8) #Another hidden layer with 15 neurons.
net = tflearn.fully_connected(net, 8) #Another hidden layer with 15 neurons.
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #length will be how many labels (how many tags) we have. Give us probability for each neuron in the network. The higest probability will be the output.
net = tflearn.regression(net)

model = tflearn.DNN(net) #Model gets trained.

try:
    model.predict("model.tflearn")
    model.load()
except:
    model.fit(training, output, n_epoch=10, batch_size=8, show_metric=True) #Fitting our data to the model. The number of epochs we set is the amount of times that the model will see the same information while training.
    model.save("model.tflearn") #we can save it to the file model.tflearn for use in other scripts.


def bag_of_words(s, list_words): #bag_of_words function will transform our string input to a bag of words using our created words list
    bag = [0 for _ in range(len(list_words))]

    inp_str_words = nltk.word_tokenize(s)
    inp_str_words = [stemmer.stem(word.lower()) for word in inp_str_words]

    for search_element in inp_str_words:
        for i, w in enumerate(list_words):
            if w == search_element:
                bag[i] = 1 
            
    return numpy.array(bag)


def check():


    docs_all=[]        
    set_sents = []

    print("Type an answer (Enter 'quit' to stop)")
    while True:
        inp = input("You: ")
        inp=inp.lower()
        if inp == "quit":
            break

        results = model.predict([bag_of_words(inp, list_words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        split_sent_inp = nltk.sent_tokenize(inp)
        for tg in dataset["intents"]:
            if tg['tag'] == tag:
                response = tg['texts']
        #something ent wrong after this:
        split_sent_resp = nltk.sent_tokenize(response)
        print("Plagarism found with the text: ")
        print(response)
        docs_all.extend(split_sent_inp)
        docs_all.extend(split_sent_resp)
        set_sents.extend(docs_all)
        len_resp = len(split_sent_resp)
        set_sents = [stemmer.stem(sents) for sents in set_sents]
        len_initial=len(set_sents)
        set_sents = sorted(list(set(set_sents)))
        len_final=len(set_sents)
        len_difference = len_initial - len_final
        percent = (((len_resp - len_difference)/(len_resp))*100)
        plag_percent=100-percent
        print("The percent of plagarism in the document is: ",plag_percent)

        docs_all=[]
        set_sents=[]
        
        #– Get some input from the user
        #– Convert it to a bag of words
        #– Get a prediction from the model
        #– Find the most probable class
        #– Pick a response from that class
        #- Empty the lists again

check()
