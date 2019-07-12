import keras
import numpy as np
import string
import csv
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

def create_data(file):
    with open(file, encoding = 'utf8') as csv_file:
        reader = csv.reader(csv_file, delimiter = ",", quotechar = '"')
        title = []
        description = []
        leafnode = []
        for row in reader:
            title.append(row[0])
            description.append(row[1])
            leafnode.append(row[2])
        return title[1:], description[1:], leafnode[1:]

def clean_data(data):
    cleaned = data[:]
    irrelevant = stopwords.words('english')
    special_characters = ['!','@','?','*','^','#','(',')','/','>','<','.']
    L = len(cleaned)
    for i in range(L):
        text = word_tokenize(cleaned[i])
        for word in text:
            if word in irrelevant or word in string.punctuation:
            #if word in irrelevant or word in special_characters:
                text.remove(word)
        cleaned[i] = ' '.join(text)
    return cleaned


####creating and cleaning the CSV data
file = 'C:\\Users\\Sid\\Documents\\Regalix\\appareldata2.csv'
title, desc, leaf = create_data(file)
desc = clean_data(desc) #the function removes stopwords and punctuations
combined = desc[:] #putting title and description together
for i in range(len(combined)):
    combined[i] = title[i] + ' ' + desc[i]


####creating word to integer mapping
label_indexes = {}  
label_names = list(set(leaf))
for i in range(len(label_names)):
    label_indexes[label_names[i]] = i


####tokenizing
tokenizer = Tokenizer() #Tokenizer(num_words = N), if you want to consider only top N words
tokenizer.fit_on_texts(title)
word_index = tokenizer.word_index  #length = 21776. This is the number of unique words found in the data
vocabulary_size = len(word_index) + 1 

####creating X
X = tokenizer.texts_to_sequences(combined) #each sentence has its words replaced by their corresponding numbers in the vocabulary
X = pad_sequences(X)    #all sentences are now of equal length, 31, which is the longest sentence length in the data

####creating Y
Y = [label_indexes[value] for value in leaf]
Y1h = np_utils.to_categorical(Y)    #one hot representation

####train test splitting
x_train, x_test, y_train, y_test = train_test_split(X, Y1h, test_size = 0.25, random_state = 0)

####training the network
#use small filters and strides
model = Sequential()
model.add(Embedding(input_dim = vocabulary_size, output_dim = 500, input_length = 31))
model.add(Conv1D(10, 5, padding = 'valid', activation = 'relu', strides = 1))
model.add(MaxPooling1D(pool_size = 2))
model.add(LSTM(209, recurrent_dropout = 0.25))
model.add(Dropout(0.30))
model.add(Dense(209, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 25, batch_size = 128)





