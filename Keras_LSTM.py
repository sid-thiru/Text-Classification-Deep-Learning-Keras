import keras
import numpy as np
import string
import csv
import re
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


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

def clean_data(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    stops = stopwords.words('english')
    nonan = re.compile(r'[^a-zA-Z ]')
    output = []
    for i in range(len(text)):
        sentence = nonan.sub('', text[i])
        words = word_tokenize(sentence.lower())
        filtered_words = [w for w in words if not w.isdigit()]
        filtered_words = [w for w in filtered_words if not w in stops]
        filtered_words = [w for w in filtered_wordswords if not w in string.punctuation]
        tags = pos_tag(filtered_words)
        cleaned = ''
        for word, tag in tags:
          if tag == 'NN' or tag == 'NNS' or tag == 'VBZ' or tag == 'JJ' or tag == 'RB' or tag == 'NNP' or tag == 'NNPS' or tag == 'RBR':
            cleaned = cleaned + wordnet_lemmatizer.lemmatize(word) + ' '
        output.append(cleaned.strip())
    return output


	
#reading the data from file
file = 'C:\\Users\\Sid\\Documents\\Regalix\\appareldata2.csv'

#cleaning the data
title, desc, leaf = create_data(file)
desc = clean_data(desc) #the function removes stopwords and punctuations

#combining the title and description of products
combined = desc[:] #putting title and description together
for i in range(len(combined)):
    combined[i] = title[i] + ' ' + desc[i]


#creating class label to integer mapping
label_indexes = {}  
label_names = list(set(leaf))
for i in range(len(label_names)):
    label_indexes[label_names[i]] = i


#tokenizing
tokenizer = Tokenizer() #Tokenizer(num_words = N), if you want to consider only top N words
tokenizer.fit_on_texts(combined)
word_index = tokenizer.word_index  #length = 21776. This is the number of unique words found in the data
vocabulary_size = len(word_index) + 1 

#creating X
X = tokenizer.texts_to_sequences(combined) #each sentence has its words replaced by their corresponding numbers in the vocabulary
X = pad_sequences(X)    #all sentences are now of equal length, 31, which is the longest sentence length in the data
size = X.shape[1]

#creating Y
#One hot representation is created
Y = [label_indexes[value] for value in leaf]
Y1h = np_utils.to_categorical(Y)    #one hot representation

#train test split
x_train, x_test, y_train, y_test = train_test_split(X, Y1h, test_size = 0.30, random_state = 0)

####training the network
#mask_zero = True tells the network to not train on the zeros in the input vectors
#word embedding layer is the first layer in the network. Each word gets a vector representation of size output_dim
model = Sequential()
model.add(Embedding(input_dim = vocabulary_size, output_dim = 500, input_length = size, mask_zero = True))
model.add(Dropout(0.20))
model.add(LSTM(300, recurrent_dropout = 0.25))
model.add(Dropout(0.40))
model.add(Dense(209, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 8, batch_size = 128)



