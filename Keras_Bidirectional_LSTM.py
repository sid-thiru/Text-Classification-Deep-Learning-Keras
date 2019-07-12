import keras
import numpy as np
import string
import csv
import re
import os
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Activation, GRU, Bidirectional, Input
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras import initializers




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
        filtered_words = [w for w in words if not w.isdigit() and not w in stops and not w in string.punctuation]
        tags = pos_tag(filtered_words)
        cleaned = ''
        for word, tag in tags:
          if tag == 'NN' or tag == 'NNS' or tag == 'VBZ' or tag == 'JJ' or tag == 'RB' or tag == 'NNP' or tag == 'NNPS' or tag == 'RBR':
            cleaned = cleaned + wordnet_lemmatizer.lemmatize(word) + ' '
        output.append(cleaned.strip())
    return output


#data pre-preocessing - creating a tf-idf matrix
def tfidf(data, vocab, ma = 0.6, mi = 0.0001):
    tfidf_vectorize = TfidfVectorizer(max_df = ma, min_df = mi, vocabulary = vocab)
    tfidf_data = tfidf_vectorize.fit_transform(data)
    return tfidf_data



	
#reading data from file and cleaning it
file = 'C:\\Users\\Sid\\Documents\\Regalix\\appareldata2.csv'
title, desc, leaf = create_data(file)
desc = clean_data(desc) 

#combining the title and description together
combined = desc[:] 
for i in range(len(combined)):
    combined[i] = title[i] + ' ' + desc[i]

#label indexes - create an integer-label mapping for each of the 209 classes
label_indexes = {}  
label_names = list(set(leaf))
for i in range(len(label_names)):
    label_indexes[label_names[i]] = i

####tokenizing
tokenizer = Tokenizer() #Tokenizer(num_words = N), if you want to consider only top N words
tokenizer.fit_on_texts(combined)
word_index = tokenizer.word_index  
vocabulary_size = len(word_index) + 1 

####creating X and Y
X = tokenizer.texts_to_sequences(combined) #each sentence has its words replaced by their corresponding numbers in the vocabulary
X = pad_sequences(X)    	#all sentences are now of equal length
size = X.shape[1]
Y = [label_indexes[value] for value in leaf]
Y1h = np_utils.to_categorical(Y)    #one hot representation

#loading GLOVE word embeddings
directory = "C:\\Users\\Sid\\Documents\\Regalix"
embeddings_index = {}
f = open(os.path.join(directory, 'glove.6B.300d.txt'), encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#getting word embeddings for words that are both in GLOVE and the training data
embedding_matrix = np.random.random((vocabulary_size + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#defining an embedding layer. This will be added to the Keras Model
embedding_layer = Embedding(vocabulary_size + 1, 300, weights = [embedding_matrix], input_length = 402, trainable = True)

#Building a keras model
#The embedding layer is the first layer, it is followed by a bidirectional LSTM layer and finally a regular dense output layer
sequence_input = Input(shape = (402,), dtype = 'float64')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(209))(embedded_sequences)
preds = Dense(209, activation = 'softmax')(l_lstm)
model = Model(sequence_input, preds)
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train test split of data
x_train, x_test, y_train, y_test = train_test_split(X, Y1h, test_size = 0.25, random_state = 0)

#fit the model
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 8, batch_size = 128)
