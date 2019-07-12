import keras
import numpy as np
import string
import csv
import re
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
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

#removing stopwords and punctuations/special characters from the description
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
        filtered_words = [w for w in filtered_words if not w in string.punctuation]
        tags = pos_tag(filtered_words)
        cleaned = ''
        for word, tag in tags:
          if tag == 'NN' or tag == 'NNS' or tag == 'VBZ' or tag == 'JJ' or tag == 'RB' or tag == 'NNP' or tag == 'NNPS' or tag == 'RBR':
            cleaned = cleaned + wordnet_lemmatizer.lemmatize(word) + ' '
        output.append(cleaned.strip())
    return output



#data pre-preocessing - creating a tf-idf matrix
def tfidf(data, ma = 0.6):
    tfidf_vectorize = TfidfVectorizer(max_df = ma, min_df = mi, ngram_range = (1,3))
    tfidf_data = tfidf_vectorize.fit_transform(data)
    return tfidf_data


#reading and cleaning the CSV data
file = 'C:\\Users\\Sid\\Documents\\Regalix\\appareldata2.csv'
title, desc, leaf = create_data(file)
desc = clean_data(desc) #the function removes stopwords and punctuations

#combining the title and description
combined = desc[:] #putting title and description together
for i in range(len(combined)):
    combined[i] = title[i] + ' ' + desc[i]

#creating class label to integer mapping
label_indexes = {}  
label_names = list(set(leaf))
for i in range(len(label_names)):
    label_indexes[label_names[i]] = i

#creating X
X = tfidf(combined)
size = X.shape[1]

#creating Y
#One hot representation is required
Y = [label_indexes[value] for value in leaf]
Y1h = np_utils.to_categorical(Y)    #one hot representation

#train test splitting
x_train, x_test, y_train, y_test = train_test_split(X, Y1h, test_size = 0.25, random_state = 0)

#training the network
#only using dense layers
model = Sequential()
model.add(Dense(418, activation = 'relu', input_shape = (size,)))
model.add(Dropout(0.3))
model.add(Dense(418, activation = 'relu', input_shape = (50306,)))
model.add(Dropout(0.3))
model.add(Dense(209, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 8, batch_size = 128)





