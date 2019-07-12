# EXPERIMENTS WITH KERAS

A number of models have been implemented and tested using the Keras libray (using Tensorflow backend). These include - Convolutional Neural Nets (CNN), the 
Long Short term memory (LSTM), and a network with multiple dense layers. LSTMs are a special type of Recurrent Neural Network (RNN). CNNs are better suited to 
work with image data, but there have been reports of good performance with text data as well. For the RNN, CNN, and LSTM models, words are represented as high 
dimensional vectors (using word embeddings). For the network that contains only dense layers, TFIDF features are used. The output layer in all the models is a 
dense layer with the softmax activation function.


### DATA PRE-PROCESSING
* Each product is represented by its 'Title' and 'Description'. There are 209 classes into which a given product can be classified. This is not a multilabel classification problem, so each product has to be assigned one out of the 209 classes only
* The data was cleaned by removing stopwords, punctuations and special characters from the text


### FEATURE EXTRACTION	
* Recurrent Neural Nets (RNN), Convolutional Neural Nets (CNN) and the LSTM models all require the cleaned data to be tokenized
* The tokens retain the same order as in the original text. RNNs and LSTMs are good at learning sequences, so the order of tokens matters
* After tokenization, the sequences are padded with zeros. This is to ensure that all sequences/sentences are of the same length
* Word Embedding is done to represent the words as vectors
* For the network that contains only dense layers, TFIDF representation of the data is used. In this case, there are no word vectors


### KERAS MODELS

#### LSTM
* The first layer is the word embeddings layer. This is a custom embedding layer, created using Keras and based on the training corpus 
* This is followed by an LSTM layer, and finally a dense output layer
* Dropouts are added in between layers to avoid overfitting
* The model achieves 75% precision

#### CNN
* The first layer is the word embeddings layer. This is a custom embedding layer, created using Keras and based on the training corpus 
* This is followed by a convolution layer. Filter size and stride need to be set carefully. The convolutional layer is followed by max pooling. Max pooling is a down sampling technqiue, it reduces the dimensionality of the data
* This is followed by an LSTM layer, and finally a dense output layer
* Dropout is added after the LSTM layer to tackle overfitting
* The model achieves 70% precision
	
#### Dense network
* This consists of two dense layers, followed by a dense output layer
* For the given data, RELU performs better than hyperbolic tangent/sigmoid activation functions
* The input data to the model is in TFIDF format (unlike the other models, where word embeddings are used)
* This model achieves 77% precision
	
#### Bidirectional LSTM
* Bidirectional LSTMs are a variation of LSTMs. Here, two LSTMs are trained on the data. The first LSTM is trained on the data in its original order, and the second LSTM is trained on the data in reversed order. This learning method has found to be particularly useful in sequence classification problems
* The first layer is the word embeddings layer. Instead of a custom Keras embeddings layer, GloVe word vectors are used
* This is followed by the LSTM layers. Keras has a wrapper for creating bidirectional LSTMs
* Finally there is a dense output layer
* Dropouts are added in between layers to avoid overfitting
* The model achieves 71% precision


	