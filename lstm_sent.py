#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


# In[2]:


from cleanup import *
from data import *
from ner import *
from tokenize_data import *


# In[7]:


train_data = load_training_data()
print(type(train_data))
sample = train_data['review'][:1]
print(sample)


# In[16]:


max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(train_data['review'])
X = tokenizer.texts_to_sequences(train_data['review'])
print(type(X))
X = pad_sequences(X)
print(X.shape)
print(type(X.shape))
print(X.shape[1])


# In[24]:


X_train = encode_the_words(train_data['review'])
print(X_train[:1])
Y_train = train_data['rating']
import pandas as pd
Y = pd.get_dummies(Y_train).values


# In[29]:


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[31]:


batch_size = 32
model.fit(X, Y, epochs = 7, batch_size=batch_size)


# In[ ]:




