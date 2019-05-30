"""

This file tokenizes data for sentiment analysis for LSTM

It created encodings using word lengths and mapping them to a enumerated value.
There are some length parameters that can be changed while calling the methods from testing.
The label encoding has been selected as 3-D however, this can also be changed to [-1, 0, 1]
If so, code changed are required for training and and testing.


"""

from collections import Counter
import numpy as np

def count_total_words(data):
  all_text     = ' '.join(data)
  words        = all_text.split()
  # count all the words using Counter Method
  count_words  = Counter(words)
  total_words  = len(words)
  sorted_words = count_words.most_common(total_words)
  return sorted_words

def vocab_to_int_mapping(data):
  sorted_words = count_total_words(data)
  vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
  # print(vocab_to_int)
  return vocab_to_int

def encode_the_words(data):
  vocab_to_int = vocab_to_int_mapping(data)
  reviews_int  = []
  for review in data:
      r = [vocab_to_int[w] for w in review.split()]
      reviews_int.append(r)
  # print(reviews_int[0:3])
  return reviews_int

def encode_the_labels(labels):
  # this needs to be looked at
  # decision needs to be made to create [1 0 0], [0 1 0] and [0 0 1] or (1, 0, -1)
  # encoded_labels = [1 if label =='positive' else -1 if label =='negative' else 0 for label in labels]
  encoded_labels = []
  for label in labels:
    if label =='positive':
      encoded_labels.append([0,0,1])
    elif label =='negative':
      encoded_labels.append([1,0,0])
    else:
      encoded_labels.append([0,1,0])
  encoded_labels = np.array(encoded_labels, dtype=np.float32)
  return encoded_labels
  # print (encoded_labels[:10])


"""
#
# this is to pad the average review with 0's if it's not the same size
# since LSTM/Attention/Transformer require same length input 
# reviews_int             : from encode_the_words
# padding_length          : user provided based on analysis 
#
"""
def pad_features(reviews_int, padding_length):
  # Return features of review_ints, where each review is padded with 0's or truncated to the input padding_length.
  features = np.zeros((len(reviews_int), padding_length), dtype = int)
  
  for i, review in enumerate(reviews_int):
    review_len = len(review)
    
    if review_len <= padding_length:
      zeroes = list(np.zeros(padding_length-review_len))
      new = zeroes+review
    elif review_len > padding_length:
      new = review[0:padding_length]
    
    features[i,:] = np.array(new)
  
  return features
