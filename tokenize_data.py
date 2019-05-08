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
  encoded_labels = [1 if label =='positive' else -1 if label =='negative' else 0 in labels_split]
  encoded_labels = np.array(encoded_labels)