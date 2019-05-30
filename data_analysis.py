"""
This file just performs data analysis on the input csv file
Usage:
  This can be used to find out average sentence length in words and then perform sentiment analysis encoding
  The word lengths can significantly affect the quality of encoding
  It is recommended to run this on a new dataset if you're not using UCI Medical Data that has been cleaned up.
"""
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tokenize_data import encode_the_words, encode_the_labels
from data import load_testing_data, load_training_data

def analyze_review_length(reviews_int):
  reviews_len = [len(x) for x in reviews_int]
  pd.Series(reviews_len).hist()
  # plt.plot(reviews_len)
  plt.show()
  pd.Series(reviews_len).describe()

def perform_data_analysis(raw_data):
  reviews_int = encode_the_words(raw_data.get('review'))
  analyze_review_length(reviews_int)

if __name__ == '__main__':
  # simple testing code
  raw_data    = load_training_data()
  # print(raw_data.get('review')[:10])
  perform_data_analysis(raw_data)