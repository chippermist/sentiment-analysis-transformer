from data import load_testing_data, load_training_data
from ner import name_entity_recognition
import os

if __name__ == '__main__':

  # check if file exists
  exists = os.path.isfile('dataset/drug-data/drugsTrain_raw_clean.csv')
  if not exists:
    print('File does not exist. \nPlease run `cleanup.py` to create files.')

  raw_data = load_training_data()
  print(raw_data.keys())
  print()

  # named entity recognition using nltk
  entities = name_entity_recognition(raw_data)
  print(entities[10:])



