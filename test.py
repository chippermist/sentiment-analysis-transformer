from data import load_testing_data, load_training_data
from ner import name_entity_recognition
from cleanup import call_cleanup
from termcolor import colored
import os

if __name__ == '__main__':

  # check if file exists
  exists = os.path.isfile('dataset/drug-data/drugsTrain_raw_clean.csv')
  if not exists:
    print(colored('\nFile does not exist.', 'red'))
    call_cleanup()
  else:
    raw_data = load_training_data()
    print(raw_data.keys())
    print()

    # named entity recognition using nltk
    entities = name_entity_recognition(raw_data)
    print(entities[:2])

    # print(len(raw_data.get('rating')))
    # print(sum(1 for x in raw_data.get('rating') if x == 'neutral'))