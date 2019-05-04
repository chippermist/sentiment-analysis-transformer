from data import load_testing_data, load_training_data
from ner import name_entity_recognition

if __name__ == '__main__':
  raw_data = load_training_data()
  print(raw_data.keys())
  print()

  # named entity recognition using nltk
  entities = name_entity_recognition(raw_data)
  print(entities[10:])