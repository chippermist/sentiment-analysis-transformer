import nltk
from data import load_training_data

def name_entity_recognition(raw_data):
  for x in raw_data['review'] :
    tokens = nltk.word_tokenize(x)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    return entities