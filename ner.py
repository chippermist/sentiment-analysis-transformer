import nltk
from data import load_training_data

def name_entity_recognition(raw_data):
  result_entities = []
  for x in raw_data['review'][:10] :
    tokens   = nltk.word_tokenize(x)
    tagged   = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    result_entities.append(entities)
  return result_entities