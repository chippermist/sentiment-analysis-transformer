import nltk
from nltk.corpus import stopwords
from data import load_training_data

# creating a list of trees for each sentence in raw_data
def name_entity_recognition(raw_data):
  result_entities = []
  stop_words = stopwords.words('english')
  for x in raw_data['review'][:10] :
    tokens   = nltk.word_tokenize(x)
    # removing all stopwords from tokens
    tokens   = [word for word in tokens if word not in stop_words]
    tagged   = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    result_entities.append(entities)
  return result_entities