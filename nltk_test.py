import nltk
from data import load_training_data

if __name__ == '__main__':
  raw_data = load_training_data()


  for x in raw_data['review'] :
    tokens = nltk.word_tokenize(x)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    print()
    print() 
    print(entities)
    print() 
    print() 
