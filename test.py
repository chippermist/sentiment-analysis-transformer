# import std libraries
import os
import sys, select
import numpy as np

# import from python files
from data import load_testing_data, load_training_data
from ner import name_entity_recognition
from cleanup import call_cleanup
from termcolor import colored
from tokenize_data import encode_the_words, encode_the_labels, pad_features
from model_operations import predict_from_model

# import third-party libraries
# from torch.utils.data import DataLoader, TensorDataset
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras.callbacks import ReduceLROnPlateau

def main_test():
  # check if file exists
  exists = os.path.isfile('dataset/drug-data/drugsTrain_raw_clean.csv')
  if not exists:
    print(colored('\nFile does not exist.', 'red'))
    call_cleanup()
  else:
    # checking if we are training or loading
    pre_train = len(sys.argv) == 1


    train_data = load_training_data()
    # print(train_data.keys())
    # print()

    # named entity recognition using nltk
    # entities = name_entity_recognition(train_data)
    # print(entities[:2])

    # print(len(train_data.get('rating')))
    # print(sum(1 for x in train_data.get('rating') if x == 'neutral'))

    encoded_words  = encode_the_words(train_data['review'])
    encoded_labels = encode_the_labels(train_data['rating'])
    # print(colored('Shape of labels is', 'blue'), encoded_labels.shape)

    features       = pad_features(encoded_words, 100)
    # print(encoded_labels[:5])
    # print(features[:5,:])

    # print(len(encoded_words))
    split_frac = 0.8
    len_feat    = len(features)
    # print(len(features), len(encoded_labels))

    train_x     = features[0:int(split_frac*len_feat)]
    train_y     = encoded_labels[0:int(split_frac*len_feat)]
    remaining_x = features[int(split_frac*len_feat):]
    remaining_y = encoded_labels[int(split_frac*len_feat):]
    valid_x     = remaining_x[0:int(len(remaining_x)*0.5)]
    valid_y     = remaining_y[0:int(len(remaining_y)*0.5)]
    test_x      = remaining_x[int(len(remaining_x)*0.5):]
    test_y      = remaining_y[int(len(remaining_y)*0.5):]

    print(colored('Shape of X is', 'blue'), train_x.shape)
    print(colored('Shape of Y is', 'blue'), train_y.shape)

    batch_size = 1024

    if pre_train:
      if sys.argv == 2:
        model = load_model(sys.argv[1])
      else:
        model = load_model('lstm_model.h5')
      print(model.summary())
      print(colored('Loaded pre-trained model.', 'green'))
      index = 2
      # print(len(test_x[index]))
      # loss, acc = model.evaluate(test_x, test_y)
      # print('loss:', loss, 'accuracy:', acc)
      # pred = model.predict(test_x)
      # print(pred.argmax())
      # print(predict_from_model(model, test_x[index]))

      print('Would you like to enter a sentence? (y/n) ')
      next_step, o, e = select.select( [sys.stdin], [], [], 50)
      next_step       = sys.stdin.readline().strip()
      if(next_step == 'y'):
        sentence = input('Enter a sentence to classify: ')
        sentence = encode_the_words([sentence])
        sentence = pad_features(sentence, 100)
        print(predict_from_model(model, sentence))

    else:
      # setting some constants
      WORDS  = len(encoded_words)
      LENGTH = 100
      DEPTH  = 32

      # vocab_size = len(encoded_words)

      print(train_x.shape)
      print(type(train_x))

      model = Sequential()
      model.add(Embedding(WORDS, DEPTH, input_length = LENGTH))
      model.add(LSTM(DEPTH))
      model.add(Dense(128, activation='relu'))
      model.add(Dense(3, activation = 'softmax'))
      model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
      reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
      # print(model.summary())
      # model.fit(train_x, train_y, batch_size=128, epochs=5, validation_split=0.1, shuffle=True)
      model.fit(train_x, train_y, batch_size=512, epochs=5, shuffle=True, callbacks=[reduce_lr])

      model.save('lstm_model_saved.h5')
      # del model 
    print('Test file', colored('successfully', 'green'), 'run.')

if __name__ == '__main__':
  main_test()