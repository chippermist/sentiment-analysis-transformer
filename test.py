from data import load_testing_data, load_training_data
from ner import name_entity_recognition
from cleanup import call_cleanup
from termcolor import colored
from tokenize_data import encode_the_words, encode_the_labels, pad_features
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if __name__ == '__main__':

  # check if file exists
  exists = os.path.isfile('dataset/drug-data/drugsTrain_raw_clean.csv')
  if not exists:
    print(colored('\nFile does not exist.', 'red'))
    call_cleanup()
  else:
    train_data = load_training_data()
    print(train_data.keys())
    print()

    # named entity recognition using nltk
    # entities = name_entity_recognition(train_data)
    # print(entities[:2])

    # print(len(train_data.get('rating')))
    # print(sum(1 for x in train_data.get('rating') if x == 'neutral'))

    encoded_words  = encode_the_words(train_data['review'])
    encoded_labels = encode_the_labels(train_data['rating'])

    features       = pad_features(encoded_words, 80)
    print(encoded_labels[:5])
    print(features[:5,:])


    split_frac = 0.8
    len_feat = len(features)
    print(len(features), len(encoded_labels))

    train_x = features[0:int(split_frac*len_feat)]
    train_y = encoded_labels[0:int(split_frac*len_feat)]
    remaining_x = features[int(split_frac*len_feat):]
    remaining_y = encoded_labels[int(split_frac*len_feat):]
    valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
    valid_y = remaining_y[0:int(len(remaining_y)*0.5)]
    test_x = remaining_x[int(len(remaining_x)*0.5):]
    test_y = remaining_y[int(len(remaining_y)*0.5):]


    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    # dataloaders
    batch_size = 50
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    print('Sample input size: ', sample_x.size()) # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size()) # batch_size
    print('Sample label: \n', sample_y)


    print('Test file', colored('successfully', 'green'), 'run.')