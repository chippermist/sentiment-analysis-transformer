from data import load_testing_data, load_training_data
from ner import name_entity_recognition
from cleanup import call_cleanup
from termcolor import colored
from tokenize_data import encode_the_words, encode_the_labels, pad_features
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sentiment_model_define_class import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM


if __name__ == '__main__':

  # check if file exists
  exists = os.path.isfile('dataset/drug-data/drugsTrain_raw_clean.csv')
  if not exists:
    print(colored('\nFile does not exist.', 'red'))
    call_cleanup()
  else:
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

    features       = pad_features(encoded_words, 80)
    # print(encoded_labels[:5])
    # print(features[:5,:])

    print(len(encoded_words))
    split_frac = 0.8
    len_feat = len(features)
    # print(len(features), len(encoded_labels))

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
    batch_size = 1024
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    # print('Sample input size: ', sample_x.size()) # batch_size, seq_length
    # print('Sample input: \n', sample_x)
    # print()
    # print('Sample label size: ', sample_y.size()) # batch_size
    # print('Sample label: \n', sample_y)

    vocab_size = 161297
    max_length = 80
    dim = 3
    embedding_matrix = np.random.randn(vocab_size + 1, dim) * 0.01

    print(train_x.shape)
    print(type(train_x))

    model = Sequential()
    model.add(Embedding(vocab_size + 1, dim, weights=[embedding_matrix], input_length=max_length))
    model.add(Dropout(0.4))
    model.add(LSTM(128))
    model.add(Dense(3))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # filepath = "./models/lstm-{epoch:02d}-{loss:0.3f}-{acc:0.3f}-{val_loss:0.3f}-{val_acc:0.3f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
    print(model.summary())
    # model.fit(train_x, train_y, batch_size=128, epochs=5, validation_split=0.1, shuffle=True)
    model.fit(train_x, train_y, batch_size=128, epochs=5)

    index = 0
    # pred = model.predict(test_x[index])
    # print(pred.argmax(),test_y[index])
    # # Instantiate the model w/ hyperparams
    # vocab_size = 1 # +1 for the 0 padding
    # output_size = 1
    # embedding_dim = 50 #400
    # hidden_dim = 2    #256
    # n_layers = 2
    # net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    # print(net)


    # # loss and optimization functions
    # lr=0.001

    # criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)


    # # training params

    # epochs = 2 # 3-4 is approx where I noticed the validation loss stop decreasing

    # counter = 0
    # print_every = 100
    # clip=5 # gradient clipping

    # net.train()

    # # train for some number of epochs
    # for e in range(epochs):
    #     # initialize hidden state
    #     h = net.init_hidden(batch_size)
    #     # batch loop
    #     for inputs, labels in train_loader:
    #         counter += 1

    #         # Creating new variables for the hidden state, otherwise
    #         # we'd backprop through the entire training history
    #         h = tuple([each.data for each in h])

    #         # zero accumulated gradients
    #         net.zero_grad()

    #         # get the output from the model
    #         inputs = inputs.type(torch.LongTensor)
    #         print(h)
    #         output = net(inputs, h)

    #         # calculate the loss and perform backprop
    #         loss = criterion(output.squeeze(), labels.float())
    #         loss.backward()
    #         # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    #         nn.utils.clip_grad_norm_(net.parameters(), clip)
    #         optimizer.step()

    #         # loss stats
    #         if counter % print_every == 0:
    #             # Get validation loss
    #             val_h = net.init_hidden(batch_size)
    #             val_losses = []
    #             net.eval()
    #             for inputs, labels in valid_loader:

    #                 # Creating new variables for the hidden state, otherwise
    #                 # we'd backprop through the entire training history
    #                 val_h = tuple([each.data for each in val_h])

    #                 inputs = inputs.type(torch.LongTensor)
    #                 output, val_h = net(inputs, val_h)
    #                 val_loss = criterion(output.squeeze(), labels.float())

    #                 val_losses.append(val_loss.item())

    #             net.train()
    #             print("Epoch: {}/{}...".format(e+1, epochs),
    #                   "Step: {}...".format(counter),
    #                   "Loss: {:.6f}...".format(loss.item()),
    #                   "Val Loss: {:.6f}".format(np.mean(val_losses)))


    print('Test file', colored('successfully', 'green'), 'run.')