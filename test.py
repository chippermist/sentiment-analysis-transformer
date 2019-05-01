from data import load_testing_data, load_training_data

if __name__ == '__main__':
  raw_data = load_training_data()
  print raw_data.keys()