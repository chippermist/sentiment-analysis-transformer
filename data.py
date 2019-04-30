import csv

def load_csv(filename):
    lines = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def load_data(filename):
    return load_csv(filename)

def load_training_data():
  return load_data("dataset/drug-data/drugsTrain_raw.csv")

def load_testing_data():
  return load_data("dataset/drug-data/drugsTest_raw.csv")
