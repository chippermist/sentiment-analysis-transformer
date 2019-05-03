import csv

# importing data into a list
# def load_csv(filename):
#   lines = []
#   with open(filename) as csvfile:
#       reader = csv.DictReader(csvfile)
#       for line in reader:
#           lines.append(line)
#   return lines

# importing data into a dictionary
def load_csv(filename):
  reader = csv.DictReader(open(filename))
  result = {}
  for row in reader:
    for column, value in row.items(): # change to iteritems() for python 2.x
        result.setdefault(column, []).append(value)
  return result

def load_data(filename):
    return load_csv(filename)

def load_training_data():
  return load_data("dataset/drug-data/drugsTrain_raw_clean.csv")

def load_testing_data():
  return load_data("dataset/drug-data/drugsTest_raw_clean.csv")