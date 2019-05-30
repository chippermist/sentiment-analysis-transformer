"""
This file loads the data into a dictionary format instead of pandas
Manual dict construction for easier usage without having model issues
This is not needed if you plan to use pandas 
However, major code changed may be needed if this is not utilized.
"""

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
  reader = csv.DictReader(open(filename, encoding="utf8"))
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