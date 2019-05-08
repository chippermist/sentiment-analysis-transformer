import csv
from string import punctuation
from termcolor import colored


# Generates the cleaner data after removing unnecessary characters
def generate_clean_csv(filename, newfilename):
  inputfile = csv.reader(open(filename,'r', encoding="utf8"))

  # replace and overwrite if exists
  # create if doesn't -- writing new clean data
  with open(newfilename, mode='w') as clean_file:
    outputfile = csv.writer(clean_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # initialize i for row count so we can skip first row
    i=0
    for row in inputfile:
      # writing the first row as-is
      if i == 0:
        outputfile.writerow([row[3], row[4]])
        i  += 1
        continue

      # converting sentence to all lowercase
      row[3]   = row[3].lower()
      # replacing characters in the review
      row[3]   = row[3].replace('&#039;', "'")
      row[3]   = row[3].replace('&amp;', '&')
      row[3]   = ''.join([c for c in row[3] if c not in punctuation])

      # changing rating to positive, neutral and negative
      # since the numbers will be pretty pointless overall
      if row[4] in ['0', '1', '2', '3']:
        row[4] = 'negative'
      elif row[4] in ['4','5','6']:
        row[4] = 'neutral'
      else:
        row[4] = 'positive'
      outputfile.writerow([row[3], row[4]])

def call_cleanup():
  print('You are about to create new data files.\n')
  if input('Are you sure you want to run cleanup on data? (y/n) ') == 'y':
    # generate clean csv for training data
    generate_clean_csv('dataset/drug-data/drugsTrain_raw.csv', 'dataset/drug-data/drugsTrain_raw_clean.csv')
    print()
    print('Clean training data -- created.')
    # generate clean csv for testing data
    generate_clean_csv('dataset/drug-data/drugsTest_raw.csv', 'dataset/drug-data/drugsTest_raw_clean.csv')
    print('Clean testing data -- created.')
    print(colored('Cleanup file creation successful.', 'green'))
  else:
    print(colored('Cleanup cancelled.', 'red'))


if __name__ == '__main__':
  call_cleanup()



