import csv

def generate_clean_csv(filename, newfilename):
  inputfile = csv.reader(open(filename,'r'))

  with open(newfilename, mode='w') as clean_file:
    outputfile = csv.writer(clean_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # initialize i for row count so we can skip first row
    i=0
    for row in inputfile:
      # writing the first row as-is
      if i == 0:
        outputfile.writerow(row)
        i += 1
        continue

      # replacing characters in the review
      row[3] = row[3].replace('&#039;', "'")
      row[3] = row[3].replace('&amp;', '&')
      outputfile.writerow(row)

# uncomment to use 
# make sure the cleaned data does not already exist in dataset/

if __name__ == '__main__':
  print 'You are about to create new data files.\n'
  if raw_input('Are you sure you want to run cleanup on data? ') == 'y':
    # generate clean csv for training data
    generate_clean_csv('dataset/drug-data/drugsTrain_raw.csv', 'dataset/drug-data/drugsTrain_raw_clean.csv')
    # generate clean csv for testing data
    generate_clean_csv('dataset/drug-data/drugsTest_raw.csv', 'dataset/drug-data/drugsTest_raw_clean.csv')
  



