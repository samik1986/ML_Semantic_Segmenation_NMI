import os
import sys


path = sys.argv[1]
output_csv = 'folds4.csv'

files = os.listdir(path)

with open(output_csv, 'w') as csvf:
    csvf.write(',fold\n')
    for id, f in enumerate(files):
        csvf.write(','.join([f, str(id % 4)]) + '\n')



