import sys
import csv

input_filename = sys.argv[1]
output_filename = sys.argv[2]

edges = []
with open(input_filename, 'r') as input_file:
    reader = csv.reader(input_file, delimiter=' ')
    for row in reader:
        v0 = int(row[0])
        v1 = int(row[1])
        if v0 < v1:
            vmin = v0
            vmax = v1
        else:
            vmin = v1
            vmax = v0
        if (vmin, vmax) not in edges:
            edges.append((vmin, vmax))
    input_file.close()

with open(output_filename, 'w') as output_file:
    for e in edges:
        output_file.write(str(e[0]) + ' ' + str(e[1]) + '\n')
    output_file.close()
