import sys
import os
from PIL import Image
import re


input_folder = 'results_180830/'
output_folder = ''

files = sorted(os.listdir(input_folder))

X_range, Y_range = 8192, 11264
# how many sections
Z_range = 1
# how many tiles in each section
tile_cnt = 352


if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for z in range(Z_range):
    start = z * tile_cnt
    end = start + tile_cnt
    # print(z, ':')
    img = Image.new('L', (X_range, Y_range))

    for i in range(start, end):
        filename = os.path.join(input_folder, files[i])
        print(filename)
        tile = Image.open(filename)
        result = re.search('.*X(.*)Y(.*).tif', files[i])
        cy = int(result.group(1))
        cx = int(result.group(2))
        img.paste(tile, (cx - 1, cy - 1))

    output_path = os.path.join(output_folder, '{0:03d}'.format(z + 1) + '.tif')
    img.save(output_path, compression='tiff_deflate')

