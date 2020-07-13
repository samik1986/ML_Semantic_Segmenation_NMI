import os
from PIL import Image
import sys


input_image = 'Samik_84_masked.tif'
output_dir = 'results_180830'
output_format = 'StitchedImage_Z084_L001_lossless_X{}Y{}.tif'
size = 512

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

image = Image.open(input_image)
Y, X = image.size
xs, ys = X // size, Y // size
print(xs, ys)


for x in range(xs):
    for y in range(ys):
        x_start, y_start = x * size, y * size
        region = image.crop((y_start, x_start, y_start + size, x_start + size ))
        region.save(os.path.join(output_dir, output_format.format(x_start + 1, y_start + 1)))