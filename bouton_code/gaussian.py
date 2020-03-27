import sys
import os
import numpy as np
from scipy import ndimage as ndimage
from matplotlib import image as mpimg
import cv2

input_filename = sys.argv[1]
output_filename = sys.argv[2]

img = mpimg.imread(input_filename)

output_image = ndimage.gaussian_filter(img, sigma=2)
output_image = output_image.astype('uint16')
cv2.imwrite(output_filename, output_image)
