import sys
from matplotlib import image as mpimg
import numpy as np
import scipy.misc
#from scipy.misc import
import cv2
import imageio

bouton_filename = sys.argv[1]
process_filename = sys.argv[2]
output_filename = sys.argv[3]

bouton_img = mpimg.imread(bouton_filename)
process_img = mpimg.imread(process_filename)

nx, ny = bouton_img.shape

output = []
for r in range(nx):
    row = []
    for c in range(ny):
        row.append(0)
    output.append(row)
output = np.asarray(output)

kept = 0
total = 0
for j in range(ny):
    for i in range(nx):
        if bouton_img[i, j] >= 1 and process_img[i, j] > 38:
            kept += 1
            output[i, j] = 255
        if bouton_img[i, j] > 0:
            total += 1

print(kept,'/',total)
#cv2.imwrite(output_filename, output)
imageio.imsave(output_filename, output)