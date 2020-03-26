import os
from PIL import Image
import sys
import numpy as np

input_folder = sys.argv[1]
output_folder = sys.argv[2]

files = os.listdir(input_folder)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for f in files:
    
    path = os.path.join(input_folder, f)
    img = Image.open(path)
    # print(f)
    pixels = img.load()
    
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            r = np.float(pixels[i, j])
    #         ## r = (r - minI)/ (minI - maxI) 
    #         #r = (r * 255.)
            pixels[i,j] = int(r)
        #     pixels[i, j] = pixels[i, j] // 16
            # pixels[i, j] = ((pixels[i, j] - minI)/ (maxI - minI)) * 255.
    rgbimg = Image.new('RGB', img.size)
    rgbimg.paste(img)
    # minI, maxI = rgbimg.getextrema()
    # print(maxI)
    # minI, maxI = img.getextrema()
    # print(maxI)
    rgbimg.save(os.path.join(output_folder, f))
