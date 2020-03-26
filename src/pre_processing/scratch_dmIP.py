import os
from PIL import Image
import sys
import numpy as np
import cv2
import fnmatch

input_folder = '/home/samik/ProcessDet/wdata/train/images/'
# input_folder = '/home/samik/ProcessDet/wdata/Process/annotV/'
input_folder2 = '/home/samik/ProcessDet/wdata/test/images/'
files = os.listdir(input_folder2)
files2 = os.listdir(input_folder2)

for f1 in files:
    # if f1.replace('png', 'tif') not in files:
    # print("cp " + os.path.join('/nfs/data/main/M32/PMD1605_Annotations/dataMBA/cropped_annotation/Process/rawD', f1.replace('_img.png', '.tif')) + " /home/samik/ProcessDet/wdata/train/images/")
    # os.system("cp " + os.path.join('/home/samik/ProcessDet/wdata/TrainingData/rawD/', f1) + " /home/samik/ProcessDet/wdata/test/images/")
    # img = np.zeros((512,512), dtype = np.uint8)
    imgRaw =  cv2.imread(os.path.join('/home/samik/ProcessDet/wdata/test/images', f1), cv2.IMREAD_UNCHANGED)
    print(f1)
    if fnmatch.fnmatch(f1, '*R.tif'):
        img = imgRaw[:,:,2]
        print(f1)
    if fnmatch.fnmatch(f1, '*G.tif'):
        img = imgRaw[:,:,1]
        print(f1)
    # dmimg = Image.fromarray(img)
    # # rgbimg.paste(img)
    cv2.imwrite("/home/samik/ProcessDet/wdata/test/dmIP/" + f1, img)
    