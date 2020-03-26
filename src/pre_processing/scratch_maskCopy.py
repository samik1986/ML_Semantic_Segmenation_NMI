import os
from PIL import Image
import sys
import numpy as np
import cv2
import fnmatch

input_folder = '/home/samik/ProcessDet/wdata/train/masks2m/'
# input_folder = '/home/samik/ProcessDet/wdata/Process/annotV/'
input_folder2 = '/home/samik/ProcessDet/wdata/test/images/'
files = os.listdir(input_folder)
files2 = os.listdir(input_folder2)

for f1 in files:
    # if f1.replace('png', 'tif') not in files:
    os.system("cp " + os.path.join('/home/samik/ProcessDet/wdata/TrainingData2/rawD/', f1) + " /home/samik/ProcessDet/wdata/train/images/")
    # os.system("cp " + os.path.join('/home/samik/ProcessDet/wdata/TrainingData/dmV/', f1) + " /home/samik/ProcessDet/wdata/test/dmIP/")
    # img = np.zeros((512,512,3), dtype = np.uint8)
    # imgRaw =  cv2.imread(os.path.join('/nfs/data/main/M32/PMD1605_Annotations/dataMBA/cropped_annotation/Process/rawD', f1), cv2.IMREAD_UNCHANGED)
    # # print(f1)
    # if fnmatch.fnmatch(f1, '*R.tif'):
    #     img[:,:,0] = imgRaw
    #     print(f1)
    # if fnmatch.fnmatch(f1, '*G.tif'):
    #     img[:,:,1] = imgRaw
    #     print(f1)
    # rgbimg = Image.fromarray(img)
    # # rgbimg.paste(img)
    # rgbimg.save("/home/samik/ProcessDet/wdata/train/images/" + f1)
    