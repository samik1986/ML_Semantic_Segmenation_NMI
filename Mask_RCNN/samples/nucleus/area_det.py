import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import keras.backend as K
import cv2
from matplotlib import pyplot as plt
import scipy.ndimage as misc
from skimage.io import imread
from skimage.color import rgb2gray

def imread_fast(img_path):
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    print("./kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 1")
    # os.system("cd /home/samik/v7_A_6-01832N/bin/Linux-x86-64-gcc/")
    err_code = os.system("./kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 16")
    print(err_code)
    # os.system("cd /home/samik/Mask_RCNN/samples/nucleus")
    img = imread('temp/'+base+'.tif')
    os.system("rm temp/"+base_C+'.tif')
    return img

os.environ['OPENCV_IO_ENABLE_JASPER']= '1'
filePath = '/home/samik/mnt/temp/M25/mba_converted_imaging_data/PTM850&849/PTM849/'
outDir = 'PTM849_a/'
outDir1 = 'PTM849_a1/'
fileList1 = os.listdir(filePath)
fileList2 = os.listdir(outDir)
for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fichier.endswith("_lossy.jp2")):
        fileList1.remove(fichier)
#print(fileList1)

for files in fileList1:
	if files not in fileList2:
	    print(files)
        image = imread_fast(os.path.join(filePath, files))
        w, h, c= image.shape
        idx = image > 255.
        image[idx] = 255.
        image1 = image.astype(np.uint8)
        image = rgb2gray(image1)
        ret, thresh = cv2.threshold(image,0,127,cv2.THRESH_BINARY)
        #th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
         #   cv2.THRESH_BINARY,11,2)
        op = misc.binary_fill_holes(thresh,structure=np.ones((5,5))).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(outDir, files), op)
        cv2.imwrite(os.path.join(outDir1, files), image)