import numpy as np
import cv2
# from torch.multiprocessing import Pool, Process, set_start_method, freeze_support
from skimage.io import imread
from PIL import Image
import time
import os

def imread_fast(img_path):
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    err_code = os.system("kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 16")
    print(err_code)
    img = imread('temp/'+base+'.tif')
    os.system("rm temp/"+base_C+'.tif')
    return img

file = '/nfs/data/main/M32/PMD1605_Annotations/Orig/PMD1605&1604-F47-2014.05.31-11.00.36_PMD1605_2_0140_lossless.jp2'
start_time = time.time()
image = imread_fast(file)
print(time.time() - start_time)

start_time = time.time()
image = Image.open(file)
print(time.time() - start_time)