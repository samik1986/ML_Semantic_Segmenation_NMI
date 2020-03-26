import os
from PIL import Image
import sys
import numpy as np
import cv2
import fnmatch
from skimage.io import imread
import numpy.matlib as matlib

mainDir = '/nfs/data/main/M32/Process_Detection/ProcessDetPass1/'

outDir = '/nfs/data/main/M32/Process_Detection/ProcessDetPass1_web/'

def imread_fast(img_path):
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    err_code = os.system("kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 16")
    img = imread('temp/'+base+'.tif')
    os.system("rm temp/"+base_C+'.tif')
    return img

mainFolders = os.listdir(mainDir)
outFolders = os.listdir(outDir)

for folders in mainFolders:
    if folders not in outFolders:
        files = os.listdir(os.path.join(mainDir,folders))
        for fichier in files[:]:
            if not(fnmatch.fnmatch(fichier, '*.jp2')):
                files.remove(fichier)
        os.system("mkdir " + (os.path.join(outDir, folders)))
        for file in files:
            print(file)
            
            img = imread_fast(os.path.join(mainDir + folders, file))
            imgR = np.zeros((24000,24000,3), dtype = np.uint8)
            print(imgR.shape)
            # imgR = matlib.repmat(imgR, 1,3)
            # print(imgR.shape)
            imgR[:,:,2] = img
            cv2.imwrite(os.path.join(outDir + folders, file), imgR)








# input_folder = '/home/samik/ProcessDet/wdata/train/masks2m/'
# # input_folder = '/home/samik/ProcessDet/wdata/Process/annotV/'
# input_folder2 = '/home/samik/ProcessDet/wdata/test/images/'
# files = os.listdir(input_folder)
# files2 = os.listdir(input_folder2)

# for f1 in files:
#     # if f1.replace('png', 'tif') not in files:
#     os.system("cp " + os.path.join('/home/samik/ProcessDet/wdata/TrainingData2/rawD/', f1) + " /home/samik/ProcessDet/wdata/train/images/")
#     # os.system("cp " + os.path.join('/home/samik/ProcessDet/wdata/TrainingData/dmV/', f1) + " /home/samik/ProcessDet/wdata/test/dmIP/")
#     # img = np.zeros((512,512,3), dtype = np.uint8)
#     # imgRaw =  cv2.imread(os.path.join('/nfs/data/main/M32/PMD1605_Annotations/dataMBA/cropped_annotation/Process/rawD', f1), cv2.IMREAD_UNCHANGED)
#     # # print(f1)
#     # if fnmatch.fnmatch(f1, '*R.tif'):
#     #     img[:,:,0] = imgRaw
#     #     print(f1)
#     # if fnmatch.fnmatch(f1, '*G.tif'):
#     #     img[:,:,1] = imgRaw
#     #     print(f1)
#     # rgbimg = Image.fromarray(img)
#     # # rgbimg.paste(img)
#     # rgbimg.save("/home/samik/ProcessDet/wdata/train/images/" + f1)
    