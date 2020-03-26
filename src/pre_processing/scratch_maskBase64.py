import os
from PIL import Image
import sys
import numpy as np
import cv2
import fnmatch
from skimage.io import imread
import numpy.matlib as matlib
import json
import zlib
import base64

mainDir = '/home/samik/NISSL/Aarti/ds/'
annDir = mainDir + 'ann/'
imgdir = mainDir + 'img/'
outDir = mainDir + 'maskout/'

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

mainFolders = os.listdir(imgdir)
# outFolders = os.listdir(outDir)

for files in mainFolders:
    img = cv2.imread(imgdir + files)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masknew = np.zeros((img.shape[0], img.shape[1]), dtype= np.bool)
    print(masknew.shape)

    ann = annDir + files + '.json'
    print(ann)
    with open(ann) as json_file:
        data = json.load(json_file)
        count = 0
        os.system("mkdir " + outDir + files[:-4] + '_mask')
        for p in data['objects']:
            z = zlib.decompress(base64.b64decode(p['bitmap']['data']))
            r = p['bitmap']['origin']
            # r[0] = img.shape[1] - r[1]
            # r[1] = img.shape[0] - r[0]
            n = np.fromstring(z,np.uint8)
            mask = cv2.imdecode(n,cv2.IMREAD_UNCHANGED)[:,:,3].astype(np.bool)
            maskOut = np.zeros((img.shape[0], img.shape[1]), dtype= np.bool)
            maskOut[r[1]: r[1] + mask.shape[0], r[0]:r[0] + mask.shape[1]] = mask
            print(outDir + files[:-4] + '_mask/' + str(count) + '.tif')
            cv2.imwrite(outDir + files[:-4] + '_mask/' + str(r[0])+ '_' + str(r[1]) + '.tif', np.uint8(maskOut)*255)
            count = count + 1






    








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
    