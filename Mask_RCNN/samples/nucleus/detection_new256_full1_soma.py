import nucleus0 as nucleus1
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import fnmatch
import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import tensorflow as tf
import keras.backend as K
from imgaug import augmenters as iaa
from PIL import Image
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
import os
from skimage.io import imread
from multiprocessing import Pool
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import active_contour
from skimage.morphology import skeletonize
from skimage.morphology import convex_hull_image
from functools import partial
from time import gmtime, strftime
import matplotlib.image as mpimg
import pickle
import matplotlib.path as mplPath
import sys
from skimage.io import imread
from scipy import misc 
from scipy.io import loadmat
import h5py
import hdf5storage
import bisect
import statistics


def imread_fast(img_path):
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    #print("./kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 1")
    # os.system("cd /home/samik/v7_A_6-01832N/bin/Linux-x86-64-gcc/")
    err_code = os.system("kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 16")
    #print(err_code)
    # os.system("cd /home/samik/Mask_RCNN/samples/nucleus")
    img = imread('temp/'+base+'.tif')
    os.system("rm temp/"+base_C+'.tif')
    return img

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    #STEPS_PER_EPOCH = (35 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    #VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.6

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 4000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([0, 0, 0])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def image2mask(image, maskBR, maskBG, opR, opG, w, h, flagR=False, flagG=False):
    for row in range(0, w - 255, 256):
        # print(row)
        #tileS = []
        for col in range(0, h - 255, 256):
            maskR = np.zeros((512, 512), dtype='bool')
            maskG = np.zeros((512, 512), dtype='bool')
            #tileC = np.zeros((512, 512))
            tile = image[row:row + 256, col:col + 256, :]
            tileM = maskBR[row:row + 256, col:col + 256] + maskBG[row:row + 256, col:col + 256]
            if np.sum(tileM): 
                tile = cv2.resize(tile, (512,512))
                #tileC = tile[:, :, 0]
                if flagR:
                    tileR = np.zeros((512, 512, 3), dtype= np.uint8)
                    arr = tile[:, :, 0]
                    arr = arr.astype(np.float)
                    arr = np.power(((arr - arr.min())/(arr.max()-arr.min())),0.4)*255.
                    tileR[:, :, 0] = np.uint8(arr)
                    out = nucleus1.detect(model, tileR)
                    r, c, num = out.shape
                    #print(num)
                    for k in range(0, num):
                        maskR = maskR + out[:, :, k]
                
                if flagG:
                    tileG = np.zeros((512, 512, 3), dtype=np.uint8)
                    arr = tile[:, :, 1]
                    arr = arr.astype(np.float)
                    arr = np.power(((arr - arr.min())/(arr.max()-arr.min())),0.4)*255.
                    tileG[:, :, 1] = np.uint8(arr)
                    out = nucleus1.detect(model, tileG)
                    r, c, num = out.shape
                    # print(num)
                    for k in range(0, num):
                        maskG = maskG + out[:, :, k]
               
                
            maskR = np.uint8(maskR)
            maskR = cv2.resize(maskR,(256, 256))
            maskG = np.uint8(maskG)
            maskG = cv2.resize(maskG,(256, 256))
            opR[row:row + 256, col:col + 256] = maskR
            opG[row:row + 256, col:col + 256] = maskG
            #opT[row:row + 512, col:col + 512] = mask
    return opR, opG #, opT

weights_path = '/home/samik/Mask_RCNN/logs/nucleus20191030T1510/mask_rcnn_nucleus_0199.h5'

os.environ['OPENCV_IO_ENABLE_JASPER']= '1'
config = NucleusInferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir='/home/samik/Mask_RCNN/logs/')

model.load_weights(weights_path, by_name=True)

brainNo = 'PMD1476'

filePath = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/' + brainNo + '_img/reg_high_tif_pad_jp2/'
# filePath = '/nfs/data/main/M27/mba_converted_imaging_data/PMD1028&1027/PMD1027/reg_high_tif_pad_jp2'
#filePath = 'input/'
maskDirG = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/InjG_OUTPUT/reg_high_tif_pad_jp2/'
maskDirR = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/InjR_OUTPUT/reg_high_tif_pad_jp2/'
outDir = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_soma/' + brainNo + '/'
# maskOutG = os.path.join(outDir, 'maskG/')
# maskOutR = os.path.join(outDir, 'maskR/')
jsonOutG = os.path.join(outDir, 'jsonG/')
jsonOutR = os.path.join(outDir, 'jsonR/')

os.system("mkdir " + outDir)
# os.system("mkdir " + maskOutG)
# os.system("mkdir " + maskOutR)
# os.system("mkdir " + jsonOut)
os.system("mkdir " + jsonOutR)
os.system("mkdir " + jsonOutG)
#outDir = 'output/'
fileList1 = os.listdir(filePath)
fileList2 = [] #os.listdir(maskOut)
fileListR = os.listdir(maskDirR)
fileListG = os.listdir(maskDirG)
for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*.jp2')):
        fileList1.remove(fichier)



for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*F*')):
        fileList1.remove(fichier)


print(fileList1)

for files in fileList1:
    if files not in fileList2:
        print(files)
        maskBG = np.zeros((24000,24000), dtype = np.uint8)
        maskBR = np.zeros((24000,24000), dtype = np.uint8)
        maskFlag = False
        flagR = False
        flagG = False
        if files in fileListR:
            maskFlag = True
            flagR = True
            maskBR = imread_fast(os.path.join(maskDirR, files))
            maskBR = maskBR / maskBR.max()
            maskBRi = maskBR
            ret,maskBR = cv2.threshold(maskBR,0,1,cv2.THRESH_BINARY)
            maskBR = np.uint8(maskBR) * 255
        if files in fileListG:
            maskFlag = True
            flagG = True
            maskBG = imread_fast(os.path.join(maskDirG, files))
            maskBG = maskBG / maskBG.max()
            maskBGi = maskBG
            ret,maskBG = cv2.threshold(maskBG,0,1,cv2.THRESH_BINARY)
            maskBG = np.uint8(maskBG) * 255
        if maskFlag:
            image = imread_fast(os.path.join(filePath, files))
            # maskBR = maskBR[:,:,0]
            # # r, c = maskBR.shape
            # image = cv2.resize(image, (c,r))
            # print(image.shape)
            # print(maskBR.max())
            # # maskBG = maskBG[:,:,0]
            #  # r, c = maskBG.shape
            # image = cv2.resize(image, (c,r))
            
            w, h, c = image.shape
            image = image // 16
            image = image.astype(np.uint8)
            #opT = np.zeros((w,h))
            opR = np.zeros((w,h),dtype='bool')
            opG = np.zeros((w,h),dtype='bool')
            #print(image.max())
            opR, opG = image2mask(image, maskBR, maskBG, opR, opG, w, h, flagR=flagR, flagG=flagG)
            opR = np.uint8(opR) * 255
            if flagR:
                opR = np.uint8(np.multiply(opR, maskBRi))
            opG = np.uint8(opG) * 255
            if flagG:
                opG = np.uint8(np.multiply(opG, maskBGi))

            _, thresh = cv2.threshold(opR,127,255,cv2.THRESH_BINARY)
            _, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
            f = open(os.path.join(jsonOutR, files.replace('jp2', 'json')), "w")
            f.write("{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"id\":\"1\",\"properties\":{\"name\":\"Red Cell\"},\"geometry\":{\"type\":\"MultiPoint\",\"coordinates\":")
            f.write(str(np.int64(centroids).tolist()).replace(', ', ', -').replace('], -', '], '))
            f.write("}}]}")
            # threshC = np.zeros((24000,24000,3), dtype=np.uint8)
            # threshC[:,:,0] = thresh
            # cv2.imwrite(os.path.join(maskOutR, files[:-4]+'.tif'), threshC)
            # img = imread(os.path.join(maskOutR, files[:-4]+'.tif'))
            # err_code = os.system("kdu_compress -i " + os.path.join(maskOutR, files[:-4]+'.tif').replace("&", "\&") + " -o " + os.path.join(maskOutR, files.replace("&", "\&")) + " -num_threads 16")
            # os.system("rm "+ os.path.join(maskOutR, files[:-4]+'.tif').replace("&", "\&"))

            _, thresh = cv2.threshold(opG,127,255,cv2.THRESH_BINARY)
            _, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
            # print(centroids)
            f = open(os.path.join(jsonOutG, files.replace('jp2', 'json')), "w")
            f.write("{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"id\":\"2\",\"properties\":{\"name\":\"Green Cell\"},\"geometry\":{\"type\":\"MultiPoint\",\"coordinates\":")
            f.write(str(np.int64(centroids).tolist()).replace(', ', ', -').replace('], -', '], '))
            f.write("}}]}")
            f.close()
            # threshC = np.zeros((24000,24000,3), dtype=np.uint8)
            # threshC[:,:,2] = thresh
            # cv2.imwrite(os.path.join(maskOutG, files[:-4]+'.tif'), threshC)
            # img = imread(os.path.join(maskOutG, files[:-4]+'.tif'))
            # err_code = os.system("kdu_compress -i " + os.path.join(maskOutG, files[:-4]+'.tif').replace("&", "\&") + " -o " + os.path.join(maskOutG, files.replace("&", "\&")) + " -num_threads 16")
            # os.system("rm "+ os.path.join(maskOutG, files[:-4]+'.tif').replace("&", "\&"))







