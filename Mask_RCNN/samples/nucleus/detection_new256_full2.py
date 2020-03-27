import nucleus1
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
    DETECTION_MIN_CONFIDENCE = 0.7

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
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.8

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

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


def image2mask(image, maskB, op, w, h):
    for row in range(0, w - 255, 256):
        # print(row)
        #tileS = []
        for col in range(0, h - 255, 256):
            mask = np.zeros((512, 512), dtype='bool')
            #tileC = np.zeros((512, 512))
            tile = image[row:row + 256, col:col + 256, :]
            tileM = maskB[row:row + 256, col:col + 256]
            if np.sum(tileM): 
                tile = cv2.resize(tile, (512,512))
                #tileC = tile[:, :, 0]
                tileR = np.zeros((512, 512, 3))
                tileR[:, :, 0] = tile[:, :, 0]
                #tileS.append(tileR)
                #tileR = cv2.resize(tileR, (512,512))
                tileG = np.zeros((512, 512, 3))
                tileG[:, :, 1] = tile[:, :, 1]
                #tileB = np.zeros((512, 512, 3))
                #tileB[:, :, 0] = tile[:, :, 2]
                
                out = nucleus1.detect(model, tileR)
                #print(out.shape)
                r, c, num = out.shape
                #print(num)
                for k in range(0, num):
                    mask = mask + out[:, :, k]

                out = nucleus1.detect(model, tileG)
                r, c, num = out.shape
                # print(num)
                for k in range(0, num):
                    mask = mask + out[:, :, k]

            #out = nucleus1.detect(model, tileB)
            #r, c, num = out.shape
            #print(num)
            #for k in range(0, num):
            #mask = mask + out[:, :, k]
            mask = np.uint8(mask)
            mask = cv2.resize(mask,(256, 256))
            op[row:row + 256, col:col + 256] = mask
            #opT[row:row + 512, col:col + 512] = mask
    return op #, opT




weights_path = '/home/samik/Mask_RCNN/logs/nucleus20191030T1510/mask_rcnn_nucleus_0199.h5'

os.environ['OPENCV_IO_ENABLE_JASPER']= '1'
config = NucleusInferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir='/home/samik/Mask_RCNN/logs/')

model.load_weights(weights_path, by_name=True)

filePath = '/nfs/data/main/M32/RegistrationData/Data/PMD1605/Transformation_OUTPUT/PMD1605_img/'
#filePath = 'input/'
maskDir = '/nfs/data/main/M32/RegistrationData/Data/PMD1605/Transformation_OUTPUT/reg_high_seg_pad/'
outDir = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_reg/PMD1605/'
maskOut = os.path.join(outDir, 'mask/')
jsonOut = os.path.join(outDir, 'json/')
os.system("mkdir " + outDir)
os.system("mkdir " + maskOut)
os.system("mkdir " + jsonOut)
#outDir = 'output/'
fileList1 = os.listdir(filePath)
fileList2 = os.listdir(maskOut)
for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*.jp2')):
        fileList1.remove(fichier)


# for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
#     if fichier.endswith("_lossless.jp2"):
#         fileList1.remove(fichier)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*F*')):
        fileList1.remove(fichier)


#print(fileList1)

for files in fileList1:
    if files not in fileList2:
        print(files)
        image = imread_fast(os.path.join(filePath, files))

        maskB = hdf5storage.loadmat(os.path.join(maskDir,files.replace('jp2', 'mat')))['seg']
        maskB = maskB / maskB.max()

        imageR = image[:,:,0]
        r, c = imageR.shape
        # maskB = cv2.resize(maskB, (c,r))
        ret,maskB = cv2.threshold(maskB,0,1,cv2.THRESH_BINARY)
        maskB = np.uint8(maskB) * 255
        #image = np.multiply(maskB, imageR)
        w, h, c = image.shape
        #print(image.max())
        # idx = image > 255.
        image = image // 16
        image = image.astype(np.uint8)
        #opT = np.zeros((w,h))
        op = np.zeros((w,h),dtype='bool')
        #print(image.max())
        op = image2mask(image, maskB, op, w, h)
        op = np.uint8(op) * 255
        # se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(101,101))
        # erd = cv2.erode(maskB, se, iterations = 1) / 255
        # op = np.multiply(erd, op)
        # _, thresh = cv2.threshold(op,127,255,cv2.THRESH_BINARY)
        # # print(thresh.shape)
        # # connectivity = 8
        # _, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
        # # print(centroids)
        # f = open(os.path.join(jsonOut, files.replace('jp2', 'json')), "w")
        # f.write("{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"id\":\"0\",\"properties\":{\"name\":\"Cell\"},\"geometry\":{\"type\":\"MultiPoint\",\"coordinates\":[")
        # for pts in centroids:
        #     # print(pts[0])
        #     pts64 = pts.astype(np.int64)
        #     f.write("[" + str(pts64[0]) + "," + str(pts64[1]*-1) + "],")
        # f.write("[]]}}]}")
        # f.close()
        # #op = np.multiply(maskB, op)
        # print(op.max())
        # #ret, thresh = cv2.threshold(op,0,0.7*255,cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(maskOut, files[-4]+'.tif'), op)
        img = imread(os.path.join(maskOut, files[-4]+'.tif'))
        err_code = os.system("kdu_compress -i " + os.path.join(maskOut, files[-4]+'.tif') + " -o " + os.path.join(maskOut, files) + " -num_threads 16")
        os.system("rm "+ os.path.join(maskOut, files[-4]+'.tif'))
        # cv2.imwrite(os.path.join(outDir, files), image[:,:,0])







