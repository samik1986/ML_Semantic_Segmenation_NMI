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
from scipy.ndimage.morphology import binary_fill_holes
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
from scipy.ndimage.morphology import binary_fill_holes
import geojson
from geojson import Feature, FeatureCollection, MultiPoint
from numpy import nonzero


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
    POST_NMS_ROIS_INFERENCE = 4000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.8

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
    RPN_NMS_THRESHOLD = 0.8

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)



def image2mask(image, maskB, maskBR, maskBG, op,  w, h, flagR=False, flagG=False):
    # val1 = 511
    val2 = 256
    for row in range(0, w - val2 - 1, val2):
        # print(row)
        for col in range(0, h - val2 - 1, val2):
            mask = np.zeros((val2, val2), dtype='uint8')
            #tileC = np.zeros((512, 512))
            tile = image[row:row + val2, col:col + val2, :]
            tileM = maskB[row:row + val2, col:col + val2] 
            tileMR = maskBR[row:row + val2, col:col + val2] 
            tileMG = maskBG[row:row + val2, col:col + val2] 
            if np.sum(tileM): 
                tile = cv2.resize(tile, (512,512))
                outK = np.zeros((512, 512), dtype= np.uint8)
                tileR = np.zeros((512, 512, 3), dtype= np.uint8)
                arr = tile[:, :, 0]
                if np.sum(tileMR):
                    arr = arr.astype(np.float)
                    arr = np.power(((arr - arr.min())/(arr.max()-arr.min())),0.4)*255.
                tileR[:, :, 0] = np.uint8(arr)
                out = nucleus1.detect(model, tileR)
                # print(np.sum(out,axis=2), np.sum(out,axis=2).nonzero())
                # R = np.sum(out,axis=2)
                # mask[np.sum(out,axis=2).nonzero()] = 1
                outK[np.sum(out,axis=2).nonzero()] = 1
                outK = cv2.resize(np.uint8(outK), (val2,val2))
                outK = cv2.threshold(outK, 0, 1, cv2.THRESH_BINARY)
                outK = np.asarray(outK[1])
                # print(outK.max())
                outK = binary_fill_holes(outK)
                if np.sum(tileMR):
                    mask[outK.nonzero()] = 101
                else:
                    mask[outK.nonzero()] = 1
                # r, c, num = out.shape
                # for k in range(0, num):
                #     mask = mask + out[:, :, k]*1
                
                tileG = np.zeros((512, 512, 3), dtype=np.uint8)
                outK = np.zeros((512, 512), dtype= np.uint8)
                arr = tile[:, :, 1]
                if np.sum(tileMG):
                    arr = arr.astype(np.float)
                    arr = np.power(((arr - arr.min())/(arr.max()-arr.min())),0.4)*255.
                tileG[:, :, 1] = np.uint8(arr)
                out = nucleus1.detect(model, tileG)
                # r, c, num = out.shape
                # for k in range(0, num):
                outK[np.sum(out,axis=2).nonzero()] = 1
                outK = cv2.resize(np.uint8(outK), (val2,val2))
                outK = cv2.threshold(outK, 0, 1, cv2.THRESH_BINARY)
                outK = np.asarray(outK[1])
                outK = binary_fill_holes(outK)
                if np.sum(tileMG):
                    mask[outK.nonzero()] = 102
                else:
                    mask[outK.nonzero()] = 2
                # out = cv2.resize(np.uint8(out), (val2,val2))
                # if np.sum(tileMG):
                #     mask[np.sum(out,axis=2).nonzero()] = 102
                # else:
                #     mask[np.sum(out,axis=2).nonzero()] = 2
               
                
            mask = np.uint8(mask)
            mask[0,0] = 0
            mask[255,255] = 0
            mask[0,255] = 0
            mask[255,0] = 0
            # mask = cv2.resize(mask,(val2, val2))
            op[row:row + val2, col:col + val2] = mask
    return op

weights_path = '/home/samik/Mask_RCNN/logs/nucleus20191030T1510/mask_rcnn_nucleus_0199.h5'

os.environ['OPENCV_IO_ENABLE_JASPER']= '1'
config = NucleusInferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir='/home/samik/Mask_RCNN/logs/')

model.load_weights(weights_path, by_name=True)

brainNo = 'PMD1228'

filePath = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/' + brainNo + '_img/'
# filePath = 'temp/'
maskDir = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/reg_high_seg_pad/'

outDir = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_soma/' + brainNo + '/'
outDirN = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_reg/' + brainNo + '/'
jsonOutG = os.path.join(outDir, 'jsonG/')
jsonOutR = os.path.join(outDir, 'jsonR/')
jsonOutGN = os.path.join(outDirN, 'jsonG/')
jsonOutRN = os.path.join(outDirN, 'jsonR/')
maskOut = os.path.join(outDir, 'mask/')
maskOutN = os.path.join(outDirN, 'mask/')
# maskOutN = 'temp_out/'


os.system("mkdir " + outDir)
os.system("mkdir " + outDirN)
os.system("mkdir " + maskOut)
os.system("mkdir " + jsonOutR)
os.system("mkdir " + jsonOutG)
os.system("mkdir " + maskOutN)
os.system("mkdir " + jsonOutRN)
os.system("mkdir " + jsonOutGN)

fileList1 = os.listdir(filePath)
fileList2 = [] #os.listdir(maskOutN)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*.jp2')):
        fileList1.remove(fichier)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*F*')):
        fileList1.remove(fichier)


#print(fileList1)

for files in fileList1:
    if files not in fileList2:
        print(files)
        image = imread_fast(os.path.join(filePath, files))

        image1 = image[:,:,0] //16

        maskBR = image1 > np.uint16(10.) 
        remove_small_objects(maskBR, min_size=1024, connectivity=2, in_place=True)
        # print(np.count_nonzero(maskBR))
        maskBR = binary_fill_holes(maskBR)
        # print(sum(maskBR[maskBR>0]))

        image1 = image[:,:,1] // 16
        # image1[image1>255] = 255
        maskBG = image1 > np.uint16(10.)
        # _,maskBG = cv2.threshold(image1,10,255,cv2.THRESH_BINARY)
        remove_small_objects(maskBG, min_size=1024, connectivity=2, in_place=True)
        maskBG = binary_fill_holes(maskBG)
        print(np.count_nonzero(maskBR),np.count_nonzero(maskBG))
        flagR = False
        flagG = False
        if np.count_nonzero(maskBR):
            flagR = True

        if np.count_nonzero(maskBG):
            flagG = True
   
        maskB = hdf5storage.loadmat(os.path.join(maskDir,files.replace('jp2', 'mat')))['seg']
        maskB = maskB / maskB.max()
        ret,maskB = cv2.threshold(maskB,0,1,cv2.THRESH_BINARY)
        maskB = np.uint8(maskB) * 255        
        w, h, c = image.shape
        image = image // 16
        image = image.astype(np.uint8)
        
        op = np.zeros((w,h),dtype='uint8')
        opRN = np.zeros((w,h),dtype='bool')
        opGN = np.zeros((w,h),dtype='bool')
        opR = np.zeros((w,h),dtype='bool')
        opG = np.zeros((w,h),dtype='bool')
        opM = np.zeros((w,h),dtype='uint8')
        # opOut = np.zeros((w,h),dtype='uint8')

        op = image2mask(image, maskB, maskBR, maskBG, op, w, h, flagR=flagR, flagG=flagG)


        opR[op==101] = True
        opG[op==102] = True
        opRN[op==1] = True
        # opRN[op==101] = True
        opGN[op==2] = True 
        # opGN[op==102] = True

        print(maskB.max())
        ### Injection Soma Detects ###

        if flagR:
            opR8 = np.uint8(opR)
            _, thresh = cv2.threshold(opR8,0,1,cv2.THRESH_BINARY)
            _, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
            if len(centroids)>5:
                f = open(os.path.join(jsonOutR, files.replace('jp2', 'json')), "w")
                f.write("{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"id\":\"101\",\"properties\":{\"name\":\"Red Cell\"},\"geometry\":{\"type\":\"MultiPoint\",\"coordinates\":")
                f.write(str(np.int64(centroids).tolist()).replace(', ', ', -').replace('], -', '], '))
                f.write("}}]}")
                f.close()

        if flagG:
            opG8 = np.uint8(opG)
            _, thresh = cv2.threshold(opG8,0,1,cv2.THRESH_BINARY)
            _, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
            if len(centroids)>5:
                f = open(os.path.join(jsonOutG, files.replace('jp2', 'json')), "w")
                f.write("{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"id\":\"102\",\"properties\":{\"name\":\"Green Cell\"},\"geometry\":{\"type\":\"MultiPoint\",\"coordinates\":")
                f.write(str(np.int64(centroids).tolist()).replace(', ', ', -').replace('], -', '], '))
                f.write("}}]}")
                f.close()
            

        # if flagR or flagG:
        #     opOut[np.bitwise_or(opR,opG)] = op[np.bitwise_or(opR,opG)]
        #     cv2.imwrite(os.path.join(maskOut, files[:-4]+'.tif'), np.uint8(opOut))
        #     img = imread(os.path.join(maskOut, files[:-4]+'.tif'))
        #     err_code = os.system("kdu_compress -i " + os.path.join(maskOut, files[:-4]+'.tif').replace("&", "\&") + " -o " + os.path.join(maskOut, files.replace("&", "\&")) + " -num_threads 4")
        #     os.system("rm "+ os.path.join(maskOut, files[:-4]+'.tif').replace("&", "\&"))
         

        #### Cell Detection Full ###

        opR8 = np.uint8(opRN)
        _, thresh = cv2.threshold(opR8,0,1,cv2.THRESH_BINARY)
        _, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
        if len(centroids)>5:
            f = open(os.path.join(jsonOutRN, files.replace('jp2', 'json')), "w")
            f.write("{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"id\":\"1\",\"properties\":{\"name\":\"Red Cell\"},\"geometry\":{\"type\":\"MultiPoint\",\"coordinates\":")
            f.write(str(np.int64(centroids).tolist()).replace(', ', ', -').replace('], -', '], '))
            f.write("}}]}")
            f.close() 

        opG8 = np.uint8(opGN)
        _, thresh = cv2.threshold(opG8,0,1,cv2.THRESH_BINARY)
        _, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
        if len(centroids)>5:
            f = open(os.path.join(jsonOutGN, files.replace('jp2', 'json')), "w")
            f.write("{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"id\":\"2\",\"properties\":{\"name\":\"Green Cell\"},\"geometry\":{\"type\":\"MultiPoint\",\"coordinates\":")
            f.write(str(np.int64(centroids).tolist()).replace(', ', ', -').replace('], -', '], '))
            f.write("}}]}")
            f.close()

        opM[maskB.nonzero()] = op[maskB.nonzero()]
        cv2.imwrite(os.path.join(maskOutN, files), np.uint8(opM))
        # img = imread(os.path.join(maskOutN, files[:-4]+'.tif'))
        # err_code = os.system("kdu_compress -i " + os.path.join(maskOutN, files[:-4]+'.tif').replace("&", "\&") + " -o " + os.path.join(maskOutN, files.replace("&", "\&")) + " -num_threads 4")
        # os.system("rm "+ os.path.join(maskOutN, files[:-4]+'.tif').replace("&", "\&"))
    






