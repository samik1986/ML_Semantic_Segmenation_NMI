import numpy as np
import keras
# import matplotlib.pyplot as plt
import tensorflow as tf
import keras.models
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, merge, UpSampling2D, Reshape, BatchNormalization
from keras.layers import Input, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
from keras.engine.topology import Layer
# from tensorflow.python.ops import array_ops
# from scipy.linalg._expm_frechet import vec
# from tensorflow.python.framework import ops
# from tensorflow.python.framework.op_def_library import _Flatten, _IsListValue
from keras.callbacks import TensorBoard, ModelCheckpoint
# from modelUnet import *
# from data import *
import cv2
import tensorflow as tf
# from skimage import transform
from scipy import ndimage
import numpy as np
from scipy.ndimage import zoom
from keras.callbacks import TensorBoard
from scipy import misc
import os
# import 
# import albu_dingkang
from multiprocessing import Pool
# from matplotlib import image as mpimg
import scipy.misc
# from torch.multiprocessing import Pool, Process, set_start_method, freeze_support

# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass


import new_dm_mba


eps = 0.0001
#fileList1 = os.listdir('m25/jhuangU19/level_1/180605_WG_Tle4lslFlpRPCFA_female/stitchedImage_ch2/')

# filePath = '/nfs/data/main/M25/MorseSkeleton_OSUMITRA/TrainingData/180830/test/data/'
# filePath = '/home/samik/ProcessDet/temp/'
filePath = '/home/samik/ProcessDet/wdata/PS_Process/train/DM/'
# filePath = 'newDM/'
fileList1 =os.listdir(filePath)
#print(fileList1)
for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
   if not(fichier.endswith(".tif")):
       fileList1.remove(fichier)
#print(fileList1).
outDir = '/home/samik/ProcessDet/wdata/PS_Process/train/dmOP/'
# outDir = 'outDM/'
os.system("mkdir " +  outDir)
def dm_fn(tile,id):
        # id = np.random.randint(0,1000)
        # print(id)
        # tile16 = tile * 16.
        # print(tile16.max(), tile16.min())
        dm_op = new_dm_mba.dm_cal(tile, id)
        # print(dm_op.max(), dm_op.min())
        return dm_op

# print fileList1
def testImages(files1, inDir, outDir):
        L = len(files1)
        count = 0
        id = []
        tile = []
        for f1 in sorted(files1):
                # f1 = files1
                print(L, "------------------>", f1)
                img = cv2.imread(inDir + "/" + f1, cv2.IMREAD_UNCHANGED)
                # print(img.max(), img.min())
                # img = mpimg.imread(inDir + "/" + f1)
                # print(img.shape)
                tile.append(img)
                id.append(count)
                count = count+1

        p = Pool(40)
        # print(id)
        dm_opL = p.starmap(dm_fn, zip(tile, id))
        p.close()
        p.join()
        # results = dm_fn(img,count)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # print(img.shape)
                # tile = img[:,:].astype('float32')
                # w, h= tile.shape
                #op = np.zeros((w,h))
                # print(w,h)
        count = 0        
        for f1 in sorted(files1):        
                scipy.misc.toimage(dm_opL[count],cmin=0.0,cmax=1.0).save(outDir + f1)
                count = count + 1
                #cv2.imwrite(outDir + f1, results)


testImages(fileList1, filePath, outDir)


