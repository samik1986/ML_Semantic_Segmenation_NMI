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
# from torch.multiprocessing import Pool, Process, set_start_method, freeze_support

# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass


import new_dm


eps = 0.0001
#fileList1 = os.listdir('m25/jhuangU19/level_1/180605_WG_Tle4lslFlpRPCFA_female/stitchedImage_ch2/')

filePath = '/nfs/data/main/M25/MorseSkeleton_OSUMITRA/TrainingData/180405/train/img/'
# filePath = '/home/samik/ProcessDet/temp/'
fileList1 =os.listdir(filePath)
#print(fileList1)
for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
   if not(fichier.endswith(".tif")):
       fileList1.remove(fichier)
#print(fileList1).
outDir = '/home/samik/ProcessDet/DM/180405/train/'
os.system("mkdir " +  outDir)
def dm_fn(tile,id):
        # id = np.random.randint(0,1000)
        print(id)
        tile16 = (tile/(tile.max()+eps)) * 65536.
        dm_op = new_dm.dm_cal(tile16, id)
        return dm_op

# print fileList1
def testImages(files1, inDir, outDir):
    L = len(files1)
    count = 0
    for f1 in sorted(files1):
        # f1 = files1
        print(L, "------------------>", f1)
        img = cv2.imread(inDir + "/" + f1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img.shape)
        tile = img[:,:].astype('float32')
        w, h= tile.shape
        #op = np.zeros((w,h))
        print(w,h)
        count = count + 1
        results = dm_fn(tile,count) * 255.
        cv2.imwrite(outDir + f1, results.astype(np.uint8))


testImages(fileList1, filePath, outDir)


