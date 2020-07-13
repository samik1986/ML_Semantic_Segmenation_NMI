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
import albu_dingkang
from multiprocessing import Pool
# from torch.multiprocessing import Pool, Process, set_start_method, freeze_support

# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass


#import dm
# from albu import test_eval
# from dmp import tst

eps = 0.0001
filePath = '/home/samik/ProcessDet/wdata/PS_Process/train/images/'

fileList1 = os.listdir(filePath)
outDir = '/home/samik/ProcessDet/wdata/PS_Process/train/predicted/'
os.system("mkdir " + outDir)
#fileList1 ='StitchedImage_Z052_L001.tif'
#print(fileList1)
for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fichier.endswith(".tif")):
        fileList1.remove(fichier)
#print(fileList1)


# print fileList1
def testImages(files1, inDir, outDir):
    L = len(files1)
    for f1 in sorted(files1):
        # print(L, "------------------>", f1)
        img = cv2.imread(inDir + "/" + f1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        # img = img[:,:]
        # w, h= img.shape
        # print(img.max(), img.min())
        # print(w,h)
        tile = img
        op = albu_dingkang.predict(models_albu, tile, image_type='8_bit_RGB')

#################### write back Mask ###############
        # print(op.shape)
        op = np.uint8(op)
        if(op.max() > 30):
                print(op.max(), op.min())
#        ret, thresh = cv2.threshold(op,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        M = cv2.getRotationMatrix2D((w/2,h/2),-90,1)
#        rotated = cv2.warpAffine(thresh,M,(w,h))
#        red_op = np.uint8(np.zeros((h,w,3)))
#        red_op[:,:,2] = rotated
        cv2.imwrite(outDir + "/" + f1, op)
#


# model_albu = load_model('albu_model.hdf5')
# model_dmp = load_model('dmnet_membraneFT.hdf5')

## Dingkang: Here I load all four models of albu. Need to put foldi_best.pth files under folder albu_weights.
models_albu = albu_dingkang.read_model([os.path.join('/home/samik/ProcessDet/wts_PS/', 'fold{}_best.pth'.format(i)) for i in range(4)])

# def run():
#     freeze_support()
#     print('loop')

# if __name__ == '__main__':
#     run()
testImages(fileList1, filePath, outDir)


