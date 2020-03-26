import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model, Model, save_model
# import Image
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.models
import os
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
from tensorflow.python.ops import array_ops
from scipy.linalg._expm_frechet import vec
from tensorflow.python.framework import ops
from tensorflow.python.framework.op_def_library import _Flatten, _IsListValue
from keras.callbacks import TensorBoard, ModelCheckpoint
from modelUnet import *
from data import *
import cv2
import tensorflow as tf
from skimage import transform
from scipy import ndimage
import numpy as np
from scipy.ndimage import zoom
# import torch.nn.functional as F
from keras.callbacks import TensorBoard
from scipy import misc
from scipy.misc.pilutil import imread,imsave
from createNetR import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

# model = dmnet()

model = dmnet('dmnet_membrane_6Dec.hdf5')

# X = np.load("imgTst.npy")
# Y = np.load("dmTST.npy")



fileList1 = os.listdir('tosamik/red_count/albu/')
fileList2 = os.listdir('tosamik/red_count/dm/')
# fileList3 = os.listdir('/home/samik/mnt/m25/MorseSkeleton_OSUMITRA/TrainingData/180830/')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)

def sigmoid(x):
    return 1 / 1 + np.exp(-x)

def testImages(files1, name1, name2):
    L = len(files1)
    X = []
    Y = []
    # bndry = misc.imread('Offsetmask.png').astype('float32')
    #filePath = 'membrane/morseUpdate/'
    for f1 in files1:
        print(f1)
        img = imread(name1 + "/" + f1)
        print (img.max())
        # print max(max(row) for row in img)
        img = img.astype('float32')
        dm = imread(name2 + "/" + f1).astype('float32')
        img = img / 255.
        dm = dm / 255.
        print(img.max(), dm.max())       # if img.max():
        #     img = img / img.max()
        # dm = dm / 255.
        # print dm.max(), img.max()
        # org = misc.imread(name3 + "/" + f1[:-8] + '.tif').astype('float32')
        # if org.max():
        #     org = org / org.max()
        X_arr = np.asarray(img)
        X_arr = X_arr[..., np.newaxis]
        X_arr = X_arr[np.newaxis, ...]
        
        Y_arr = np.asarray(dm)
        Y_arr = Y_arr[..., np.newaxis]
        Y_arr = Y_arr[np.newaxis, ...]
        # P_arr = np.asarray(org)
        # P_arr = P_arr[..., np.newaxis]
        # P_arr = P_arr[np.newaxis, ...]
#        print model.summary()
        out_img = model.predict([X_arr, Y_arr])
        # out_img = sigmoid(out_img)

        # print(out_img.min(), out_img.max())
        img = np.squeeze(out_img[0]) * 255. #* 100000.
        print(img.min(), img.max())

        imsave("tosamik/red_count/dmpp/" + f1, img.astype('uint8'))
        

testImages(fileList1,'tosamik/red_count/albu/', 'tosamik/red_count/dm/')
