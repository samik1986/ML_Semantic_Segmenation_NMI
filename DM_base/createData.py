import numpy as np
import keras
import matplotlib.pyplot as plt
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
from tensorflow.python.ops import array_ops
from scipy.linalg._expm_frechet import vec
from tensorflow.python.framework import ops
from tensorflow.python.framework.op_def_library import _Flatten, _IsListValue
from keras.callbacks import TensorBoard, ModelCheckpoint
# from modelUnet import *
# from data import *
import cv2
import os
import tensorflow as tf
from skimage import transform
from scipy import ndimage
import numpy as np
from scipy.ndimage import zoom
from keras.callbacks import TensorBoard
from scipy import misc
# from createNEt import *


filePath1 = 'Train/train_PS/pred1'
filePath2 = 'Train/train_PS/masks2m'
filePath3 = 'Train/train_PS/dmOP'

fileList1 = os.listdir(filePath1)
fileList2 = os.listdir(filePath2)
fileList3 = os.listdir(filePath3)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)



def readImagesTwice(files1, name1, name2, name3):
    L = len(files1)
    X = []
    Y = []
    Z = []
    # P = []
    #filePath = 'membrane//morseUpdate/'
    for f1 in files1:
        img = misc.imread(name1 + "/" + f1)
        # print max(max(row) for row in img)
        img = img.astype('float32')
        # print(img.max())
        # print filePath + name2 + "/" + f1[:-8] + '_mask.tif', filePath + name1 + "/" + f1
        mask = misc.imread(name2 + "/" + f1.replace('tif', 'png')).astype('float32')
        dm = misc.imread(name3 + "/" + f1).astype('float32')
        print(img.max(),dm.max(),mask.max())
        img = img / 255.
        dm = dm / 255.
        mask = mask / 255.
        X.append(img)
        Y.append(dm)
        Z.append(mask)
	# P.append(org)
    X_arr = np.asarray(X)
    X_arr = X_arr[..., np.newaxis]
    Y_arr = np.asarray(Y)
    Y_arr = Y_arr[..., np.newaxis]
    Z_arr = np.asarray(Z)
    Z_arr = Z_arr[..., np.newaxis]
    # P_arr = np.asarray(P)
    # P_arr = P_arr[..., np.newaxis]

    return X_arr, Y_arr, Z_arr



[X, Y, Z] = readImagesTwice(fileList1, filePath1, filePath2, filePath3)
np.save('dmPS.npy', Y)
np.save('albuTrnPS.npy', X)
np.save('segTrnPS.npy', Z)
# np.save('imgTrnR.npy', P)

