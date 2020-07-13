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
from modelUnet import *
from data import *
import cv2
import tensorflow as tf
from skimage import transform
from scipy import ndimage
import numpy as np
from scipy.ndimage import zoom
from keras.callbacks import TensorBoard
from scipy import misc
from createNetR import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)



X = np.load("albuTrnMB.npy")
Y = np.load("dmMB.npy")
Z = np.load("segTrnMB.npy")
# P = np.load("imgTrnR.npy")
# print(X.max(), Y.max(), Z.max())
# print(X.min(), Y.min(), Z.min())
model = dmnet()
print(model.summary())

print(X.max(), Y.max(), Z.max())
print(X.min(), Y.min(), Z.min())
model.load_weights('dmnet_membrane.hdf5')
model_checkpoint = ModelCheckpoint('dmnet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit([X, Y], Z, epochs=100, batch_size= 3, callbacks=[model_checkpoint], validation_split = 0.3)#, initial_epoch = 30)# ,validation_data=imageLoaderV(fileList1, 2),validation_steps=578)

