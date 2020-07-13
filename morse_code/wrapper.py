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
from createNetR import dmnet
import albu_dingkang
from multiprocessing import Pool
# from torch.multiprocessing import Pool, Process, set_start_method, freeze_support
from skimage.io import imread
import h5py
import hdf5storage
import fnmatch
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass


import new_dm_mba
# from albu import test_eval
# from dmp import tst
os.environ['OPENCV_IO_ENABLE_JASPER']= '1'
eps = 0.0001
filePath = '/nfs/data/main/M32/RegistrationData/Data/PMD1605/Transformation_OUTPUT/PMD1605_img/'
fileList1 = os.listdir(filePath)
maskDir = '/nfs/data/main/M32/RegistrationData/Data/PMD1605/Transformation_OUTPUT/reg_high_seg_pad/'
outDir = '/nfs/data/main/M32/ProcessDetPass1/PMD1605/'
outDirG = '/nfs/data/main/M32/ProcessDetPass1/PMD1605/G/'
outDirG = '/nfs/data/main/M32/ProcessDetPass1/PMD1605/R/'
os.system("mkdir " + outDir)
os.system("mkdir " + outDirG)
os.system("mkdir " + outDirR)
fileList2 = os.listdir(outDir)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*.jp2')):
        fileList1.remove(fichier)


for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*F*')):
        fileList1.remove(fichier)

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



def dm_fn(tile,id):
        print(id)
        # tile16 = (tile/(tile.max()+eps)) * 65536.
        dm_op = new_dm_mba.dm_cal(tile, id)
        return dm_op

# print fileList1
def testImages(files1, files2, inDir, outDir):
    L = len(files1)
    for f1 in sorted(files1):

        image = imread_fast(os.path.join(filePath, f1))

        maskB = hdf5storage.loadmat(os.path.join(maskDir,f1.replace('jp2', 'mat')))['seg']
        maskB = maskB / maskB.max()
        _,maskB = cv2.threshold(maskB,0,1,cv2.THRESH_BINARY)
        maskB = np.uint8(maskB) * 255
        print(L, "------------------>", f1)
        print(image.shape)
        img = image
        w, h, c= img.shape
        op = np.zeros((w,h))
        opR = np.zeros((w,h))
        opG = np.zeros((w,h))
        # print(w,h)
        ################## Tiling #####################
        # this is to identify the tile
        id = []
        tile = []
        count = 0
        for row in range(0, w-511, 512):
                for col in range(0,  h-511, 512):
                        tileM = maskB[row:row + 512, col:col + 512]
                        if np.sum(tileM):
                                tile.append(img[row:row+512, col:col+512,0])
                                tile.append(img[row:row+512, col:col+512,1])
                                id.append(count)
                                count += 1
                                id.append(count)
                                count += 1
        p = Pool(40)
        dm_opL = p.starmap(dm_fn, zip(tile, id))
        p.close()
        p.join()
        count = 0
        for row in range(0, w-511, 512):
            for col in range(0,  h-511, 512):
                tileM = maskB[row:row + 512, col:col + 512]
                if np.sum(tileM):
############################# Channel Red ##############################################################
                ###################### ALBU ###################
                        tileN = np.zeros((512,512,3)).astype(np.uint8)
                        tileN[:,:,0] = tile[count]//16
                        # tileN = tileN.astype('float32')
                        albu_op = albu_dingkang.predict(models_albu, tileN, image_type='8_bit_RGB')
                        albu_op_norm = albu_op / 255.
                        albu_op_norm = albu_op_norm.astype('float32')
                ##################### DM++ ###################
                        dm_op = dm_opL[count]
                        dm_op = dm_op[np.newaxis, ..., np.newaxis]
                        # dm_op = dm_op / 255.
                        dm_op = dm_op.astype('float32')
                        albu_op = albu_op_norm[np.newaxis, ..., np.newaxis]
                        out_img = model_dmp.predict([dm_op, albu_op])
                        dmp_op = np.squeeze(out_img[0]) * 255.
                        dmp_op1 = np.uint8(dmp_op)
                        #  print(op.max(), op.min())
                        _, thresh1 = cv2.threshold(dmp_op1,0.3*255,255,cv2.THRESH_BINARY)
                        opR[row:row+512, col:col+512] = thresh2
                        count = count + 1
################################# Channel Green ##########################################################
                        tileN = np.zeros((512,512,3)).astype(np.uint8)
                        tileN[:,:,1] = tile[count]//16
                        # tileN = tileN.astype('float32')
                        albu_op = albu_dingkang.predict(models_albu, tileN, image_type='8_bit_RGB')
                        albu_op_norm = albu_op / 255.
                ##################### DM++ ###################
                        dm_op = dm_opL[count]
                        dm_op = dm_op[np.newaxis, ..., np.newaxis]
                        albu_op = albu_op_norm[np.newaxis, ..., np.newaxis]
                        out_img = model_dmp.predict([dm_op, albu_op])
                        dmp_op = np.squeeze(out_img[0]) * 255.
                        dmp_op2 = np.uint8(dmp_op)
                        #  print(op.max(), op.min())
                        _, thresh2 = cv2.threshold(dmp_op2,0.3*255,255,cv2.THRESH_BINARY)
                        opg[row:row+512, col:col+512] = thresh2
                #################### Stitching ###############
                        op[row:row+512, col:col+512] = np.clip(thresh1 + thresh2, 0, 255)
                        #print(count, row,col)
                        count += 1

#################### write back Mask ###############
        # print(op.shape)
        op = np.uint8(op)
        cv2.imwrite(outDir + "/" + f1[:-4] + ".jp2", op)
#



## Dingkang: Here I load all four models of albu. Need to put foldi_best.pth files under folder albu_weights.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
models_albu = albu_dingkang.read_model([os.path.join('wts_MBA/', 'fold{}_best.pth'.format(i)) for i in range(4)])


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
model_dmp = dmnet('dmnet_membrane_4Dec.hdf5')

testImages(fileList1, fileList2, filePath, outDir)


