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

from PIL import Image
import new_dm_mba

_errstr = "Mode is unknown or incompatible with input array shape."


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image

eps = 0.0001
#fileList1 = os.listdir('m25/jhuangU19/level_1/180605_WG_Tle4lslFlpRPCFA_female/stitchedImage_ch2/')

# filePath = '/nfs/data/main/M25/MorseSkeleton_OSUMITRA/TrainingData/180830/test/data/'
# filePath = '/home/samik/ProcessDet/temp/'
filePath = '/home/samik/ProcessDet/wdata/PS_Process/train/DM/'
filePath = '/nfs/data/main/M32/Samik/180830/180830_JH_WG_Fezf2LSLflp_CFA_female_processed/TrainingDataProofread/DM/'
fileList1 =os.listdir(filePath)
#print(fileList1)
for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
   if not(fichier.endswith(".tif")):
       fileList1.remove(fichier)
#print(fileList1).
outDir = '/home/samik/ProcessDet/wdata/PS_Process/train/dmOP/'
outDir = '/nfs/data/main/M32/Samik/180830/180830_JH_WG_Fezf2LSLflp_CFA_female_processed/TrainingDataProofread/dmOP/'
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
                # toimage(dm_opL[count],cmin=0.0,cmax=1.0).save(outDir + f1)
                print(outDir + f1)
                toimage(dm_opL[count],cmin=0.0,cmax=1.0).save(outDir + f1)
                count = count + 1
                #cv2.imwrite(outDir + f1, results)


testImages(fileList1, filePath, outDir)


