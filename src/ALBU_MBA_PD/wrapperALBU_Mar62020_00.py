import numpy as np
import keras
import os
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
from keras.callbacks import TensorBoard, ModelCheckpoint
import cv2
import tensorflow as tf
from scipy import ndimage
import numpy as np
# from createNetR import dmnet
import albu_dingkang
from multiprocessing import Pool
from skimage.io import imread
import h5py
import hdf5storage
import fnmatch
from skimage.color import label2rgb
# from skimage.exposure import match_histograms
from skimage import morphology, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes
# import tsting_single_cal
# import new_dm_mba1 as new_dm_mba
import time


start_time = time.time()
os.environ['OPENCV_IO_ENABLE_JASPER']= '1'
eps = 0.0001
brainNo = 'PMD2055'
filePath = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/' + brainNo + '_img/'
# filePathIR = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_soma/' + brainNo + '/jsonR/'
# filePathIG = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_soma/' + brainNo + '/jsonG/'
# filePath = 'temp1/'
# fileRef = 'temp/PMD1605&1604-F17-2014.05.30-22.04.31_PMD1605_1_0049.jp2'
# fileRef = 'temp/PMD1605&1604-F13-2014.05.30-20.30.05_PMD1605_1_0037.jp2'
# fileRef = 'temp/PMD1605&1604-F37-2014.05.31-06.29.37_PMD1605_1_0109.jp2'
fileList1 = os.listdir(filePath)
# fileListIR = os.listdir(filePathIR)
# fileListIG = os.listdir(filePathIG)
maskDir = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/reg_high_seg_pad/'
outDir = '/nfs/data/main/M32/Process_Detection/ProcessDetPass1/' + brainNo + '/'
# outDir = filePath
os.system("mkdir " + outDir)
fileList2 = os.listdir(outDir)
lThresh = 0.15


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
models_albu = albu_dingkang.read_model([os.path.join('/home/samik/ProcessDet/results/weights/MBA_Mar6/', 'fold{}_best.pth'.format(i)) for i in range(4)])


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)
# model_dmp = dmnet('dmnet_membrane_4Dec.hdf5')

# model_load_time = time.time()

def _match_cumulative_cdf(source, template):
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)

def match_histograms(image, reference, multichannel=False):
    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number of channels.')
    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and reference image must match!')
        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel], reference[..., channel])
            matched[..., channel] = matched_channel
    else:
        matched = _match_cumulative_cdf(image, reference)
    return matched

for fichier in fileList1[:]: 
    if not(fnmatch.fnmatch(fichier, '*.jp2')):
        fileList1.remove(fichier)

for fichier in fileList1[:]: 
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

# imageRef = imread_fast(fileRef)

# def dm_fn(tile,id):
#         print(id)
#         # tile16 = (tile/(tile.max()+eps)) * 16.
#         dm_op = new_dm_mba.dm_cal(tile, id)
#         return dm_op

def testImages(files1, files2, inDir, outDir):
    print(files1)
    for f1 in sorted(files1):
        if f1 not in files2:
                print(f1)
                fact = np.int16(16)
                image = imread_fast(os.path.join(filePath, f1))
                # image_read_time = time.time()
                
                w, h, c= image.shape

                # image[:,:,0] = match_histograms(image[:,:,0], imageRef[:,:,0])
                # image[:,:,1] = match_histograms(image[:,:,1], imageRef[:,:,1])
                maskB = hdf5storage.loadmat(os.path.join(maskDir,f1.replace('jp2', 'mat')))['seg']
                maskB = maskB / maskB.max()
                _,maskB = cv2.threshold(maskB,0,1,cv2.THRESH_BINARY)
                maskB = np.uint8(maskB) * 255
                op = np.zeros((w,h,3), dtype=np.uint8)
                ################## Tiling #####################
                id = []
                tile = []
                tile = []
                count = 0
                for row in range(0, w-511, 512):
                        for col in range(0,  h-511, 512):
                                tileM = maskB[row:row + 512, col:col + 512]
                                if np.sum(tileM):
                                    tile.append(image[row:row+512, col:col+512,0])
                                    tile.append(image[row:row+512, col:col+512,1])
                                    id.append(count)
                                    count += 1
                                    id.append(count)
                                    count += 1
                # tiling_time = time.time()
                ############# DM Parrallel ###############
                # print(count)
                # p = Pool(16)
                # dm_opL = p.starmap(dm_fn, zip(tile, id))
                # p.close()
                # p.join()
                count = 0
                # dm_time = time.time()
                # fact = np.int(16)
                ###################### Tile - wise Processing #################################
                for row in range(0, w-511, 512):
                    for col in range(0,  h-511, 512):
                        tileM = maskB[row:row + 512, col:col + 512]
                        # maskN = np.zeros((512,512),dtype='uint8')
                        maskRGB = np.zeros((512,512,3),dtype='uint8')
                        if np.sum(tileM):
                ############################# Channel Red ##############################################################
                            ##################### ALBU ###################
                            maskN = np.zeros((512,512),dtype='uint8')
                            tileN = np.zeros((512,512,3)).astype(np.uint8)
                            arr = tile[count]//fact
                            tileN[:, :, 0] = np.uint8(arr)
                            albu_op1 = albu_dingkang.predict(models_albu, tileN, image_type='8_bit_RGB')
                            ##################### DM++ ###################
                            # out_img = tsting_single_cal.testImages(albu_op1, dm_opL[count], model_dmp)
                            _, thresh1 = cv2.threshold(albu_op1,lThresh*255,255,cv2.THRESH_BINARY)
                            # print(thresh1.nonzero())
                            maskN[thresh1.nonzero()] = 255
                            maskRGB[:,:,1] = maskN
                            maskRGB[:,:,2] = maskN
                            count = count + 1
                ############################ Channel Green ######################################################
                            ###################### ALBU ###################
                            maskN = np.zeros((512,512),dtype='uint8')
                            tileN = np.zeros((512,512,3)).astype(np.uint8)
                            arr = tile[count]//fact
                            tileN[:, :, 1] = np.uint8(arr)
                            albu_op2 = albu_dingkang.predict(models_albu, tileN, image_type='8_bit_RGB')
                            ##################### DM++ ###################
                            # out_img = tsting_single_cal.testImages(albu_op2, dm_opL[count], model_dmp)
                            _, thresh2 = cv2.threshold(albu_op2,lThresh*255,255,cv2.THRESH_BINARY)
                            # print(thresh2.nonzero())
                            maskN[thresh2.nonzero()] = 255
                            maskRGB[:,:,0] = maskN
                            maskRGB[:,:,2] = np.clip(maskRGB[:,:,2] + maskN,0, 255)
                            # maskN = np.clip(maskN,0,13)
                            count = count + 1
                            #################### Stitching ###############
                            op[row:row+512, col:col+512,:] = maskRGB
                # pred_time = time.time()
        #################### write back Mask ###############
                # cv2.imwrite(outDir + "/M_" + f1, np.uint8(op))
                # write_time = time.time()
                # print(np.unique(op))                
                # op1 = np.zeros((w,h), dtype= np.uint8)
                # labels, num = measure.label(op, return_num=True, connectivity=2)
                # # print(num)
                # props = measure.regionprops(labels)
                # for prop in props:
                #     circ = (4*prop.area* 3.14)/(prop.perimeter*prop.perimeter)
                #     if circ < 0.5:
                #         if prop.area > 64:
                #                 for coord in prop.coords:
                #                         op1[coord[0], coord[1]] = op[coord[0], coord[1]]              
                cv2.imwrite(outDir + f1, np.uint8(op))


testImages(fileList1, fileList2, filePath, outDir)

