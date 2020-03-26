
import os
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
import torch.nn.functional as F
from torch.serialization import SourceChangeWarning
import warnings
from skimage.io import imread
from PIL import Image

# torch.backends.cudnn.enabled = False
# torch.cuda.clear_memory_allocated()

def flip_tensor_lr(batch):
    columns = batch.data.size()[-1]
    index = torch.autograd.Variable(torch.LongTensor(list(reversed(range(columns)))).cuda())
    return batch.index_select(3, index)


def flip_tensor_ud(batch):
    rows = batch.data.size()[-2]
    index = torch.autograd.Variable(torch.LongTensor(list(reversed(range(rows)))).cuda())
    return batch.index_select(2, index)


def to_numpy(batch):
    return np.moveaxis(batch.data.cpu().numpy(), 1, -1)


def _8bitGray2Input(img_arr):
    return np.repeat(np.expand_dims(img_arr, axis=2), 3, axis=2).astype(np.uint8)


def _16bitGray2Input(img_arr):
    return np.repeat(np.expand_dims(img_arr, axis=2) / 256, 3, axis=2).astype(np.uint8)


def _8bitRGB2Input(img_arr):
    return img_arr

def _12bitRGB2Input(img_arr):
    return (img_arr / 16).astype(np.uint8)


def img_to_tensor(im):
    return torch.from_numpy(np.expand_dims(np.moveaxis(im / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32), axis=0))

#def recursion_change_bn(module):
#    if isinstance(module, torch.nn.BatchNorm2d):
#        module.track_running_stats = 1
#    else:
#        for i, (name, module1) in enumerate(module._modules.items()):
#            module1 = recursion_change_bn(module1)
#
#    if isinstance(module, torch.nn.Conv2d):
#        module.padding_mode = 1
#    else:
#        for i, (name, module1) in enumerate(module._modules.items()):
#            module1 = recursion_change_bn(module1)
#
#
#    if isinstance(module, torch.nn.Upsample):
#        module.name = 1
#        module.align_corners = False
#    else:
#        for i, (name1, module1) in enumerate(module._modules.items()):
#            module1 = recursion_change_bn(module1)
#    return module



# Load all four models, return a list of these models.
def read_model(model_paths):
    # torch.cuda.clear_memory_allocated
    with warnings.catch_warnings():
        models = []
        for model_path in model_paths:
            print('start')
            warnings.simplefilter('ignore', SourceChangeWarning)
            model = torch.load(model_path)
            print(model_path)
#            for i, (name, module) in enumerate(model._modules.items()):
#                module = recursion_change_bn(model)
            model.eval()
            models.append(model)
        assert len(models) == 4
        return models


# Do prediction, take a (16-bit) 512*512 image numpy array, return a 512*512 8-bit image array with dtype=uint8.
# 16_bit_gray, 8_bit_gray, 8_bit_RGB
def predict(models, img_arr, image_type='12_bit_RGB'):
    # pdb.set_trace()
    conversion = {'16_bit_gray':_16bitGray2Input, '8_bit_gray':_8bitGray2Input, '8_bit_RGB':_8bitRGB2Input, '12_bit_RGB':_12bitRGB2Input}
    assert(image_type in conversion.keys())
    rgb_img_arr = conversion[image_type](img_arr)
    # with torch.nograd():
    batch = torch.autograd.Variable(img_to_tensor(rgb_img_arr)).cuda()
    # print(batch.shape)
    ret_arr = []
    for model in models:
        pred1 = F.sigmoid(model(batch))
        pred2 = flip_tensor_lr(model(flip_tensor_lr(batch)))
        pred3 = flip_tensor_ud(model(flip_tensor_ud(batch)))
        pred4 = flip_tensor_ud(flip_tensor_lr(model(flip_tensor_ud(flip_tensor_lr(batch)))))

        masks = [pred1, pred2, pred3, pred4]
        masks = list(map(F.sigmoid, masks))
        new_mask = torch.mean(torch.stack(masks, 0), 0)
        ret_arr.append(to_numpy(new_mask))
    merged = np.mean(ret_arr, axis=0)
    merged = np.squeeze(merged)
    return (merged * 255).astype(np.uint8)

















# numpy array?
# def img2RGB(img):
#     pixels = img.load()
#     for i in range(img.size[0]):
#         for j in range(img.size[1]):
#             pixels[i, j] = pixels[i, j] // 256
#     rgbimg = Image.new('RGBA', img.size)
#     rgbimg.paste(img)
#     return np.asarray(rgbimg)[..., :-1]

# demo for test
# if __name__ == '__main__':
    # model_paths = ['fold{}_best.pth'.format(i) for i in range(4)]
    # img_path = 'Sec131X2574Y2906_row.tif'
    # output_path = 'output.tif'
    # print(model_paths)
    # img = Image.open(img_path)
    # img = img_to_tensor(img2RGB(img))
    # # img = img_to_tensor(imread(img_path, mode='RGB'))
    #
    # # print(img.cpu().numpy())
    # models = read_model(model_paths)
    # ret = predict(models, img)
    # # print(ret)
    # ret = np.squeeze(ret)
    # final_ret = (ret * 255).astype(np.uint8)
    # cv2.imwrite(output_path, final_ret)
