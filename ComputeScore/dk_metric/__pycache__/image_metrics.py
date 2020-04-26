import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import sys
import os
from sklearn.neighbors import NearestNeighbors

def __get_image_point_gt(nor_img, threshold):
    pixels = np.asarray(nor_img)
    # print(pixels.max())
    point_list = np.argwhere(pixels > threshold).tolist()
    point_list = [tuple(ele) for ele in point_list]
    return point_list

def __get_image_point(nor_img, threshold):
    pixels = np.asarray(nor_img)
    # print(pixels.max())
    point_list = np.argwhere(pixels > threshold).tolist()
    point_list = [tuple(ele) for ele in point_list]
    return point_list

# Compute TP, FP, FN
def get_TP_FP_FN(gt_path, prop_path, threshold=128):
    # gt_nor, prop_nor = __normalize_image(gt_path), __normalize_image(prop_path)
    gt_img = Image.open(gt_path)
    prop_img = Image.open(prop_path)
    gt_points, prop_points = __get_image_point_gt(gt_img, 0.1), __get_image_point(prop_img, threshold)
    gt_pset = set(gt_points)
    prop_pset = set(prop_points)
    TP = len(gt_pset.intersection(prop_pset))
    FP = len(prop_pset.difference(gt_pset))
    FN = len(gt_pset.difference(prop_pset))
    return TP, FP, FN


# Compute Modified TP, FP, FN
# Modified TP: in prop, and if its NN in gt is no greater than  radius
# Modified FP: in prop, its NN in gt is not within radius, we guarantee that TP + NP = P
# Modified FN: in gt, its NN in gt is more than radius away.
# TN: Others
def get_mod_TP_FP_FN(gt_path, prop_path, radius=2, threshold=128):
    gt_img = Image.open(gt_path)
    # print(gt_img.getextrema())
    prop_img = Image.open(prop_path)
    # print(prop_img.getextrema())
    gt_points, prop_points = __get_image_point(gt_img, 0.1), __get_image_point(prop_img, threshold)

    # print('gt_points', gt_points, len(gt_points))
    # print('prop_points', prop_points, len(prop_points))

    if len(gt_points) == 0:
        return 0, len(prop_points), 0
    if len(prop_points) == 0:
        return 0, 0, len(gt_points)

    gt_nntree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(gt_points)
    prop_nntree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(prop_points)

    distances, _ = gt_nntree.kneighbors(prop_points)

    TP = len(np.argwhere(distances <= radius))
    FP = len(prop_points) - TP

    distances, _ = prop_nntree.kneighbors(gt_points)

    FN = len(np.argwhere(distances > radius))
    return TP, FP, FN





