import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import sys
import os
from sklearn.neighbors import NearestNeighbors
import networkx as nx

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





# input node tuples, return a 8-neighbor graph.
def __get_neighbor_graph(node_list):
    G = nx.Graph()
    G.add_nodes_from(node_list)
    neighbors = [(0, -1), (0, 1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    for node in node_list:
        x, y = node
        for move in neighbors:
            dx, dy = move
            newx, newy = x + dx, y + dy
            if (newx, newy) in node_list:
                G.add_edge((x, y), (newx, newy))
    return G



# compute the connectivity metric given the ground truth and prediction
# 1). we randomly selected $N pairs of CONNECTED points (a, b) from the foreground pixels in ground truth picture.
# 2). Then we found the nearest neighbors (a', b') of these two points in the predictions.
# Input: ground_truth_path, prediction path, threshold for foreground pixels, #random pairs $N.
# Output: the image_score, if returns None, means this image is meaningless (no pixels in pred.)
def get_connectivity(gt_path, prop_path, threshold=128, N=100, Suppress=True):
    if Suppress:
        sys.stdout = open(os.devnull, 'a')
    gt_img = Image.open(gt_path)
    prop_img = Image.open(prop_path)
    gt_points, prop_points = __get_image_point(gt_img, 0.1), __get_image_point(prop_img, threshold)
    print('gt: ', gt_path, ', have #', len(gt_points), 'points.')
    print('prop: ', prop_path, ', have #', len(prop_points), 'points.')

    # if there is no foreground in the label, skip this image.
    if len(gt_points) <= 1:
        sys.stdout = sys.__stdout__
        return None
    # otherwise, if no foreground in the prediction, return score = 0.
    if len(prop_points) <= 1:
        sys.stdout = sys.__stdout__
        return 0

    gt_pset = set(gt_points)
    prop_pset = set(prop_points)

    # Initialize 8-neighbor graph.
    gt_graph, prop_graph = __get_neighbor_graph(gt_pset), __get_neighbor_graph(prop_pset)
    # calculate num of components
    print('ground truth has ', nx.number_connected_components(gt_graph), 'components.')
    print('prop has ', nx.number_connected_components(prop_graph), 'components.')

    # init nearest-neighbor ball tree of the prediction image
    prop_nntree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(prop_points)

    gt_components = [list(c) for c in nx.connected_components(gt_graph)]
    gt_probabilities = [len(c)/len(gt_graph) for c in gt_components]
    choices = np.random.choice(len(gt_probabilities), N, p=gt_probabilities)
    image_score = 0
    for choice in choices:
        component = gt_components[choice]
        comp_length = len(component)
        random_pair = np.random.choice(comp_length, 2).tolist()
        a_gt, b_gt = [component[i] for i in random_pair]
        _, ab_pred = prop_nntree.kneighbors([a_gt, b_gt])
        a_prop, b_prop = prop_points[ab_pred[0][0]], prop_points[ab_pred[1][0]]
        # check if a_pred, b_pred is connected.
        image_score += 1 if nx.has_path(prop_graph, a_prop, b_prop) else 0
    sys.stdout = sys.__stdout__
    return image_score / N
