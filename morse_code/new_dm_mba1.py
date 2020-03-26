import numpy as np
from scipy.ndimage import gaussian_filter
import sys
from scipy.spatial import Delaunay
import os
import subprocess
from matplotlib import image as mpimg
from PIL import Image, ImageDraw
import shutil
import cv2
import time


GAUSS = 2
OUTDIR = 'temp5/'
PERSISTENCE_THRESHOLD = 8
l = 512
w = 512

MAX_INT_PERC = 95


def output_edge_scores(paths, st):
    output_edge = []
    for i in range(len(paths)):
        p = paths[i]
        for j in range(len(p) - 1):
            v1 = p[j]
            v2 = p[j + 1]
            output_edge.append([v1, v2, st[i]])
    return output_edge


def dist(p1, p2):
    return (((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2)) ** .5


def line_function(val, cap):
    capped_val = min(val, cap)
    return capped_val


def length(p):
    sum = 0
    for i in range(len(p) - 1):
        i_dist = dist(p[i], p[i + 1])
        sum += i_dist
    return sum


def cmp_dm_img_tri2D(output_dir, input_filename):
    # print('running morse')
    t_start = time.time()
    subprocess.check_call([r"src/a.out", output_dir + 'input.png', output_dir, str(PERSISTENCE_THRESHOLD)])
    t_end = time.time()
    print("only morse:", t_end - t_start)
    # print('process done')
    o_vert = np.loadtxt(output_dir + "/vert.txt")
    o_edge = np.loadtxt(output_dir + "/edge.txt")

    if len(o_edge) == 0:
        return [], []

    o_vert = o_vert[:, :2]

    # visualize result
    stable_vert = o_vert.copy()
    stable_vert[:, 0] = o_vert[:, 1]
    stable_vert[:, 1] = o_vert[:, 0]
    o_vert = stable_vert

    return o_vert, o_edge


def make_png(verts, edges, path, l, w, linestroke=1):
    im = Image.new('L', (l, w), color=0)
    draw = ImageDraw.Draw(im)
    for e in edges:
        bright = int(e[2] * 255)
        u, v = verts[e[0]], verts[e[1]]
        draw.line((u[0], u[1], v[0], v[1]), fill=bright, width=linestroke)

    im.save(path, format='png')


def dm_cal(tile, id):
    start = time.time()
# def dm_cal():
    #print('begin testing')
    img = tile
    '''
    img = []
    for i in range(512):
        row = []
        for j in range(512):
            row.append(0)
        img.append(row)
    '''
    # img = mpimg.imread('imgs/test1.tif')
    # id = 3
    img = np.asarray(img)
    # print('img max:', np.max(img))
    nx, ny = img.shape
    '''
    for i in range(nx):
        for j in range(ny):
            print('py vert:', i, j, img[i, j])
    
    sys.exit()
    '''
    if GAUSS > 0:
        input_img = gaussian_filter(img, GAUSS)
    else:
        input_img = img

    # print('gauss img max:', np.max(input_img))

    input_img.astype('uint16')

    # print('creating dir')
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    id_dir = OUTDIR + str(id) + '/'
    if not os.path.exists(id_dir):
        os.makedirs(id_dir)

    input_filename = id_dir + '/input.png'

    # print('outputting img for morse')
    cv2.imwrite(input_filename, input_img)

    threshold = PERSISTENCE_THRESHOLD
    verts, edges = cmp_dm_img_tri2D(id_dir, input_filename)
    verts = [(int(v[0]), int(v[1])) for v in verts]

    end = time.time()
    print("old:", end - start)


    no_dup_edges = []
    for edge in edges:
        if edge[0] < edge[1]:
            e = [int(edge[0]), int(edge[1])]
        else:
            e = [int(edge[1]), int(edge[0])]
        if e not in no_dup_edges:
            no_dup_edges.append(e)

    no_dup_edge_filename = id_dir + "no_dup_edge.txt"
    with open(no_dup_edge_filename, 'w') as output_file:
        for e in no_dup_edges:
            output_file.write(str(e[0]) + ' ' + str(e[1]) + '\n')
        output_file.close()

    # sys.exit()

    subprocess.check_call([r"paths_src/a.out", id_dir])

    perc = np.percentile(img, MAX_INT_PERC)
    intensity = {}

    for v in verts:
        x = v[0]
        y = v[1]
        f = img[y, x]
        # print('f', f)
        intensity[v] = f

    # print('max int:', intensity.items())

    above = []
    for i in range(nx):
        for j in range(ny):
            f = img[i, j]
            if f >= perc:
                above.append(f)
    MAX_INT = int(sum(above) / len(above)) + 8

    raw_paths = []
    # sys.stdout.flush()
    # print('reading in paths...')
    # sys.stdout.flush()
    lines = [line.rstrip('\n').split(' ') for line in open(id_dir + 'paths.txt')]

    for i in range(len(lines)):
        lines[i] = lines[i][:len(lines[i]) - 1]

    for line in lines:
        raw_paths.append([int(x) for x in line])

    paths = []
    # sys.stdout.flush()
    # print('computing valid paths...')
    # sys.stdout.flush()
    for p in raw_paths:
        if len(p) <= 1:
            # print('path of len 1 or 0')
            sys.exit()

        edges = [(p[i], p[i + 1]) for i in range(len(p) - 1)]

        '''
        if test_edge not in edges and reflect_edge not in edges:
            continue


        print('edge:', test_edge)
        print('path', p)
        '''

        bad_ind = []
        for i in range(len(edges)):
            edge = edges[i]
            # print('edge:', edge)
            e_len = dist(verts[edge[0]], verts[edge[1]])
            if float(20) < e_len:
                bad_ind.append(i)
        num_bad = len(bad_ind)
        # print('bad:', num_bad)
        if num_bad == 0:
            paths.append(p)
            continue
        if num_bad == len(edges):
            continue
        if num_bad == 1:
            b = bad_ind[0]
            p1 = p[:b + 1]
            p2 = p[b + 1:]
            if len(p1) > 1:
                paths.append(p1)
            if len(p2) > 1:
                paths.append(p2)
            continue

        # print('bad inds', bad_ind)

        b0 = bad_ind[0]
        bl = bad_ind[num_bad - 1]
        p1 = p[:b0 + 1]
        p2 = p[bl + 1:]

        if len(p1) > 1:
            paths.append(p1)
        if len(p2) > 1:
            paths.append(p2)

        if num_bad == 2:
            if len(p[b0 + 1:bl + 1]) > 1:
                paths.append(p[b0 + 1:bl + 1])

        # if num bad == 2 then range is empty
        # print('range:', [i for i in range(1,len(bad_ind) - 1)])
        for i in range(1, len(bad_ind) - 1):
            bim1 = bad_ind[i - 1]
            bi = bad_ind[i]
            bip1 = bad_ind[i + 1]
            p1 = p[bim1 + 1:bi + 1]
            p2 = p[bi + 1: bip1 + 1]
            if len(p1) > 1:
                paths.append(p1)
            if len(p2) > 1:
                paths.append(p2)

    # sys.stdout.flush()
    # print(len(paths), 'valid paths')

    # sys.stdout.flush()
    # print('ready to go!')
    # sys.stdout.flush()
    scores = []
    lengths = []
    degree_dict = {}
    for i in range(len(verts)):
        degree_dict[i] = 0

    p_index = 0
    for path in paths:
        p_index += 1
        # print('path:', p_index,'/',len(paths))
        v_path = [verts[i] for i in path]
        # print(v_path)
        p_len = length(v_path)

        degree_dict[path[0]] += 1
        degree_dict[path[len(path) - 1]] += 1

        lengths.append(p_len)
        # print(p_len)
        f_vals = [intensity[v] for v in v_path]

        score = 0
        vec_score = 0
        for i in range(len(v_path) - 1):
            b1 = line_function(f_vals[i], MAX_INT)
            b2 = line_function(f_vals[i + 1], MAX_INT)
            h = dist(v_path[i], v_path[i + 1])
            area = h * (b1 + b2) / 2

            # print(h, b1, b2)

            score += area
        # print('score for path:', score/p_len)
        scores.append(score / p_len)

    # sys.stdout.flush()
    # print('outputting...')
    # sys.stdout.flush()
    
    if len(scores) == 0:
        # print('empty morse!')
        empty = []
        for i in range(nx):
            row = []
            for j in range(ny):
                row.append(0)
            empty.append(row)
        empty = np.asarray(empty)
        return empty

    min_score = min(scores)
    max_score = max(scores)
    # sys.stdout.flush()

    print('min max', min_score, max_score)
    # sys.stdout.flush()

    if min_score == 0 and max_score == 0:
        normalized_scores =    [0 for s in scores]
    else:
        normalized_scores = [s / MAX_INT for s in scores]

    # print('scores:', normalized_scores)
    for s in normalized_scores:
        assert (0 <= s <= 1)

    # pe_filename = id_dir + 'simplify_edges.txt'
    output_edges = output_edge_scores(paths, normalized_scores)
    output_path = id_dir + 'g_dimo.png'
    make_png(verts, output_edges, output_path, nx, ny, linestroke=1)
    morse_tile = mpimg.imread(output_path)
    return morse_tile

'''
if __name__ == '__main__':
    dm_cal()
'''
