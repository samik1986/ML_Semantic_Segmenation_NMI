import csv
import math
from matplotlib import image as mpimg
import numpy as np
from os.path import join
from PIL import Image, ImageDraw
import PIL
import shutil
import sys

PIL.Image.MAX_IMAGE_PIXELS = None

IMAGE_FILENAME = sys.argv[1]
INPUT_FOLDER = sys.argv[2]
ALGO_THRESH = int(sys.argv[3]) / 100
LEN_THRESH = float(sys.argv[4])
# MAX_INT = 2000
# Vars
arg_len = len(sys.argv)

if arg_len < 6:
    ALPHA = 0
else:
    ALPHA = float(sys.argv[5])

if arg_len < 7:
    PATH_RADIUS = 2
else:
    PATH_RADIUS = int(sys.argv[6])
if arg_len < 8:
    print('crap')
    sys.exit()
    MAX_INT = 1.0
else:
    MAX_INT_PERC = int(sys.argv[7])

output_dir = sys.argv[8]


id = int(INPUT_FOLDER[INPUT_FOLDER.find('/')+1:len(INPUT_FOLDER)-1])
#sys.stdout.flush()
# print('Max intensity for line integral is:', MAX_INT)
#sys.stdout.flush()

# Files
VERT_FILENAME = join(INPUT_FOLDER, 'vert.txt')
PATH_FILENAME = join(INPUT_FOLDER, 'paths.txt')
GT_FILENAME = join(INPUT_FOLDER, 'diffusion_gt.txt')
DOMAIN_FILENAME = join(INPUT_FOLDER, 'diffusion_domain.txt')
pe_filename = join(output_dir, str(id) + '_simplify_edges.txt')

shutil.copyfile(VERT_FILENAME, output_dir + str(id)  + '_vert.txt' )



def compute_abs_cos_angle(v1, v2):
    v1_array = np.asarray(v1)
    v2_array = np.asarray(v2)

    if np.linalg.norm(v1_array) == 0 or np.linalg.norm(v2_array) == 0:
        return 0

    v1_unit = v1_array / np.linalg.norm(v1_array)
    v2_unit = v2_array / np.linalg.norm(v2_array)
    angle = np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0))
    cos = math.cos(angle)
    return abs(cos)


def compute_tangents(p):
    estimated_tangents = []
    for i in range(len(p)):
        left = max(0, i - PATH_RADIUS)
        right = min(i + PATH_RADIUS, len(p) - 1)
        lv = p[left]
        rv = p[right]
        vector = (lv[0] - rv[0], lv[1] - rv[1])
        estimated_tangents.append(vector)
    return estimated_tangents


def dist(p1, p2):
    return (((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2)) ** .5


def line_function(val, cos):
    capped_val = min(val, MAX_INT)
    return (ALPHA + cos) * capped_val


def length(p):
    sum = 0
    for i in range(len(p) - 1):
        i_dist = dist(p[i], p[i + 1])
        sum += i_dist
    return sum


def make_png(paths, output_path, st, lens):
    with open(output_path, 'w') as output_file:
        for i in range(len(paths)):
            p = paths[i]
            for j in range(len(p) - 1):
                v1 = p[j]
                v2 = p[j + 1]
                output_file.write(str(v1) + ' ' + str(v2) + ' ' + str(st[i])  + '\n')

    '''
    good_paths = []
    bad_paths = []
    bad_len = []
    
    for i in range(len(paths)):
        path = paths[i]

        if st[i] > ALGO_THRESH and st[i] > 0:
            good_paths.append(path)
        else:
            bad_paths.append(path)
            bad_len.append(lens[i])

    con_paths = []
    for i in range(len(bad_paths)):
        path = bad_paths[i]
        p_len = bad_len[i]
        start = path[0]
        end = path[len(path)-1]
        g_start = [pa for pa in good_paths if pa[0] == start or pa[len(pa)-1] == start]
        g_end = [pa for pa in good_paths if pa[0] == end or pa[len(pa)-1] == end]
        if p_len < LEN_THRESH and len(g_start) > 0 and len(g_end) > 0 and (len(g_start) == 1 or len(g_end) == 1):
            con_paths.append(path)

    with open(output_path, 'w') as output_file:
        for p in good_paths:
            for i in range(len(p) - 1):
                v1 = p[i]
                v2 = p[i + 1]
                output_file.write(str(v1) + ' ' + str(v2) + '\n')

        for p in con_paths:
            for i in range(len(p) - 1):
                v1 = p[i]
                v2 = p[i + 1]
                output_file.write(str(v1) + ' ' + str(v2) + '\n')

        output_file.close()
    '''

verts = []
#sys.stdout.flush()
print('reading in verts...')
#sys.stdout.flush()
with open(VERT_FILENAME, 'r') as vert_file:
    reader = csv.reader(vert_file, delimiter=' ')
    for line in reader:
        verts.append((int(line[0]), int(line[1])))
    vert_file.close()
#sys.stdout.flush()
print('VERTS', len(verts))
#sys.stdout.flush()

'''
nedges = []
print('reading in verts...')
with open(EDGE_FILENAME, 'r') as edge_file:
    reader = csv.reader(edge_file, delimiter=' ')
    for line in reader:
        nedges.append((int(line[0]), int(line[1])))
    edge_file.close()
'''
'''
for e in nedges:
    print(e)
    assert(30 >= dist(verts[e[0]], verts[e[1]]))
'''

intensity = {}
#sys.stdout.flush()
print('reading image...')
#sys.stdout.flush()
input_img = mpimg.imread(IMAGE_FILENAME)
#max_val = np.max(input_img)
#MAX_INT = 2000.0 / max_val
#print('MAX', max_val)
filtered_img = input_img

nr, nc = input_img.shape

freq = [0 for i in range(256)]

for i in range(nr):
    for j in range(nc):
        val = input_img[i,j]
        # print('val:', val)

'''
for i in range(len(freq)):
    print(i,':', freq[i])
'''

#MAX_INT = np.max(filtered_img)
perc = np.percentile(filtered_img, MAX_INT_PERC)

above = []
for i in range(nr):
    for j in range(nc):
        f = filtered_img[i,j]
        if f >= perc:
            above.append(f)
MAX_INT = int(sum(above) / len(above)) + 8

#MAX_INT = perc

#scaled_input_img = input_img/max_val
#filtered_img = scaled_input_img
# filtered_img = gaussian_filter(scaled_input_img, SIGMA)
for v in verts:
    x = v[0]
    y = v[1]
    f = filtered_img[x, y]
    # print('f', f)
    intensity[v] = f
del input_img
#del scaled_input_img
del filtered_img

raw_paths = []
#sys.stdout.flush()
print('reading in paths...')
#sys.stdout.flush()
lines = [line.rstrip('\n').split(' ') for line in open(PATH_FILENAME)]

for i in range(len(lines)):
    lines[i] = lines[i][:len(lines[i]) - 1]

for line in lines:
    raw_paths.append([int(x) for x in line])
# paths = raw_paths

'''
for e in nedges:
    assert(len([p for p in raw_paths if e[0] in p and e[1] in p]) == 1)


test_edge = nedges[0]
reflect_edge = (test_edge[1], test_edge[0])
'''

paths = []
#sys.stdout.flush()
print('computing valid paths...')
#sys.stdout.flush()
for p in raw_paths:
    if len(p) <= 1:
        print('path of len 1 or 0')
        sys.exit()

    edges = [(p[i], p[i+1]) for i in range(len(p) - 1)]

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
        p1 = p[:b+1]
        p2 = p[b+1:]
        if len(p1) > 1:
            paths.append(p1)
        if len(p2) > 1:
            paths.append(p2)
        continue

    # print('bad inds', bad_ind)

    b0 = bad_ind[0]
    bl = bad_ind[num_bad-1]
    p1 = p[:b0 + 1]
    p2 = p[bl + 1:]

    if len(p1) > 1:
        paths.append(p1)
    if len(p2) > 1:
        paths.append(p2)

    if num_bad == 2:
        if len(p[b0 + 1:bl + 1]) > 1:
            paths.append(p[b0+1:bl+1])

    # if num bad == 2 then range is empty
    # print('range:', [i for i in range(1,len(bad_ind) - 1)])
    for i in range(1,len(bad_ind) - 1):
        bim1 = bad_ind[i-1]
        bi = bad_ind[i]
        bip1 = bad_ind[i+1]
        p1 = p[bim1 + 1:bi+1]
        p2 = p[bi + 1: bip1+1]
        if len(p1) > 1:
            paths.append(p1)
        if len(p2) > 1:
            paths.append(p2)

#sys.stdout.flush()
print(len(paths), 'valid paths')
#sys.stdout.flush()
'''
for e in nedges:
    if len([p for p in paths if e[0] in p and e[1] in p]) == 1:
        print('good!')
'''
# sys.exit()

#sys.stdout.flush()
print('reading domain...')
#sys.stdout.flush()
domain = []
with open(DOMAIN_FILENAME, 'r') as domain_file:
    reader = csv.reader(domain_file, delimiter=' ')
    for line in reader:
        domain.append((int(line[0]), int(line[1])))
    domain_file.close()
#sys.stdout.flush()
print('DOMAIN', len(domain))
#sys.stdout.flush()

#sys.stdout.flush()
print('reading vectors...')
#sys.stdout.flush()
vectors = []
with open(GT_FILENAME, 'r') as gt_file:
    reader = csv.reader(gt_file, delimiter=' ')
    for line in reader:
        vectors.append((float(line[0]), float(line[1])))
    gt_file.close()

assert(len(domain) == len(vectors))

gt_dict = {}
#sys.stdout.flush()
print('building dict...')
#sys.stdout.flush()
for i in range(len(verts)):
    gt_dict[verts[i]] = vectors[i]

del domain
del vectors

#sys.stdout.flush()
print('ready to go!')
#sys.stdout.flush()
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
    degree_dict[path[len(path)-1]] += 1

    lengths.append(p_len)
    # print(p_len)
    f_vals = [intensity[v] for v in v_path]
    # print(f_vals)
    tangents = compute_tangents(v_path)
    abs_cosines = []
    for i in range(len(v_path)):
        v = v_path[i]
        gt = gt_dict[v]
        tangent = tangents[i]
        abs_cosines.append(compute_abs_cos_angle(gt, tangent))

    score = 0
    vec_score = 0
    for i in range(len(v_path) - 1):
        b1 = line_function(f_vals[i], abs_cosines[i])
        b2 = line_function(f_vals[i+1], abs_cosines[i+1])
        h = dist(v_path[i], v_path[i+1])
        area = h * (b1 + b2) / 2

        # print(h, b1, b2)

        score += area

        b3 = abs_cosines[i]
        b4 = abs_cosines[i+1]
        area2 = h*(b3+b4) / 2
        vec_score += area2
    # print('score for path:', score/p_len)
    scores.append(score/p_len)

#sys.stdout.flush()
print('outputting...')
#sys.stdout.flush()

if len(lengths) == 0:
    with open(pe_filename, 'w') as output_file:
        output_file.close()
    print('NOTHING')
    sys.exit()

min_len = min(lengths)
max_len = max(lengths)
if max_len == min_len:
    normalized_lens = [1.1 for l in lengths]
else:
    normalized_lens = [(s - min_len) / (max_len - min_len) for s in lengths]

min_score = min(scores)
max_score = max(scores)
#sys.stdout.flush()

print('min max', min_score, max_score)
# sys.stdout.flush()

if min_score == 0 and max_score == 0:
    normalized_scores = [0 for s in scores]
else:
    normalized_scores = [s / ((ALPHA + 1) * MAX_INT)  for s in scores]

print('scores:', normalized_scores)
for s in normalized_scores:
    assert(0 <= s <= 1)
make_png(paths, pe_filename, normalized_scores, normalized_lens)

#sys.stdout.flush()
print('perc:', MAX_INT_PERC)
print('Max intensity for line integral is:', MAX_INT)
#sys.stdout.flush()
