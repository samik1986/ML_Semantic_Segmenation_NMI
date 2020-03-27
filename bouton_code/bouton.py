import sys
import csv
from matplotlib import image as mpimg
import numpy as np
import scipy.misc
import cv2


vert_filename = sys.argv[1]
edge_filename = sys.argv[2]
img_filename = sys.argv[3]
output_img_filename = sys.argv[4]
thresh = int(sys.argv[5])

print('reading in verts...')
verts = []
with open(vert_filename, 'r') as vert_file:
    reader = csv.reader(vert_file, delimiter=' ')
    for row in reader:
        v = [int(row[0]), int(row[1]), int(row[2])]
        verts.append(v)
    vert_file.close()
print('verts:', len(verts))

print('reading in edges...')
edges = []
with open(edge_filename, 'r') as edge_file:
    reader = csv.reader(edge_file, delimiter=' ')
    for row in reader:
        e = [int(row[0]), int(row[1])]
        edges.append(e)
    edge_file.close()
print('edges:', len(edges))

img = mpimg.imread(img_filename)

nx, ny = img.shape

output = []
for r in range(nx):
    row = []
    for c in range(ny):
        row.append(0)
    output.append(row)
output = np.asarray(output)

print('building adjacency')
adj = []
for i in range(len(verts)):
    adj.append([])

for e in edges:
    #print(e)
    v0 = e[0]
    v1 = e[1]
    adj[v0].append(v1)
    adj[v1].append(v0)


maxs = 0
print('building max')
for i in range(len(verts)):
    #print(i, adj[i])
    v = verts[i]
    f_v = img[v[0], v[1]]
    local_max = True

    for j in adj[i]:
        u = verts[j]
        f_u = img[u[0], u[1]]
        if f_u > f_v:
            local_max = False
            break

    if local_max and v[2] > thresh:
        maxs += 1
        # print('max at:', i, 'verts: (',v[1], v[0],') val:', f_v)
        output[v[0], v[1]] = 255

print('maxs:', maxs)
cv2.imwrite(output_img_filename, output)
#scipy.misc.imsave(output_img_filename, output)