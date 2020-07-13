import matplotlib
matplotlib.use('agg')

from scipy.ndimage import gaussian_filter
import sys
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay


def build_vert_by_th(im_cube, nx, ny):
    vertex = []
    for j in range(ny):
        for i in range(nx):
            vertex.append([i, j, im_cube[i, j]])
    vertex = np.asarray(vertex)
    return vertex


def buildTriFromTetra(tetra):
    tri = {}
    nTe = tetra.shape[0]
    tri_index = 0
    for i in range(nTe):
        print('tetra:', i)
        for j in range(4):
            # Four triangles
            newTri = []
            for k in range(4):
                # Triangles' three vertices
                if k != j:
                    newTri.append(tetra[i, k])
            newTri = tuple(newTri)
            if newTri not in tri:
                # Add new triangles
                tri[newTri] = tri_index
                tri_index = tri_index + 1

    # Convert everything into list
    nTri = len(tri)
    tri_array = np.zeros([nTri, 3])
    for key, value in tri.items():
        tri_array[value, :] = list(key)
    return tri_array


def builEdgeFromTri(tri):
    edge = {}
    edge_index = 0
    nTri = len(tri)

    for i in range(nTri):
        print('tri:', i)
        for j in range(3):
            # 3 edges
            newEdge = []
            for k in range(3):
                if k != j:
                    newEdge.append(tri[i, k])
            newEdge = tuple(newEdge)
            if newEdge not in edge:
                edge[newEdge] = edge_index
                edge_index = edge_index + 1

    nEdge = len(edge)
    edge_array = np.zeros([nEdge, 2])
    for key, value in edge.items():
        edge_array[value, :] = list(key)

    return edge_array


def outBinary(vert, edge, triangle, nV, nE, nT,file_name):
    open(file_name, 'wb').close()
    with open(file_name, 'wb') as f:
        nV.astype(np.int32).tofile(f)
        vert.astype('d').tofile(f)
        nE.astype(np.int32).tofile(f)
        edge.astype(np.int32).tofile(f)
        nT.astype(np.int32).tofile(f)
        triangle.astype(np.int32).tofile(f)
    f.close()


input_filename = sys.argv[1]
out_dir = sys.argv[2]
if len(sys.argv) > 3:
    gauss = int(sys.argv[3])
else:
    gauss = 0


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print(out_dir)

'''
name_list = [name for name in os.listdir(d_name) if
             (os.path.isfile(d_name + '/' + name)) and (name != ".DS_Store")]
name_list.sort()
#nFile = len(name_list)

# nFile=len([name for name in os.listdir('data/'+d_name) if os.path.isfile('data/'+d_name+'/'+name)])
im = plt.imread(d_name + "/" + name_list[0])
#im = plt.imread("data/"+d_name+"/1.tif")
'''

img = plt.imread(input_filename)
nx, ny = img.shape

if gauss > 0:
    img = gaussian_filter(img, gauss)

# for i in range(1,nFile+1):
#     fileName = "data/"+d_name+"/"
#     fileName = fileName+str(i)+".tif"
#     im_cube[:,:,i-1] = plt.imread(fileName)

vert = build_vert_by_th(img, nx, ny)
print('verts:', len(vert))
sys.stdout.flush()

base_square_vert = np.asarray([vert[0], vert[1], vert[nx], vert[nx + 1]])
tri_vert_to_og_vert = {}
tri_vert_to_og_vert[0] = 0
tri_vert_to_og_vert[1] = 1
tri_vert_to_og_vert[2] = nx
tri_vert_to_og_vert[3] = nx + 1

# print(base_cube)
tri = Delaunay(base_square_vert[:,:2])
tri.simplices.sort()
# np.savetxt("simpliecs.txt",tri.simplices)
print("Build tri from tetra.")
sys.stdout.flush()
base_square_tri = tri.simplices
print("Build edge from tri.")
sys.stdout.flush()
base_square_edge = builEdgeFromTri(base_square_tri)

print('base edges:', base_square_edge)
print('base tris:', base_square_tri)

# print(base_cube_edge)
square_edge = []
for e in base_square_edge:
    new_e = [tri_vert_to_og_vert[e[0]], tri_vert_to_og_vert[e[1]]]
    square_edge.append(new_e)

square_tri = []
for t in base_square_tri:
    new_t = [tri_vert_to_og_vert[t[0]], tri_vert_to_og_vert[t[1]], tri_vert_to_og_vert[t[2]]]
    square_tri.append(new_t)
# print(cube_edge)

print('creating dups for row...')
sys.stdout.flush()
edges_to_dup = []
for e in square_edge:
    if (e[0] == 0 or e[0] == nx) and (e[1] == 0 or e[1] == nx):
        continue
    edges_to_dup.append(e)

tris_to_dup = []
for t in square_tri:
    tris_to_dup.append(t)

# print(edges_to_dup)
print('creating row...')
sys.stdout.flush()
row_edge = []
for e in square_edge:
    row_edge.append(e)

row_tri = []
for t in square_tri:
    row_tri.append(t)

for i in range(1, nx - 1):
    # print('working on', i)
    for e in edges_to_dup:
        new_e = [e[0] + i, e[1] + i]
        row_edge.append(new_e)
    for t in tris_to_dup:
        new_t = [t[0] + i, t[1] + i, t[2] + i]
        row_tri.append(new_t)

'''
for e in row_edge:
    print(e)
'''

print('creating plane...')
sys.stdout.flush()
edges_to_dup = []
for e in row_edge:
    v0_y = vert[e[0]][1]
    v1_y = vert[e[1]][1]
    if v0_y == 0 and v1_y == 0:
        continue
    edges_to_dup.append(e)

tris_to_dup = []
for t in row_tri:
    '''
    v0_y = vert[t[0]][1]
    v1_y = vert[t[1]][1]
    v2_y = vert[t[2]][1]
    if v0_y == 0 and v1_y == 0 and v2_y == 0:
        continue
    '''
    tris_to_dup.append(t)

'''
vert_filename = out_dir + '/vert.txt'
edge_filename = out_dir + '/edge.txt'

with open(vert_filename, 'w') as vert_file:
    for v in vert:
        vert_file.write(str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')
    vert_file.close()

with open(edge_filename, 'w') as edge_file:
    for e in edges_to_dup:
        edge_file.write(str(e[0]) + ' ' + str(e[1]) + '\n')
    edge_file.close()
'''
plane_edge = []
for e in row_edge:
    plane_edge.append(e)

plane_tri = []
for t in row_tri:
    plane_tri.append(t)

for i in range(1, ny - 1):
#for i in range(1, 2):
    # print('working on', i)
    shift = i * nx
    for e in edges_to_dup:
        new_e = [e[0] + shift, e[1] + shift]
        plane_edge.append(new_e)
    for t in tris_to_dup:
        new_t = [t[0] + shift, t[1] + shift, t[2] + shift]
        plane_tri.append(new_t)

'''
print('working on entire domain!')
sys.stdout.flush()
border = nx * ny

edges_to_dup = []
for e in plane_edge:
    if e[0] < border and e[1] < border:
        continue
    edges_to_dup.append(e)

tris_to_dup = []
for t in plane_tri:
    if t[0] < border and t[1] < border and t[2] < border:
        continue
    tris_to_dup.append(t)

edge = []
for e in plane_edge:
    edge.append(e)

tri = []
for t in plane_tri:
    tri.append(t)

for i in range(1, nz - 1):
    print('working on plane', i)
    sys.stdout.flush()
    shift = i * border
    for e in edges_to_dup:
        new_e = [e[0] + shift, e[1] + shift]
        edge.append(new_e)
    for t in tris_to_dup:
        new_t = [t[0] + shift, t[1] + shift, t[2] + shift]
        tri.append(new_t)
'''

edge = np.asarray(plane_edge)
tri = np.asarray(plane_tri)

nV = vert.shape[0]*np.ones(1)
nE = edge.shape[0]*np.ones(1)
nT = tri.shape[0]*np.ones(1)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print('writing binary...')
bi_file_name = out_dir+"/SC.bin"
outBinary(vert, edge, tri, nV, nE, nT, bi_file_name)

'''
vert_filename = out_dir + '/vert.txt'
edge_filename = out_dir + '/edge.txt'
tri_filename = out_dir + '/tri.txt'

with open(vert_filename, 'w') as vert_file:
    for v in vert:
        vert_file.write(str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')
    vert_file.close()

with open(edge_filename, 'w') as edge_file:
    for e in edge:
        edge_file.write(str(e[0]) + ' ' + str(e[1]) + '\n')
    edge_file.close()

with open(tri_filename, 'w') as tri_file:
    for t in tri:
        tri_file.write(str(t[0]) + ' ' + str(t[1]) + ' ' + str(t[2]) + '\n')
    tri_file.close()
'''

# [nx,ny,nz] = im_cube.shape
# print(len(vert),nx*ny*nz)
# im_cube_1d = np.reshape(im_cube,nx*ny*nz)
# plt.hist(vert[:,3], bins='auto')
# plt.savefig("histogram.png")
# im_cube_1d.sort()
# plt.clf()
# plt.plot(im_cube_1d)
# plt.savefig("plot.png")
