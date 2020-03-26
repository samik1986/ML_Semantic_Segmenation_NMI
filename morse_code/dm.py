import numpy as np
from scipy.ndimage import gaussian_filter
import sys
from scipy.spatial import Delaunay
import os
import subprocess
#from matplotlib import image as mpimg
from PIL import Image, ImageDraw
import shutil


GAUSS = 2
OUTDIR = 'dm/temp/'
PERSISTENCE_THRESHOLD = 512
l = 512
w = 512


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


def cmp_dm_img_tri2D(i_file_name, i_th):
    print(i_file_name)
    subprocess.check_call([r"spt_cpp/spt_cpp", i_file_name + '/SC.bin', i_file_name + "/", str(i_th), str(2)])
    print('process done')
    o_vert = np.loadtxt(i_file_name + "/vert.txt")
    o_edge = np.loadtxt(i_file_name + "/edge.txt")

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
        u = verts[int(e[0])]
        v = verts[int(e[1])]
        draw.line((u[0], u[1], v[0], v[1]), fill=255, width=linestroke)
    return np.asarray(im) / 255


def dm_cal(tile, id):
    img = tile
    nx, ny = img.shape
    if GAUSS > 0:
        img = gaussian_filter(img, GAUSS)
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
    tri = Delaunay(base_square_vert[:, :2])
    tri.simplices.sort()
    # np.savetxt("simpliecs.txt",tri.simplices)
    print("Build tri from tetra.")
    sys.stdout.flush()
    base_square_tri = tri.simplices
    print("Build edge from tri.")
    sys.stdout.flush()
    base_square_edge = builEdgeFromTri(base_square_tri)

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
        tris_to_dup.append(t)

    plane_edge = []
    for e in row_edge:
        plane_edge.append(e)

    plane_tri = []
    for t in row_tri:
        plane_tri.append(t)

    for i in range(1, ny - 1):
        # for i in range(1, 2):
        # print('working on', i)
        shift = i * nx
        for e in edges_to_dup:
            new_e = [e[0] + shift, e[1] + shift]
            plane_edge.append(new_e)
        for t in tris_to_dup:
            new_t = [t[0] + shift, t[1] + shift, t[2] + shift]
            plane_tri.append(new_t)

    edge = np.asarray(plane_edge)
    tri = np.asarray(plane_tri)

    nV = vert.shape[0] * np.ones(1)
    nE = edge.shape[0] * np.ones(1)
    nT = tri.shape[0] * np.ones(1)

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    file_name = OUTDIR + str(id) + '/'
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    print('writing binary...')
    bi_file_name = OUTDIR + str(id) + '/SC.bin'
    outBinary(vert, edge, tri, nV, nE, nT, bi_file_name)

    threshold = PERSISTENCE_THRESHOLD

    verts, edges = cmp_dm_img_tri2D(file_name, threshold)

    # verts = np.asarray([[v[1], v[0]] for v in verts])

    path = os.path.join(file_name, 'dimo.png')
    morse_tile = make_png(verts, edges, path, l, w)
    # morse_tile = mpimg.imread(path)
    shutil.rmtree(file_name)    
    return morse_tile
