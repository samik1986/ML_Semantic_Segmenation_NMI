from PIL import Image, ImageDraw
import os
import sys
import numpy as np
from scipy import misc
import scipy

def make_png(verts, edges, path, l, w, linestroke=1):
    im = Image.new('L', (l, w), color=0)
    draw = ImageDraw.Draw(im)
    #im = np.zeros([l, w])
    for e in edges:
        #print(len(verts), e[0], e[1])
        u, v = verts[e[0]], verts[e[1]]
        #im[u[0], u[1]] = 255
        #im[v[0], v[1]] = 255
        draw.line((u[0], u[1], v[0], v[1]), fill=255, width=linestroke)
        #break
    im.save(path, format='png')
    #scipy.misc.imsave(path, im)



if __name__ == '__main__':

    input_folder, l, w, output_name, final_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    input_vert = os.path.join(input_folder, 'vert.txt')
    input_edge = os.path.join(input_folder, 'edge.txt')
    # input_edge = os.path.join(input_folder, 'no_hole_edge.txt')
    l, w = int(l), int(w)
    verts, edges = [], []
    print('reading files...')
    with open(input_vert) as vfile:
        for line in vfile:
            v = line.strip().split()
            # verts.append((int(v[0]), int(v[1])))
            verts.append((int(v[1]), int(v[0])))

    with open(input_edge) as efile:
        for line in efile:
            e = line.strip().split()
            edges.append((int(e[0]), int(e[1])))
    print('done')

    dirpath = os.path.dirname(input_vert)
    path = os.path.join(final_dir, os.path.basename(output_name))
    make_png(verts, edges, path, l, w)
