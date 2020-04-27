from dk_metric import image_metrics
import os
from multiprocessing import Process, Lock, Manager
import numpy as np
import time
import sys



gt_folder = sys.argv[1]
prop_folder = sys.argv[2]
lock = Lock()
Thread_Cnt = 16
Threshold = 0.4 * 255

def cal_connectivity(files, l, threshold):
    # sTP, sFP, sFN, msTP, msFP, msFN
    start_time = time.time()
    score, n = 0, 0
    for i, f in enumerate(files):
        gt_path = os.path.join(gt_folder, f)
        prop_path = os.path.join(prop_folder, f)
        if i != 0 and i % 100 == 0:
            print(os.getpid(), i, 'th file... use', time.time() - start_time, 'seconds.')
        s = image_metrics.get_connectivity(gt_path, prop_path, threshold=threshold, N=100, Suppress=True)
        if s is not None:
            score += s;
            n += 1
    with lock:
        l[0] += score
        l[1] += n


if __name__ == '__main__':

    files = os.listdir(prop_folder)
    manager = Manager()

    # Score, Image_cnt
    ml = manager.list([0, 0])

    pool = []
    files_threads = np.array_split(files, Thread_Cnt)

    for i in range(Thread_Cnt):
        pool.append(Process(target=cal_connectivity, args=(files_threads[i].tolist(), ml, Threshold,)))
    for t in pool:
        t.start()
    for t in pool:
        t.join()

    total_score, n = list(ml)
    print(total_score / n)