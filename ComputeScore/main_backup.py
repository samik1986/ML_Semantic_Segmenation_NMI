from dk_metric import image_metrics
import os
from multiprocessing import Process, Lock, Manager
import numpy as np
import time

gt_folder = './180405/180405_Label'
prop_folder = './180405/ALBU/train_merged/'
output_csv = './scores.csv'

radius = 3
files = os.listdir(prop_folder)
lock = Lock()

ALL_thresholds = []
ALL_precision, ALL_recall, ALL_F1, ALL_Jaccard, ALL_mod_prec, ALL_mod_recall, ALL_mod_F1 = [],[],[],[],[],[],[]
Thread_Cnt = 16
manager = Manager()

def cal_fp_tp(files, l, threshold):
    # sTP, sFP, sFN, msTP, msFP, msFN
    start_time = time.time()
    sTP, sFP, sFN, msTP, msFP, msFN = 0, 0, 0, 0, 0, 0
    for i, f in enumerate(files):
        gt_path = os.path.join(gt_folder, f.replace('_row_img', '_label'))
        prop_path = os.path.join(prop_folder, f)
        if i != 0 and i % 200 == 0:
            print(os.getpid(), i, 'th file... use', time.time() - start_time, 'seconds.')

        TP, FP, FN = image_metrics.get_TP_FP_FN(gt_path, prop_path, threshold=threshold)
        mTP, mFP, mFN = image_metrics.get_mod_TP_FP_FN(gt_path, prop_path, radius=radius, threshold=threshold)
        sTP += TP
        sFP += FP
        sFN += FN
        msTP += mTP
        msFP += mFP
        msFN += mFN
    with lock:
        l[0] += sTP
        l[1] += sFP
        l[2] += sFN
        l[3] += msTP
        l[4] += msFP
        l[5] += msFN


for step in range(10, 31):
    threshold = (0.025 * step + 0)
    ALL_thresholds.append(threshold)
    print('-------------', threshold, '-------------')
    threshold *= 255
    l = manager.list([0, 0, 0, 0, 0, 0])

    pool = []
    files_threads = np.array_split(files, Thread_Cnt)

    for i in range(Thread_Cnt):
        pool.append(Process(target=cal_fp_tp, args=(files_threads[i].tolist(), l, threshold,)))
    for t in pool:
        t.start()
    for t in pool:
        t.join()

    sTP, sFP, sFN, msTP, msFP, msFN = list(l)
    Precision = sTP / (sTP + sFP) if (sTP + sFP != 0) else 1
    Recall = sTP / (sTP + sFN) if(sTP + sFN != 0) else 1

    Jaccard = 1 / (1/Precision + 1/Recall - 1) if (Precision > 0 and Recall > 0) else 0
    F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision > 0 and Recall > 0) else 0
    
    ALL_precision.append(Precision)
    ALL_recall.append(Recall)
    ALL_Jaccard.append(Jaccard)
    ALL_F1.append(F1)

    mPrecision = msTP / (msTP + msFP) if (msTP + msFP != 0) else 1
    mRecall = msTP / (msTP + msFN)  if(msTP + msFN != 0) else 1
    mF1 = 2 * mPrecision * mRecall / (mPrecision + mRecall) if (mPrecision > 0 and mRecall > 0) else 0

    ALL_mod_prec.append(mPrecision)
    ALL_mod_recall.append(mRecall)
    ALL_mod_F1.append(mF1)
    

with open(output_csv, 'w') as output:
    data_thre = 'Threshold,' + ','.join(['{:3f}'.format(v) for v in ALL_thresholds])
    data_pre = 'Precision,' + ','.join(['{:3f}'.format(v) for v in ALL_precision])
    data_rec = 'Recall,' + ','.join(['{:3f}'.format(v) for v in ALL_recall])
    data_jac = 'Jaccard,' + ','.join(['{:3f}'.format(v) for v in ALL_Jaccard])
    data_f1 = 'F1,' + ','.join(['{:3f}'.format(v) for v in ALL_F1])
    data_mpre = 'Mod_Prec,' + ','.join(['{:3f}'.format(v) for v in ALL_mod_prec])
    data_mrec = 'Mod_Rec,' + ','.join(['{:3f}'.format(v) for v in ALL_mod_recall])  
    data_mf1 = 'Mod_F1,' + ','.join(['{:3f}'.format(v) for v in ALL_mod_F1])
    output.write('\n'.join([data_thre, data_pre, data_rec, data_jac, data_f1, data_mpre, data_mrec, data_mf1]))  

