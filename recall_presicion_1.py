# -*- coding: UTF-8 -*-
import numpy as np
import struct
from struct import *
from numpy.matlib import repmat,sort
import time
def fopen(path):
    fid = open(path, 'rb')
    dim = np.fromfile(fid, dtype=np.float32)
    return dim
##k为recall k的指标值
def compute_recall(query_label,k):
    q_num = len(query_label)

    recall_mean = np.zeros([q_num])
    presicion_mean = np.zeros([q_num])
    result_mtx = np.load("result.npy")
    for i in range(10000):
        print sum(result_mtx[:,i])
    ##截取前100个进行对比
    little_result = result_mtx[0:k,:]
    for q in range(q_num):
        qi = sum(little_result[:,q])
        total = sum(result_mtx[:,q])
        presicion = float(qi)/float(k)
        recall = float(qi)/float(total)
        presicion_mean[q] = presicion
        recall_mean[q] = recall

    recallmean = np.mean(recall_mean)
    presicionmean = np.mean(presicion_mean)
    return recallmean,presicionmean

if __name__ == '__main__':
    label = fopen('label.dat')
    label = np.reshape(label,[10000])
    recall,presicion = compute_recall(label,100)
    print ("recall is : %s" % recall)
    print ("presicion is : %s " % presicion)