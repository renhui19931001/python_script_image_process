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

def countMaxLabel(zhi):
    label = fopen('label.dat')
    label = np.reshape(label, [12878])
    label = [int(item) for item in label]
    Map = {}
    for num in label:
        if num not in Map:
            Map[num] = 1
        else:
            Map[num] += 1
    Map = sorted(Map.items(),key=lambda x:x[1],reverse=True)
    #取前100类输出作为比对对象
    out = []
    k = 0
    for index,value in enumerate(Map):
        out.append(value[0])
        k += 1
        if(k ==zhi): break
    return out
if __name__ == '__main__':
    usedLabel = countMaxLabel(10)
    label = fopen('label.dat')
    label = np.reshape(label,[12878])
    label = [int(item) for item in label]
    ##
    newData = []
    newLabel = []
    for i in range(len(label)):
        if label[i] in usedLabel:
            newLabel.append(label[i])
    recall,presicion = compute_recall(newLabel,5)
    print ("recall is : %s" % recall)
    print ("presicion is : %s " % presicion)