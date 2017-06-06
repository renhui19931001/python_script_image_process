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

def compute_map(dis_mtx, query_label, database_label):
    q_num = len(query_label)
    d_num = len(database_label)

    map_mean = np.zeros([q_num])
    result_mtx = np.load("result.npy")

    ##截取前100个进行对比
    little_result = result_mtx[0:10,:]
    little_resultT = np.transpose(little_result)
    for q in range(q_num):
        qi = sum(little_result[:,q])
        fenzi = [i for i in range(1,qi + 1)]
        fenmu = [j + 1 for j in range(len(little_resultT[q])) if little_resultT[q][j] == 1]
        sum1 = 0.0
        for i in range(qi):
            sum1 += float(fenzi[i]) / float(fenmu[i])
        map1 = sum1/float(10)
        map_mean[q] = map1
    mean = np.mean(map_mean)
    return mean

def sortIndex(array):
    long = array.shape
    nrow = long[0]
    ncol = long[1]
    out = []
    array = np.array(array)
    start = time.time()
    for i in range(ncol):
        index = np.lexsort([array[:, i]])
        out.append(index)
    out = np.transpose(out)
    return out

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
    usedLabel = countMaxLabel(100)
    data = fopen('code.dat')
    data = np.reshape(data, [12878, 12])
    label = fopen('label.dat')
    label = np.reshape(label, [12878])
    data = np.sign(data)
    label = [int(item) for item in label]
    ##
    newData = []
    newLabel = []
    for i in range(len(label)):
        if label[i] in usedLabel:
            newLabel.append(label[i])
            newData.append(data[i])
    newDataT = np.transpose(newData)
    juzhen = - np.mat(newData) * np.mat(newDataT)
    map = compute_map(juzhen, newLabel, newLabel)
    print map


