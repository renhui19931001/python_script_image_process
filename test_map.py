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
    ##得到一个每一列代表一个数据库，每个值代表着搜寻的结果
    database_label_mtx = np.tile(database_label,q_num)
    database_label_mtx = database_label_mtx.reshape([d_num,q_num])
    database_label_mtx = np.transpose(database_label_mtx)

    sorted_database_label_mtx = database_label_mtx
    idx_mtx = sortIndex(dis_mtx)

    ##根据查询结果对label进行排序
    for i in range(q_num):
        sorted_database_label_mtx[:, i] = database_label_mtx[idx_mtx[:, i], i]

    result_mtx = []
    hang_database = np.transpose(sorted_database_label_mtx)
    for i in range(len(query_label)):
        ##对每一列判断，一样就等于1，否则为0
        out = [1 if item == query_label[i] else 0 for item in hang_database[i]]
        result_mtx.append(out)
        print ("已经处理到了第 %s 行" % i)
    result_mtx = np.transpose(np.array(result_mtx))
    np.save("result.npy", result_mtx)
    #result_mtx = (sorted_database_label_mtx == repmat(query_label.H, d_num, 1))

    ##这个是前10个的索引值
    topN_dix = idx_mtx[0:10,:]

    ##截取前100个进行对比
    little_result = result_mtx[0:10,:]
    little_resultT = np.transpose(little_result)
    for q in range(q_num):
        qi = sum(little_result[:,q])
        #fenzi = sum([0:10]/[k for k in len(result_mtx) if result_mtx[k,q] == 1])
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
    usedLabel = countMaxLabel(10)
    data = fopen('code.dat')
    data = np.reshape(data,[12878,12])
    label = fopen('label.dat')
    label = np.reshape(label,[12878])
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
    map = compute_map(juzhen,newLabel,newLabel)
    print map


