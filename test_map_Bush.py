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

def compute_map(dis_mtx, query_label, database_label,newData):
    q_num = len(query_label)
    d_num = len(database_label)

    map_mean = np.zeros([q_num])
    ##得到一个每一列代表一个数据库，每个值代表着搜寻的结果
    database_label_mtx = np.tile(database_label,q_num)
    database_label_mtx = database_label_mtx.reshape([d_num,q_num])
    database_label_mtx = np.transpose(database_label_mtx)

    sorted_database_label_mtx = database_label_mtx
    idx_mtx = sortIndex(dis_mtx)
    ##这个是前10个的索引值
    topN_dix = idx_mtx[0:10,:]
    for i in range(1000):
        print "query_one :"

        query_one = newData[i]
        ##输出查询的值
        print i,
        print query_label[i],
        ##输出score值
        for item in query_one:
            print str(item)+" ",
        print '\n'
        print "result:"
        print "id    label   score    feature"
        for j in range(10):
            searchId = topN_dix[j,i]
            get_zhi = newData[searchId]
            print searchId,
            print database_label[searchId],
            print abs(dis_mtx[i][searchId]),
            print "   ",
            for item in get_zhi:
                print str(item),
            print ''

        print '\n'
        print '**********************************'
        print '\n'


def sortIndex(array):
    long = array.shape
    nrow = long[0]
    ncol = long[1]
    out = []
    array = np.array(array)
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
    juzhen = np.array(juzhen)
    compute_map(juzhen,newLabel,newLabel,newData)


