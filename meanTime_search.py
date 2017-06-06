#coding = utf-8
import numpy as np
import struct
from struct import *
from numpy.matlib import repmat,sort
import time

def fopen(path):
    fid = open(path, 'rb')
    dim = np.fromfile(fid, dtype=np.float32)
    return dim

##average search time computing
def sort1time(data,query):
    start = time.time()
    meanTotal = []
    for i in range(len(query)):
        code = np.array(query[i])
        code = code.reshape([12,1])
        fenlei = - np.mat(data) * np.mat(code)
        index = np.lexsort([fenlei])

        if i % 10 == 0 and i != 0:
            end = time.time()
            mean = end - start
            perTime = float(mean)/10.0
            meanTotal.append(perTime)
            print ("average retrieve 10 times for the time " + str(perTime))
            start = end
    print ("The average time per search is : %f" % np.mean(np.array(meanTotal)))


if __name__ == '__main__':
    data = fopen('code.dat')
    data = np.reshape(data,[10000,12])
    label = fopen('label.dat')
    label = np.reshape(label,[10000])
    data = np.sign(data)
    dataT = np.transpose(data)
    juzhen = - np.mat(data) * np.mat(dataT)
#   map = compute_map(juzhen,label,label)
    sort1time(data,data)