import os
import sys
import cv2
import time
import random
import numpy

# data_dir='/media/zhao/新加卷/SRAD2018/validate'
data_dir='/home/jxzhao/tianchi/SRAD2018/test'

for u,i in enumerate(os.listdir(data_dir)):
    for j in os.listdir(os.path.join(data_dir,i)):
        try:
            assert len(os.listdir(os.path.join(data_dir,i,j)))==31
        except:
            print(os.path.join(data_dir,i,j))
            for k in os.listdir(os.path.join(data_dir,i,j)):
                if os.path.splitext(k)[1]!='.png':
                    print(k)
                    if os.path.splitext(k)[1]=='.db':
                        os.remove(os.path.join(data_dir,i,j,k))
    print(u)