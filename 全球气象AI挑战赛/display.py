import os
import sys
import cv2
import time
import skimage.io
import random
import numpy
import pandas

for _ in range(3):

    # data_dir='E:/SRAD2018/train'
    data_dir='/media/zhao/新加卷/SRAD2018/train'
    all_file=os.listdir(data_dir)
    pick_one_file=random.sample(all_file,1)[0]
    one_file=os.path.join(data_dir,pick_one_file)
    one_all_rad=os.listdir(one_file)
    pick_one_rad=random.sample(one_all_rad,1)[0]
    one_rad=os.path.join(one_file,pick_one_rad)

    for root, dirs, img_files in os.walk(one_rad):
        all_img_files = list(map(lambda path: os.path.join(root, path), img_files))
        all_img_files.sort()

    for img_path in all_img_files:
        im = skimage.io.imread(img_path)
        im=pandas.DataFrame(im).where(im<=80,0).values
        cv2.imshow(os.path.join(pick_one_file,pick_one_rad), im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)
    cv2.destroyAllWindows()