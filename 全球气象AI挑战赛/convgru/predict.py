import tensorflow
import os
import cv2
import numpy
import time
import zipfile
import skimage.io

# data_dir='/media/zhao/新加卷/SRAD2018/test/SRAD2018_Test_1'
data_dir='/home/jxzhao/tianchi/SRAD2018/test/SRAD2018_Test_1'
log_dir='log/'
model_dir='model/'

predict_dir=os.path.join(os.path.split(data_dir)[0],'predict')
if not os.path.exists(predict_dir):
    os.mkdir(predict_dir)

Saver =tensorflow.train.import_meta_graph(tensorflow.train.latest_checkpoint(model_dir)+'.meta')
Session=tensorflow.Session()
Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))

input_image=tensorflow.get_default_graph().get_tensor_by_name('origin_image:0')
image0=tensorflow.get_default_graph().get_tensor_by_name('decode/decode_image:0')
image1=tensorflow.get_default_graph().get_tensor_by_name('decode_1/decode_image:0')
image2=tensorflow.get_default_graph().get_tensor_by_name('decode_2/decode_image:0')
image3=tensorflow.get_default_graph().get_tensor_by_name('decode_3/decode_image:0')
image4=tensorflow.get_default_graph().get_tensor_by_name('decode_4/decode_image:0')
image5=tensorflow.get_default_graph().get_tensor_by_name('decode_5/decode_image:0')

all_rad=os.listdir(data_dir)
rads=[os.path.join(data_dir,x) for x in all_rad]
all_image_dir=[]
for x in rads:
    image_dir=[os.path.join(x,y) for y in os.listdir(x)]
    image_dir.sort()
    all_image_dir.append(image_dir)

for x in all_image_dir:
    k1=[]
    for y in x:
        k1.append(skimage.io.imread(y))
    k1=numpy.array(k1).reshape([31,501,501,1])
    predict0=Session.run(image0, feed_dict={input_image:k1[-10:]}).reshape(501,501,1)
    predict1=Session.run(image1, feed_dict={input_image:k1[-10:]}).reshape(501,501,1)
    predict2=Session.run(image2, feed_dict={input_image:k1[-10:]}).reshape(501,501,1)
    predict3=Session.run(image3, feed_dict={input_image:k1[-10:]}).reshape(501,501,1)
    predict4=Session.run(image4, feed_dict={input_image:k1[-10:]}).reshape(501,501,1)
    predict5=Session.run(image5, feed_dict={input_image:k1[-10:]}).reshape(501,501,1)
    folder_name=os.path.split(os.path.split(y)[0])[1]
    save_root=os.path.join(predict_dir,folder_name)
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    for z in range(6):
        name=folder_name+'_f00%s.png'%(z+1)
        cv2.imwrite(os.path.join(save_root,name), locals()['predict%s'%z])

zipfile.shutil.make_archive(base_name=predict_dir,format='zip',root_dir=predict_dir)