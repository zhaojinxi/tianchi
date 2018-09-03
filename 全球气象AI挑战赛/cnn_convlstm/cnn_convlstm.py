import tensorflow
import numpy
import os
import random
import cv2
import cnn_en_de
import functools

# data_dir='E:/SRAD2018/train'
# data_dir='/media/zhao/新加卷/SRAD2018/train'
data_dir='/home/jxzhao/tianchi/SRAD2018/train'
log_dir='log/'
model_dir='model/'
init_lr=0.001
decay_rate=0.01
batch_file=2
batch_rad=2
batch_size=batch_file*batch_rad
max_step=300000//batch_size+1

def convlstm_encode(c_old,h_old,x):
    with tensorflow.variable_scope('convlstm_encode', reuse=tensorflow.AUTO_REUSE):
        fxw=tensorflow.get_variable('fxw',[3,3,64,64])
        fhw=tensorflow.get_variable('fhw',[3,3,64,64])
        fcw=tensorflow.get_variable('fcw',[batch_size,32,32,64])
        fb=tensorflow.get_variable('fb',64)
        fxw_r=tensorflow.nn.conv2d(x,fxw,[1,1,1,1],'SAME')
        fhw_r=tensorflow.nn.conv2d(h_old,fhw,[1,1,1,1],'SAME')
        fcw_r=c_old*fcw
        f=tensorflow.nn.sigmoid(fxw_r+fhw_r+fcw_r+fb)

        ixw=tensorflow.get_variable('ixw',[3,3,64,64])
        ihw=tensorflow.get_variable('ihw',[3,3,64,64])
        icw=tensorflow.get_variable('icw',[batch_size,32,32,64])
        ib=tensorflow.get_variable('ib',64)
        ixw_r=tensorflow.nn.conv2d(x,ixw,[1,1,1,1],'SAME')
        ihw_r=tensorflow.nn.conv2d(h_old,ihw,[1,1,1,1],'SAME')
        icw_r=c_old*icw
        i=tensorflow.nn.sigmoid(ixw_r+ihw_r+icw_r+ib)

        txw=tensorflow.get_variable('txw',[3,3,64,64])
        thw=tensorflow.get_variable('thw',[3,3,64,64])
        tb=tensorflow.get_variable('tb',64)
        txw_r=tensorflow.nn.conv2d(x,txw,[1,1,1,1],'SAME')
        thw_r=tensorflow.nn.conv2d(h_old,thw,[1,1,1,1],'SAME')
        t=tensorflow.nn.tanh(txw_r+thw_r+tb)

        c_new=f*c_old+i*t

        oxw=tensorflow.get_variable('oxw',[3,3,64,64])
        ohw=tensorflow.get_variable('ohw',[3,3,64,64])
        ocw=tensorflow.get_variable('ocw',[batch_size,32,32,64])
        ob=tensorflow.get_variable('ob',64)
        oxw_r=tensorflow.nn.conv2d(x,oxw,[1,1,1,1],'SAME')
        ohw_r=tensorflow.nn.conv2d(h_old,ohw,[1,1,1,1],'SAME')
        ocw_r=c_old*ocw
        o=tensorflow.nn.sigmoid(oxw_r+ohw_r+ocw_r+ob)

        h_new=o*tensorflow.nn.tanh(c_new)
    
    return c_new,h_new

def convlstm_decode(c_old,h_old):
    with tensorflow.variable_scope('convlstm_decode', reuse=tensorflow.AUTO_REUSE):
        fhw=tensorflow.get_variable('fhw',[3,3,64,64])
        fcw=tensorflow.get_variable('fcw',[batch_size,32,32,64])
        fb=tensorflow.get_variable('fb',64)
        fhw_r=tensorflow.nn.conv2d(h_old,fhw,[1,1,1,1],'SAME')
        fcw_r=c_old*fcw
        f=tensorflow.nn.sigmoid(fhw_r+fcw_r+fb)

        ihw=tensorflow.get_variable('ihw',[3,3,64,64])
        icw=tensorflow.get_variable('icw',[batch_size,32,32,64])
        ib=tensorflow.get_variable('ib',64)
        ihw_r=tensorflow.nn.conv2d(h_old,ihw,[1,1,1,1],'SAME')
        icw_r=c_old*icw
        i=tensorflow.nn.sigmoid(ihw_r+icw_r+ib)

        thw=tensorflow.get_variable('thw',[3,3,64,64])
        tb=tensorflow.get_variable('tb',64)
        thw_r=tensorflow.nn.conv2d(h_old,thw,[1,1,1,1],'SAME')
        t=tensorflow.nn.tanh(thw_r+tb)

        c_new=f*c_old+i*t

        ohw=tensorflow.get_variable('ohw',[3,3,64,64])
        ocw=tensorflow.get_variable('ocw',[batch_size,32,32,64])
        ob=tensorflow.get_variable('ob',64)
        ohw_r=tensorflow.nn.conv2d(h_old,ohw,[1,1,1,1],'SAME')
        ocw_r=c_old*ocw
        o=tensorflow.nn.sigmoid(ohw_r+ocw_r+ob)

        h_new=o*tensorflow.nn.tanh(c_new)
    
    return c_new,h_new

def model(input_encode):
    all_output_encode=[]
    init_cell=numpy.zeros([batch_size,32,32,64]).astype(numpy.float32)
    init_hide=numpy.zeros([batch_size,32,32,64]).astype(numpy.float32)
    for i in range(31):
        if i==0:
            output_encode_cell, output_encode_hide=convlstm_encode(init_cell,init_hide,input_encode[:,i,:,:,:])
            all_output_encode.append(output_encode_hide)
        else:
            output_encode_cell, output_encode_hide=convlstm_encode(output_encode_cell,output_encode_hide,input_encode[:,i,:,:,:])
            all_output_encode.append(output_encode_hide)

    all_output_decode=[]
    for i in range(30):
        if i==0:
            output_decode_cell, output_decode_hide=convlstm_decode(output_encode_cell,output_encode_hide)
            all_output_decode.append(output_decode_hide)
        else:
            output_decode_cell, output_decode_hide=convlstm_decode(output_decode_cell,output_decode_hide)
            all_output_decode.append(output_decode_hide)

    return output_encode_cell, all_output_encode, output_decode_cell, all_output_decode

len([x.name for x in tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES)])

input_encode=tensorflow.placeholder(tensorflow.float32,[batch_size,61,32,32,64])
global_step = tensorflow.get_variable('global_step',initializer=0, trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step,decay_rate)
true_image=tensorflow.placeholder(tensorflow.float32,[None,501,501,1],name='true_image')
predict_image=tensorflow.placeholder(tensorflow.float32,[None,501,501,1],name='predict_image')

output=model(input_encode)

loss=0
for i in range(30):
    loss=loss+tensorflow.losses.mean_squared_error(output[3][i],input_encode[:,31+i,:,:,:])
loss=loss/30

with tensorflow.control_dependencies(tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS)):
    minimize=tensorflow.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step,name='minimize')

Saver = tensorflow.train.Saver(max_to_keep=0,filename='cnn_convlstm')

Session=tensorflow.Session()
if tensorflow.train.latest_checkpoint(model_dir):
    Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))
else:
    Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss', loss)
tensorflow.summary.image('true image',true_image,30)
tensorflow.summary.image('predict image',predict_image,30)
merge_all = tensorflow.summary.merge_all()
FileWriter = tensorflow.summary.FileWriter(log_dir, Session.graph)

for _ in range(max_step):
    all_file=os.listdir(data_dir)
    pick_files=random.sample(all_file,batch_file)
    files=[os.path.join(data_dir,x) for x in pick_files]
    all_rad=[os.listdir(x) for x in files]
    pick_rads=[random.sample(x,batch_rad) for x in all_rad]
    rads=[[os.path.join(files[x],y) for y in pick_rads[x]] for x in range(len(files))]
    all_image_dir=[]
    for x in rads:
        for y in x:
            image_dir=[os.path.join(y,z) for z in os.listdir(y)]
            image_dir.sort()
            all_image_dir.append(image_dir)
    all_image=[]
    for x in all_image_dir:
        k1=[]
        for y in x:
            k1.append(cv2.imread(y))
        all_image.append(k1)
    all_image=numpy.array(all_image)
    encode_image=cnn_en_de.get_encode(all_image).reshape((batch_size,61,32,32,64))

    try:
        Session.run(minimize,feed_dict={input_encode:encode_image})
        if Session.run(global_step)%100==1:
            pre_code=numpy.array(Session.run(output[3],feed_dict={input_encode:encode_image}))[:,0,:]
            pre_image=cnn_en_de.get_decode(pre_code)
            summary = Session.run(merge_all, feed_dict={input_encode:encode_image,true_image:all_image[0,31:,:,:,0:1],predict_image:pre_image})
            FileWriter.add_summary(summary, Session.run(global_step))
            Saver.save(Session, model_dir, global_step)
            print(Session.run(loss,feed_dict={input_encode:encode_image}))
    except:
        with open('log/异常数据目录.txt','a') as f:
            f.write('异常数据:%s\n'%(rads))

    print(Session.run(global_step))