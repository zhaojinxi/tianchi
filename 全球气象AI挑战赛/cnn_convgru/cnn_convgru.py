import tensorflow
import numpy
import os
import random
import time
import cv2

# data_dir='E:/SRAD2018/train'
# data_dir='/media/zhao/新加卷/SRAD2018/train'
data_dir='/home/jxzhao/tianchi/SRAD2018/train'
log_dir='log/'
model_dir='model/'
init_lr=0.001
decay_rate=0.01
batch_file=1
batch_rad=1
batch_size=batch_file*batch_rad
max_step=300000//batch_size+1
input_channel=1
encode_channel1=4
encode_channel2=8
encode_channel3=16
encode_channel4=32

def cnn_encode(x):
    with tensorflow.variable_scope('cnn_encode', reuse=tensorflow.AUTO_REUSE):
        encode_w1=tensorflow.get_variable('w1', [3,3,input_channel,encode_channel1], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b1=tensorflow.get_variable('b1', encode_channel1, initializer=tensorflow.constant_initializer(0))
        encode_z1=tensorflow.nn.conv2d((x-128)/128,encode_w1,[1,2,2,1],'SAME')+encode_b1
        encode_z1=tensorflow.contrib.layers.layer_norm(encode_z1,scope='ln1')
        encode_z1=tensorflow.nn.selu(encode_z1)

        encode_w2=tensorflow.get_variable('w2', [3,3,encode_channel1,encode_channel2], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b2=tensorflow.get_variable('b2', encode_channel2, initializer=tensorflow.constant_initializer(0))
        encode_z2=tensorflow.nn.conv2d(encode_z1,encode_w2,[1,2,2,1],'SAME')+encode_b2
        encode_z2=tensorflow.contrib.layers.layer_norm(encode_z2,scope='ln2')
        encode_z2=tensorflow.nn.selu(encode_z2)

        encode_w3=tensorflow.get_variable('w3', [3,3,encode_channel2,encode_channel3], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b3=tensorflow.get_variable('b3', encode_channel3, initializer=tensorflow.constant_initializer(0))
        encode_z3=tensorflow.nn.conv2d(encode_z2,encode_w3,[1,2,2,1],'SAME')+encode_b3
        encode_z3=tensorflow.contrib.layers.layer_norm(encode_z3,scope='ln3')
        encode_z3=tensorflow.nn.selu(encode_z3)

        encode_w4=tensorflow.get_variable('w4', [3,3,encode_channel3,encode_channel4], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b4=tensorflow.get_variable('b4', encode_channel4, initializer=tensorflow.constant_initializer(0))
        encode_z4=tensorflow.nn.conv2d(encode_z3,encode_w4,[1,2,2,1],'SAME')+encode_b4
        encode_z4=tensorflow.contrib.layers.layer_norm(encode_z4,scope='ln4')
        encode_z4=tensorflow.nn.tanh(encode_z4, name='encode_image')

    return encode_z4

def cnn_decode(x):
    with tensorflow.variable_scope('cnn_decode',reuse=tensorflow.AUTO_REUSE):
        decode_w1=tensorflow.get_variable('w1', [3,3,encode_channel4,encode_channel3], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b1=tensorflow.get_variable('b1', encode_channel3, initializer=tensorflow.constant_initializer(0))
        decode_z1=tensorflow.nn.conv2d(tensorflow.image.resize_nearest_neighbor(x,[63,63]),decode_w1,[1,1,1,1],'SAME')+decode_b1
        decode_z1=tensorflow.contrib.layers.layer_norm(decode_z1,scope='ln1')
        decode_z1=tensorflow.nn.selu(decode_z1)

        decode_w2=tensorflow.get_variable('w2', [3,3,encode_channel3,encode_channel2], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b2=tensorflow.get_variable('b2', encode_channel2, initializer=tensorflow.constant_initializer(0))
        decode_z2=tensorflow.nn.conv2d(tensorflow.image.resize_nearest_neighbor(decode_z1,[126,126]),decode_w2,[1,1,1,1],'SAME')+decode_b2
        decode_z2=tensorflow.contrib.layers.layer_norm(decode_z2,scope='ln2')
        decode_z2=tensorflow.nn.selu(decode_z2)

        decode_w3=tensorflow.get_variable('w3', [3,3,encode_channel2,encode_channel1], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b3=tensorflow.get_variable('b3', encode_channel1, initializer=tensorflow.constant_initializer(0))
        decode_z3=tensorflow.nn.conv2d(tensorflow.image.resize_nearest_neighbor(decode_z2,[251,251]),decode_w3,[1,1,1,1],'SAME')+decode_b3
        decode_z3=tensorflow.contrib.layers.layer_norm(decode_z3,scope='ln3')
        decode_z3=tensorflow.nn.selu(decode_z3)

        decode_w4=tensorflow.get_variable('w4', [3,3,encode_channel1,input_channel], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b4=tensorflow.get_variable('b4', input_channel, initializer=tensorflow.constant_initializer(0))
        decode_z4=tensorflow.nn.conv2d(tensorflow.image.resize_nearest_neighbor(decode_z3,[501,501]),decode_w4,[1,1,1,1],'SAME')+decode_b4
        decode_z4=tensorflow.contrib.layers.layer_norm(decode_z4,scope='ln4')
        decode_z4=tensorflow.nn.tanh(decode_z4)
        decode_z4=tensorflow.clip_by_value(decode_z4*128+128,0,255,name='decode_image')

    return decode_z4

def convgru_encode(h_old,x):
    with tensorflow.variable_scope('convgru_encode', reuse=tensorflow.AUTO_REUSE):
        rxw=tensorflow.get_variable('rxw',[3,3,32,32])
        rhw=tensorflow.get_variable('rhw',[3,3,32,32])
        rb=tensorflow.get_variable('rb',32)
        rxw_r=tensorflow.nn.conv2d(x,rxw,[1,1,1,1],'SAME')
        rhw_r=tensorflow.nn.conv2d(h_old,rhw,[1,1,1,1],'SAME')
        r=tensorflow.nn.sigmoid(rxw_r+rhw_r+rb)

        uxw=tensorflow.get_variable('uxw',[3,3,32,32])
        uhw=tensorflow.get_variable('uhw',[3,3,32,32])
        ub=tensorflow.get_variable('ub',32)
        uxw_r=tensorflow.nn.conv2d(x,uxw,[1,1,1,1],'SAME')
        uhw_r=tensorflow.nn.conv2d(h_old,uhw,[1,1,1,1],'SAME')
        u=tensorflow.nn.sigmoid(uxw_r+uhw_r+ub)

        txw=tensorflow.get_variable('txw',[3,3,32,32])
        thw=tensorflow.get_variable('thw',[3,3,32,32])
        tb=tensorflow.get_variable('tb',32)
        txw_r=tensorflow.nn.conv2d(x,txw,[1,1,1,1],'SAME')
        thw_r=tensorflow.nn.conv2d(r*h_old,thw,[1,1,1,1],'SAME')
        t=tensorflow.nn.tanh(txw_r+thw_r+tb)

        h_new=(1-u)*h_old+u*t
        return h_new

def convgru_decode(h_old):
    with tensorflow.variable_scope('convgru_decode', reuse=tensorflow.AUTO_REUSE):
        rhw=tensorflow.get_variable('rhw',[3,3,32,32])
        rb=tensorflow.get_variable('rb',32)
        rhw_r=tensorflow.nn.conv2d(h_old,rhw,[1,1,1,1],'SAME')
        r=tensorflow.nn.sigmoid(rhw_r+rb)

        uhw=tensorflow.get_variable('uhw',[3,3,32,32])
        ub=tensorflow.get_variable('ub',32)
        uhw_r=tensorflow.nn.conv2d(h_old,uhw,[1,1,1,1],'SAME')
        u=tensorflow.nn.sigmoid(uhw_r+ub)

        thw=tensorflow.get_variable('thw',[3,3,32,32])
        tb=tensorflow.get_variable('tb',32)
        thw_r=tensorflow.nn.conv2d(r*h_old,thw,[1,1,1,1],'SAME')
        t=tensorflow.nn.tanh(thw_r+tb)

        h_new=(1-u)*h_old+u*t
        return h_new

def gru_process(input_code):
    all_output_encode=[]
    init_hide=numpy.zeros([batch_size,32,32,32]).astype(numpy.float32)
    for i in range(31):
        if i==0:
            output_hide=convgru_encode(init_hide,input_code[:,i,:,:,:])
            all_output_encode.append(output_hide)
        else:
            output_hide=convgru_encode(output_hide,input_code[:,i,:,:,:])
            all_output_encode.append(tensorflow.reshape(output_hide,[batch_size,1,32,32,32]))

    all_output_decode=[]
    for i in range(30):
        output_hide=convgru_decode(output_hide)
        all_output_decode.append(output_hide)

    return all_output_encode, all_output_decode

len([x.name for x in tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES)])

train_image=tensorflow.placeholder(tensorflow.float32,[batch_size,31,501,501,1],name='train_image')
answer_image=tensorflow.placeholder(tensorflow.float32,[batch_size,30,501,501,1],name='answer_image')
global_step = tensorflow.get_variable('global_step',initializer=0,trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step*30,decay_rate)
which_opt = tensorflow.get_variable('which_opt',initializer=0,trainable=False)

cnn_encode_result=tensorflow.map_fn(cnn_encode,train_image,name='cnn_encode_result')
gru_result=gru_process(cnn_encode_result)
pre_result=tensorflow.stack(gru_result[1],1)
cnn_decode_result=tensorflow.map_fn(cnn_decode,pre_result,name='cnn_decode_result')

loss=tensorflow.losses.mean_squared_error(answer_image[:,which_opt,:,:,:],cnn_decode_result[:,which_opt,:,:,:])

minimize=tensorflow.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step,name='minimize')

Saver = tensorflow.train.Saver(max_to_keep=0,filename='cnn_convgru')

Session=tensorflow.Session()
if tensorflow.train.latest_checkpoint(model_dir):
    Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))
else:
    Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss', loss)
tensorflow.summary.image('answer_images', answer_image[0,:,:,:,:], 10)
tensorflow.summary.image('output_images', cnn_decode_result[0], 10)
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
    try:
        for j in range(30):
            Session.run(minimize,feed_dict={train_image:all_image[:,:31,:,:,0:1],answer_image:all_image[:,31:,:,:,0:1],which_opt:j})
        if Session.run(global_step)%3000==30:
            summary = Session.run(merge_all, feed_dict={train_image:all_image[:,:31,:,:,0:1],answer_image:all_image[:,31:,:,:,0:1],which_opt:10})
            FileWriter.add_summary(summary, Session.run(global_step))
            Saver.save(Session, model_dir, global_step)
            print(Session.run(loss,feed_dict={train_image:all_image[:,:31,:,:,0:1],answer_image:all_image[:,31:,:,:,0:1]}))
    except:
        with open('log/异常数据目录.txt','a') as f:
            f.write('异常数据:%s\n'%(rads))

    print(Session.run(global_step))