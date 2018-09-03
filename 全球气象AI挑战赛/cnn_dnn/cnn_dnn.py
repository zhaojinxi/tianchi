import tensorflow
import numpy 
import os
import cv2
import random
import time

# data_dir='E:/SRAD2018/train'
data_dir='/media/zhao/新加卷/SRAD2018/train'
# data_dir='/home/jxzhao/tianchi/SRAD2018/train'
log_dir='log/'
model_dir='model/'
init_lr=0.001
decay_rate=0.01
max_step=300001
input_channel=1
encode_channel1=8
encode_channel2=16
encode_channel3=32
encode_channel4=64
encode_channel5=128
output_channel=1

def encode(x,is_train):
    with tensorflow.variable_scope('encode'):
        encode_w1=tensorflow.get_variable('w1', [3,3,input_channel,encode_channel1], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b1=tensorflow.get_variable('b1', encode_channel1, initializer=tensorflow.constant_initializer(0))
        encode_z1=tensorflow.nn.conv2d((x-128)/128,encode_w1,[1,2,2,1],'SAME')+encode_b1
        encode_z1=tensorflow.layers.batch_normalization(encode_z1,training=is_train,name='bn1')
        encode_z1=tensorflow.nn.selu(encode_z1)

        encode_w2=tensorflow.get_variable('w2', [3,3,encode_channel1,encode_channel2], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b2=tensorflow.get_variable('b2', encode_channel2, initializer=tensorflow.constant_initializer(0))
        encode_z2=tensorflow.nn.conv2d(encode_z1,encode_w2,[1,2,2,1],'SAME')+encode_b2
        encode_z2=tensorflow.layers.batch_normalization(encode_z2,training=is_train,name='bn2')
        encode_z2=tensorflow.nn.selu(encode_z2)

        encode_w3=tensorflow.get_variable('w3', [3,3,encode_channel2,encode_channel3], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b3=tensorflow.get_variable('b3', encode_channel3, initializer=tensorflow.constant_initializer(0))
        encode_z3=tensorflow.nn.conv2d(encode_z2,encode_w3,[1,2,2,1],'SAME')+encode_b3
        encode_z3=tensorflow.layers.batch_normalization(encode_z3,training=is_train,name='bn3')
        encode_z3=tensorflow.nn.selu(encode_z3)

        encode_w4=tensorflow.get_variable('w4', [3,3,encode_channel3,encode_channel4], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b4=tensorflow.get_variable('b4', encode_channel4, initializer=tensorflow.constant_initializer(0))
        encode_z4=tensorflow.nn.conv2d(encode_z3,encode_w4,[1,2,2,1],'SAME')+encode_b4
        encode_z4=tensorflow.layers.batch_normalization(encode_z4,training=is_train,name='bn4')
        encode_z4=tensorflow.nn.selu(encode_z4)

        encode_w5=tensorflow.get_variable('w4', [3,3,encode_channel4,encode_channel5], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b5=tensorflow.get_variable('b4', encode_channel5, initializer=tensorflow.constant_initializer(0))
        encode_z5=tensorflow.nn.conv2d(encode_z4,encode_w5,[1,2,2,1],'SAME')+encode_b5
        encode_z5=tensorflow.layers.batch_normalization(encode_z5,training=is_train,name='bn5')
        encode_z5=tensorflow.nn.selu(encode_z5)

        encode_w4=tensorflow.get_variable('w4', [3,3,encode_channel5,encode_channel4], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b4=tensorflow.get_variable('b4', encode_channel4, initializer=tensorflow.constant_initializer(0))
        encode_z4=tensorflow.nn.conv2d(encode_z3,encode_w4,[1,2,2,1],'SAME')+encode_b4
        encode_z4=tensorflow.layers.batch_normalization(encode_z4,training=is_train,name='bn4')
        encode_z4=tensorflow.nn.selu(encode_z4, name='encode_image')

    return encode_z4

def decode(x,is_train):
    with tensorflow.variable_scope('decode',reuse=tensorflow.AUTO_REUSE):
        decode_w1=tensorflow.get_variable('w1', [32*32*64,251001], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b1=tensorflow.get_variable('b1', 251001, initializer=tensorflow.constant_initializer(0))
        decode_z1=tensorflow.matmul(tensorflow.reshape(x,[-1,251001]),decode_w1)+decode_b1
        # decode_z1=tensorflow.layers.batch_normalization(decode_z1,training=is_train,name='bn1')
        decode_z1=tensorflow.nn.tanh(decode_z1)
        decode_z1=tensorflow.reshape(tensorflow.clip_by_value(decode_z1*128+128,0,255),[-1,501,501,1],name='decode_image')

    return decode_z4

input_image=tensorflow.placeholder(tensorflow.float32,[None,501,501,1],name='input_image')
is_train=tensorflow.placeholder(tensorflow.bool,name='is_train')
global_step = tensorflow.get_variable('global_step',initializer=0, trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step*61,decay_rate)

encode_z4=encode(input_image,is_train)
decode_z4=decode(encode_z4,is_train)

loss=tensorflow.losses.mean_squared_error(input_image,decode_z4)

with tensorflow.control_dependencies(tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS)):
    minimize=tensorflow.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step,name='minimize')

Saver = tensorflow.train.Saver(max_to_keep=0,filename='cnn')

Session=tensorflow.Session()
Saver.restore(Session,'model/-4087061')
# if tensorflow.train.latest_checkpoint(model_dir):
#     Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))
# else:
#     Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss', loss)
tensorflow.summary.image('input_images', input_image, 61)
tensorflow.summary.image('output_images', decode_z4, 61)
merge_all = tensorflow.summary.merge_all()
FileWriter = tensorflow.summary.FileWriter(log_dir, Session.graph)

for _ in range(max_step):
    all_file=os.listdir(data_dir)
    pick_one_file=random.sample(all_file,1)[0]
    one_file=os.path.join(data_dir,pick_one_file)
    one_all_rad=os.listdir(one_file)
    pick_one_rad=random.sample(one_all_rad,1)[0]
    one_rad=os.path.join(one_file,pick_one_rad)
    all_image_dir=[os.path.join(one_rad,x) for x in os.listdir(one_rad)]
    all_image_dir.sort()
    all_image=[cv2.imread(x) for x in all_image_dir]
    all_image=numpy.array(all_image)

    try:
        for j in range(all_image.shape[0]):
            Session.run(minimize,feed_dict={input_image:all_image[j:j+1,:,:,0:1],is_train:True})
        if Session.run(global_step)%61000==61:
            summary = Session.run(merge_all, feed_dict={input_image:all_image[:,:,:,0:1],is_train:False})
            FileWriter.add_summary(summary, Session.run(global_step))
            Saver.save(Session, model_dir, global_step)
            print(Session.run(loss,feed_dict={input_image:all_image[:,:,:,0:1],is_train:False}))
    except:
        with open('log/异常数据目录.txt','a') as f:
            f.write('异常数据:%s\n'%(one_rad))

    print(Session.run(global_step))

Session.close()