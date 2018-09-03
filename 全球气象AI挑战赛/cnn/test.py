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
output_channel=1

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
        encode_z4=tensorflow.nn.tanh(encode_z4, name='encode_image')

    return encode_z4

def decode(x,is_train):
    with tensorflow.variable_scope('decode',reuse=tensorflow.AUTO_REUSE):
        decode_w1=tensorflow.get_variable('w1', [3,3,encode_channel3,encode_channel4], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b1=tensorflow.get_variable('b1', encode_channel3, initializer=tensorflow.constant_initializer(0))
        decode_z1=tensorflow.nn.conv2d_transpose(x,decode_w1,tensorflow.convert_to_tensor([tensorflow.shape(x)[0],63,63,32]),[1,2,2,1],'SAME')+decode_b1
        decode_z1=tensorflow.layers.batch_normalization(decode_z1,training=is_train,name='bn1')
        decode_z1=tensorflow.nn.selu(decode_z1)

        decode_w2=tensorflow.get_variable('w2', [3,3,encode_channel2,encode_channel3], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b2=tensorflow.get_variable('b2', encode_channel2, initializer=tensorflow.constant_initializer(0))
        decode_z2=tensorflow.nn.conv2d_transpose(decode_z1,decode_w2,tensorflow.convert_to_tensor([tensorflow.shape(x)[0],126,126,16]),[1,2,2,1],'SAME')+decode_b2
        decode_z2=tensorflow.layers.batch_normalization(decode_z2,training=is_train,name='bn2')
        decode_z2=tensorflow.nn.selu(decode_z2)

        decode_w3=tensorflow.get_variable('w3', [3,3,encode_channel1,encode_channel2], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b3=tensorflow.get_variable('b3', encode_channel1, initializer=tensorflow.constant_initializer(0))
        decode_z3=tensorflow.nn.conv2d_transpose(decode_z2,decode_w3,tensorflow.convert_to_tensor([tensorflow.shape(x)[0],251,251,8]),[1,2,2,1],'SAME')+decode_b3
        decode_z3=tensorflow.layers.batch_normalization(decode_z3,training=is_train,name='bn3')
        decode_z3=tensorflow.nn.selu(decode_z3)

        decode_w4=tensorflow.get_variable('w4', [3,3,input_channel,encode_channel1], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b4=tensorflow.get_variable('b4', input_channel, initializer=tensorflow.constant_initializer(0))
        decode_z4=tensorflow.nn.conv2d_transpose(decode_z3,decode_w4,tensorflow.convert_to_tensor([tensorflow.shape(x)[0],501,501,1]),[1,2,2,1],'SAME')+decode_b4
        decode_z4=tensorflow.layers.batch_normalization(decode_z4,training=is_train,name='bn4')
        decode_z4=tensorflow.nn.tanh(decode_z4)
        decode_z4=tensorflow.clip_by_value(decode_z4*128+128,0,255,name='decode_image')

    return decode_z4

input_image=tensorflow.placeholder(tensorflow.float32,[None,501,501,1],name='input_image')
input_code=tensorflow.placeholder(tensorflow.float32,[None,32,32,64],name='input_code')
is_train=tensorflow.placeholder(tensorflow.bool,name='is_train')

encode_z4=encode(input_image,is_train)
decode_z4=decode(input_code,is_train)

Session=tensorflow.Session()

Saver = tensorflow.train.Saver()
Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))

result_en=Session.run(encode_z4, feed_dict={input_image:all_image[:,:,:,0:1], is_train:False})
result_de=Session.run(decode_z4, feed_dict={input_code:result_en, is_train:False})

for image in all_image[:,:,:,0:1]:
    cv2.imshow('true image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.1)
cv2.destroyAllWindows()

for image in result_de:
    cv2.imshow('decode image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.1)
cv2.destroyAllWindows()

Session.close()