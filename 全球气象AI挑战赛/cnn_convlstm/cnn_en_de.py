import tensorflow
import numpy 
import os
import cv2
import random
import time

input_channel=1
encode_channel1=8
encode_channel2=16
encode_channel3=32
encode_channel4=64
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
        encode_z4=tensorflow.nn.selu(encode_z4, name='encode_image')

    return encode_z4

def decode(x,is_train):
    with tensorflow.variable_scope('decode',reuse=tensorflow.AUTO_REUSE):
        decode_w1=tensorflow.get_variable('w1', [3,3,encode_channel4,encode_channel3], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b1=tensorflow.get_variable('b1', encode_channel3, initializer=tensorflow.constant_initializer(0))
        decode_z1=tensorflow.nn.conv2d(tensorflow.image.resize_nearest_neighbor(x,[126,126]),decode_w1,[1,2,2,1],'SAME')+decode_b1
        decode_z1=tensorflow.layers.batch_normalization(decode_z1,training=is_train,name='bn1')
        decode_z1=tensorflow.nn.selu(decode_z1)

        decode_w2=tensorflow.get_variable('w2', [3,3,encode_channel3,encode_channel2], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b2=tensorflow.get_variable('b2', encode_channel2, initializer=tensorflow.constant_initializer(0))
        decode_z2=tensorflow.nn.conv2d(tensorflow.image.resize_nearest_neighbor(decode_z1,[251,251]),decode_w2,[1,2,2,1],'SAME')+decode_b2
        decode_z2=tensorflow.layers.batch_normalization(decode_z2,training=is_train,name='bn2')
        decode_z2=tensorflow.nn.selu(decode_z2)

        decode_w3=tensorflow.get_variable('w3', [3,3,encode_channel2,encode_channel1], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b3=tensorflow.get_variable('b3', encode_channel1, initializer=tensorflow.constant_initializer(0))
        decode_z3=tensorflow.nn.conv2d(tensorflow.image.resize_nearest_neighbor(decode_z2,[501,501]),decode_w3,[1,2,2,1],'SAME')+decode_b3
        decode_z3=tensorflow.layers.batch_normalization(decode_z3,training=is_train,name='bn3')
        decode_z3=tensorflow.nn.selu(decode_z3)

        decode_w4=tensorflow.get_variable('w4', [3,3,encode_channel1,input_channel], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b4=tensorflow.get_variable('b4', input_channel, initializer=tensorflow.constant_initializer(0))
        decode_z4=tensorflow.nn.conv2d(tensorflow.image.resize_nearest_neighbor(decode_z3,[1001,1001]),decode_w4,[1,2,2,1],'SAME')+decode_b4
        decode_z4=tensorflow.layers.batch_normalization(decode_z4,training=is_train,name='bn4')
        decode_z4=tensorflow.nn.tanh(decode_z4)
        decode_z4=tensorflow.clip_by_value(decode_z4*128+128,0,255,name='decode_image')

    return decode_z4

def get_encode(all_image):
    Graph_encode=tensorflow.Graph()
    Session_encode = tensorflow.Session(graph=Graph_encode)
    with Graph_encode.as_default():
        input_image=tensorflow.placeholder(tensorflow.float32,[None,501,501,1],name='input_image')
        is_train=tensorflow.placeholder(tensorflow.bool,name='is_train')

        encode_z4=encode(input_image,is_train)

        Saver = tensorflow.train.Saver()
        Saver.restore(Session_encode,tensorflow.train.latest_checkpoint('../cnn/model/'))

        result=[]
        for x in all_image:
            result.append(Session_encode.run(encode_z4, feed_dict={input_image:x[:,:,:,0:1], is_train:False}))

    return numpy.array(result)

def get_decode(code):
    Graph_decode=tensorflow.Graph()
    Session_decode = tensorflow.Session(graph=Graph_decode)
    with Graph_decode.as_default():
        input_code=tensorflow.placeholder(tensorflow.float32,[None,32,32,64],name='input_code')
        is_train=tensorflow.placeholder(tensorflow.bool,name='is_train')

        decode_z4=decode(input_code,is_train)

        Saver = tensorflow.train.Saver()
        Saver.restore(Session_decode,tensorflow.train.latest_checkpoint('../cnn/model/'))

        result=Session_decode.run(decode_z4, feed_dict={input_code:code, is_train:False})

    return numpy.array(result)