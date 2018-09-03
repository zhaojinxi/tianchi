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
batch_file=2
batch_rad=2
code_dim=32*32*64
lstm_dim=128
max_step=300000//batch_file//batch_rad+1

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

    return result

def lstm(x,is_train):
    x = tensorflow.unstack(x, 31, 1)
    lstm_cell =tensorflow.nn.rnn_cell.BasicLSTMCell(lstm_dim)
    outputs, states =tensorflow.nn.static_rnn(lstm_cell, x,dtype=tensorflow.float32)

    w1=tensorflow.get_variable('w1', [lstm_dim,code_dim], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    b1=tensorflow.get_variable('b1', code_dim, initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    z1=tensorflow.matmul(outputs[-1], w1) + b1
    z1 = tensorflow.nn.softmax(z1)
    z1=tensorflow.reshape(z1,[-1,1,code_dim],'code')
    return z1

input_code = tensorflow.placeholder(tensorflow.float32,[None,31,code_dim],name='input_code')
future_code=tensorflow.placeholder(tensorflow.float32,[None,1,code_dim],name='future_code')
is_train=tensorflow.placeholder(tensorflow.bool,name='is_train')
global_step = tensorflow.get_variable('global_step',initializer=0, trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step,decay_rate)
true_image=tensorflow.placeholder(tensorflow.float32,[None,501,501,1],name='true_image')
predict_image=tensorflow.placeholder(tensorflow.float32,[None,501,501,1],name='predict_image')

output_code = lstm(input_code,is_train)

loss=tensorflow.losses.mean_squared_error(output_code,future_code)

with tensorflow.control_dependencies(tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS)):
    minimize=tensorflow.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step,name='minimize')

Saver = tensorflow.train.Saver(max_to_keep=0,filename='lstm')

Session=tensorflow.Session()
if tensorflow.train.latest_checkpoint(model_dir):
    Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))
else:
    Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss', loss)
tensorflow.summary.image('true image',true_image,batch_file*batch_rad)
tensorflow.summary.image('predict image',predict_image,true_image,batch_file*batch_rad)
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
    encode_image=get_encode(all_image).reshape((batch_file*batch_rad,61,code_dim))

    try:
        Session.run(minimize,feed_dict={input_code:encode_image[:,:31,:],future_code:encode_image[:,35:36,:],is_train:True})
        if Session.run(global_step)%100==1:
            pre_image=get_decode(Session.run(output_code,feed_dict={input_code:encode_image[:,:31,:],is_train:False}).reshape([-1,32,32,64]))
            summary = Session.run(merge_all, feed_dict={input_code:encode_image[:,:31,:],future_code:encode_image[:,35:36,:],is_train:False,true_image:all_image[:,35,:,:,0:1],predict_image:pre_image})
            FileWriter.add_summary(summary, Session.run(global_step))
            Saver.save(Session, model_dir, global_step)
            print(Session.run(loss,feed_dict={input_code:encode_image[:,:31,:],future_code:encode_image[:,35:36,:],is_train:False}))
    except:
        with open('log/异常数据目录.txt','a') as f:
            f.write('异常数据:%s\n'%(rads))

    print(Session.run(global_step))