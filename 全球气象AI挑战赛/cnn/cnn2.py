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

def encode(x,is_train):
    with tensorflow.variable_scope('encode',reuse=tensorflow.AUTO_REUSE):
        encode_w1=tensorflow.get_variable('w1', [3,3,input_channel,encode_channel1], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b1=tensorflow.get_variable('b1', encode_channel1, initializer=tensorflow.constant_initializer(0))
        encode_z1=tensorflow.nn.conv2d(x,encode_w1,[1,2,2,1],'SAME')+encode_b1
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

    return encode_z1, encode_z2, encode_z3, encode_z4

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
        decode_z4=tensorflow.add(tensorflow.nn.conv2d_transpose(decode_z3,decode_w4,tensorflow.convert_to_tensor([tensorflow.shape(x)[0],501,501,1]),[1,2,2,1],'SAME'),decode_b4,name='decode_image')
        decode_z4=tensorflow.layers.batch_normalization(decode_z4,training=is_train,name='bn4')
        decode_z4=tensorflow.nn.tanh(decode_z4)
        decode_z4=tensorflow.add(tensorflow.multiply(decode_z4,128),128,name='decode_image')

    return decode_z1, decode_z2, decode_z3, decode_z4

input_image=tensorflow.placeholder(tensorflow.float32,[None,501,501,1],name='input_image')
input_code=tensorflow.placeholder(tensorflow.float32,[None,32,32,64],name='input_code')
is_entrain=tensorflow.placeholder(tensorflow.bool,name='is_entrain')
is_detrain=tensorflow.placeholder(tensorflow.bool,name='is_detrain')

encode_z1, encode_z2, encode_z3, encode_z4=encode(input_image,is_train)
decode_z1, decode_z2, decode_z3, decode_z4=decode(input_code,is_train)

loss1=tensorflow.losses.mean_squared_error(input_image,decode_z4)
loss2=tensorflow.losses.mean_squared_error(encode_z1,decode_z3)
loss3=tensorflow.losses.mean_squared_error(encode_z2,decode_z2)
loss4=tensorflow.losses.mean_squared_error(encode_z3,decode_z1)
loss1_var=tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES)
loss2_var=tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES)
loss2_var.pop(0)
loss2_var.pop(0)
loss2_var.pop(-1)
loss2_var.pop(-1)
loss3_var=tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES)
loss3_var.pop(0)
loss3_var.pop(0)
loss3_var.pop(0)
loss3_var.pop(0)
loss3_var.pop(-1)
loss3_var.pop(-1)
loss3_var.pop(-1)
loss3_var.pop(-1)
loss4_var=tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES)
loss4_var.pop(0)
loss4_var.pop(0)
loss4_var.pop(0)
loss4_var.pop(0)
loss4_var.pop(0)
loss4_var.pop(0)
loss4_var.pop(-1)
loss4_var.pop(-1)
loss4_var.pop(-1)
loss4_var.pop(-1)
loss4_var.pop(-1)
loss4_var.pop(-1)

with tensorflow.control_dependencies(tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS)):
    AdamOptimizer=tensorflow.train.AdamOptimizer(init_lr)

    minimize1=AdamOptimizer.minimize(loss1,var_list=loss1_var,name='minimize1')
    minimize2=AdamOptimizer.minimize(loss2,var_list=loss2_var,name='minimize2')
    minimize3=AdamOptimizer.minimize(loss3,var_list=loss3_var,name='minimize3')
    minimize4=AdamOptimizer.minimize(loss4,var_list=loss4_var,name='minimize4')

Saver = tensorflow.train.Saver(max_to_keep=0,keep_checkpoint_every_n_hours=0.5)

Session=tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss1', loss1)
tensorflow.summary.scalar('loss2', loss2)
tensorflow.summary.scalar('loss3', loss3)
tensorflow.summary.scalar('loss4', loss4)
tensorflow.summary.image('input_images', input_image, 61)
tensorflow.summary.image('output_images', decode_train, 61)
merge_all = tensorflow.summary.merge_all()
FileWriter = tensorflow.summary.FileWriter(log_dir, Session.graph)

for i in range(max_step):
    AdamOptimizer._lr=tensorflow.train.exponential_decay(init_lr,i,max_step,0.01)
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
            Session.run(minimize4,feed_dict={input_image:all_image[:,:,:,0:1],is_train:True})
            Session.run(minimize3,feed_dict={input_image:all_image[:,:,:,0:1],is_train:True})
            Session.run(minimize2,feed_dict={input_image:all_image[:,:,:,0:1],is_train:True})
            Session.run(minimize1,feed_dict={input_image:all_image[:,:,:,0:1],is_train:True})

        if i%1000==0:
            # for image in all_image[:,:,:,0:1]:
            #     cv2.imshow('true image', image)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            #     time.sleep(0.1)
            # cv2.destroyAllWindows()

            # for image in Session.run(decode_z4,feed_dict={input_image:all_image[:,:,:,0:1],is_train:False}):
            #     cv2.imshow('decode image', image)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            #     time.sleep(0.1)
            # cv2.destroyAllWindows()

            summary = Session.run(merge_all, feed_dict={input_image:all_image[:,:,:,0:1],is_train:False})
            FileWriter.add_summary(summary, i)
            Saver.save(Session, model_dir, i)

            print(Session.run(loss1,feed_dict={input_image:all_image[:,:,:,0:1],is_train:False}))
            print(Session.run(loss2,feed_dict={input_image:all_image[:,:,:,0:1],is_train:False}))
            print(Session.run(loss3,feed_dict={input_image:all_image[:,:,:,0:1],is_train:False}))
            print(Session.run(loss4,feed_dict={input_image:all_image[:,:,:,0:1],is_train:False}))

        print(i)

    except:
        with open('log/异常数据目录.txt','a') as f:
            f.write('异常数据:%s,第%s张图片\n'%(one_rad,j+1))