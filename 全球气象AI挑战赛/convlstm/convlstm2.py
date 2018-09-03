import tensorflow
import numpy
import os
import random
import time
import skimage.io

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
channel_dim1=4

train_image=tensorflow.placeholder(tensorflow.float32,[batch_size,31,501,501,1],name='train_image')
answer_image=tensorflow.placeholder(tensorflow.float32,[batch_size,30,501,501,1],name='answer_image')
global_step = tensorflow.get_variable('global_step',initializer=0,trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step,decay_rate)
which_loss = tensorflow.get_variable('which_loss',initializer=0,trainable=False)

encoder_cell = tensorflow.contrib.rnn.ConvLSTMCell(conv_ndims=2,input_shape=[501,501,1], output_channels=channel_dim1, kernel_shape=[3,3], use_bias=True, forget_bias=True, initializers=tensorflow.truncated_normal_initializer(stddev=0.1), name='encode_convlstm1')
encoder_outputs, encoder_final_state = tensorflow.nn.dynamic_rnn(encoder_cell, train_image, dtype=tensorflow.float32, scope='encode')
decoder_cell = tensorflow.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[501,501,1], output_channels=channel_dim1, kernel_shape=[3,3], use_bias=True, forget_bias=True, initializers=tensorflow.truncated_normal_initializer(stddev=0.1), name='decode_convlstm1')
decoder_outputs, decoder_final_state = tensorflow.nn.dynamic_rnn(decoder_cell, encoder_outputs, initial_state=encoder_final_state, scope='decode')

def predict(x):
    with tensorflow.variable_scope('predict',reuse=tensorflow.AUTO_REUSE):
        w1=tensorflow.get_variable('w1', [3,3,channel_dim1,1], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b1=tensorflow.get_variable('b1', 1, initializer=tensorflow.constant_initializer(0))
        z1=tensorflow.nn.tanh(tensorflow.nn.conv2d(x,w1,[1,1,1,1],'SAME')+b1)
        z1=tensorflow.clip_by_value(z1*128+128,0,255)
    return z1

decode_image=tensorflow.map_fn(predict,decoder_outputs,name='decode_image')

loss=tensorflow.losses.mean_squared_error(answer_image,decode_image)

minimize=tensorflow.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step,name='minimize')

Saver = tensorflow.train.Saver(max_to_keep=0,filename='convlstm')

Session=tensorflow.Session()
if tensorflow.train.latest_checkpoint(model_dir):
    Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))
else:
    Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss', loss)
tensorflow.summary.image('answer_images', answer_image[0,:,:,:,:], 10)
tensorflow.summary.image('output_images', decode_image[0], 10)
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
            k1.append(skimage.io.imread(y))
        all_image.append(k1)
    all_image=numpy.array(all_image).reshape(batch_size,61,501,501,1)
    try:
        Session.run(minimize,feed_dict={train_image:all_image[:,:31,:,:,:],answer_image:all_image[:,31:,:,:,:]})
        if Session.run(global_step)%100==1:
            summary = Session.run(merge_all, feed_dict={train_image:all_image[:,:31,:,:,:], answer_image:all_image[:,31:,:,:,:]}) 
            FileWriter.add_summary(summary, Session.run(global_step))
            Saver.save(Session, model_dir, global_step)
            print(Session.run(loss,feed_dict={train_image:all_image[:,:31,:,:,:],answer_image:all_image[:,31:,:,:,:]}))
    except:
        with open('log/异常数据目录.txt','a') as f:
            f.write('异常数据:%s\n'%(rads))

    print(Session.run(global_step))