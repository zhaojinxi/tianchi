import tensorflow
import numpy
import os
import random
import skimage.io
import pandas

# data_dir='E:/SRAD2018/train'
# data_dir='/media/zhao/新加卷/SRAD2018/train'
data_dir='/home/jxzhao/tianchi/SRAD2018/train'
log_dir='log/'
model_dir='model/'
init_lr=0.1
decay_rate=0.1
max_step=300001
input_dim=[501,1]
layer_dim1=[251,4]
layer_dim2=[126,8]
layer_dim3=[63,16]

def convgru(x,h_old,name,in_dim,out_dim):
    with tensorflow.variable_scope('convgru%s'%name,reuse=tensorflow.AUTO_REUSE):
        rxw=tensorflow.get_variable('rxw',[3,3,in_dim[1],out_dim[1]])
        rhw=tensorflow.get_variable('rhw',[3,3,out_dim[1],out_dim[1]])
        rb=tensorflow.get_variable('rb',out_dim[1])
        rxw_r=tensorflow.nn.conv2d(x,rxw,[1,1,1,1],'SAME')
        rhw_r=tensorflow.nn.conv2d(h_old,rhw,[1,1,1,1],'SAME')
        rz=rxw_r+rhw_r+rb
        rz=tensorflow.contrib.layers.layer_norm(rz)
        r=tensorflow.nn.sigmoid(rz)

        uxw=tensorflow.get_variable('uxw',[3,3,in_dim[1],out_dim[1]])
        uhw=tensorflow.get_variable('uhw',[3,3,out_dim[1],out_dim[1]])
        ub=tensorflow.get_variable('ub',out_dim[1])
        uxw_r=tensorflow.nn.conv2d(x,uxw,[1,1,1,1],'SAME')
        uhw_r=tensorflow.nn.conv2d(h_old,uhw,[1,1,1,1],'SAME')
        uz=uxw_r+uhw_r+ub
        uz=tensorflow.contrib.layers.layer_norm(uz)
        u=tensorflow.nn.sigmoid(uz)

        txw=tensorflow.get_variable('txw',[3,3,in_dim[1],out_dim[1]])
        thw=tensorflow.get_variable('thw',[3,3,out_dim[1],out_dim[1]])
        tb=tensorflow.get_variable('tb',out_dim[1])
        txw_r=tensorflow.nn.conv2d(x,txw,[1,1,1,1],'SAME')
        thw_r=tensorflow.nn.conv2d(h_old,thw,[1,1,1,1],'SAME')
        tz=txw_r+r*thw_r+tb
        tz=tensorflow.contrib.layers.layer_norm(tz)
        t=tensorflow.nn.selu(tz)

        h_new=(1-u)*t+u*h_old
    return h_new

def encode(x,h_old1,h_old2,h_old3):
    with tensorflow.variable_scope('encode',reuse=tensorflow.AUTO_REUSE):
        conv_w1=tensorflow.get_variable('conv_w1', [3,3,input_dim[1],layer_dim1[1]], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        conv_b1=tensorflow.get_variable('conv_b1', layer_dim1[1], initializer=tensorflow.constant_initializer(0))
        conv_z1=tensorflow.nn.conv2d(x,conv_w1,[1,2,2,1],'SAME')+conv_b1
        conv_z1=tensorflow.contrib.layers.layer_norm(conv_z1)
        conv_z1=tensorflow.nn.selu(conv_z1)

        gru_z1=convgru(conv_z1,h_old1,'1',layer_dim1,layer_dim1)

        conv_w2=tensorflow.get_variable('conv_w2', [3,3,layer_dim1[1],layer_dim2[1]], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        conv_b2=tensorflow.get_variable('conv_b2', layer_dim2[1], initializer=tensorflow.constant_initializer(0))
        conv_z2=tensorflow.nn.conv2d(gru_z1,conv_w2,[1,2,2,1],'SAME')+conv_b2
        conv_z2=tensorflow.contrib.layers.layer_norm(conv_z2)
        conv_z2=tensorflow.nn.selu(conv_z2)

        gru_z2=convgru(conv_z2,h_old2,'2',layer_dim2,layer_dim2)

        conv_w3=tensorflow.get_variable('conv_w3', [3,3,layer_dim2[1],layer_dim3[1]], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        conv_b3=tensorflow.get_variable('conv_b3', layer_dim3[1], initializer=tensorflow.constant_initializer(0))
        conv_z3=tensorflow.nn.conv2d(gru_z2,conv_w3,[1,2,2,1],'SAME')+conv_b3
        conv_z3=tensorflow.contrib.layers.layer_norm(conv_z3)
        conv_z3=tensorflow.nn.selu(conv_z3)

        gru_z3=convgru(conv_z3,h_old3,'3',layer_dim3,layer_dim3)
    return gru_z1, gru_z2, gru_z3

def decode(x,h_old1,h_old2,h_old3):
    with tensorflow.variable_scope('decode',reuse=tensorflow.AUTO_REUSE):
        gru_z1=convgru(x,h_old1,'1',layer_dim3,layer_dim3)

        deconv_w1=tensorflow.get_variable('deconv_w1', [3,3,layer_dim3[1],layer_dim2[1]], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        deconv_b1=tensorflow.get_variable('deconv_b1', layer_dim2[1], initializer=tensorflow.constant_initializer(0))
        deconv_z1=tensorflow.nn.conv2d(tensorflow.image.resize_nearest_neighbor(gru_z1,[layer_dim2[0],layer_dim2[0]]),deconv_w1,[1,1,1,1],'SAME')+deconv_b1
        deconv_z1=tensorflow.contrib.layers.layer_norm(deconv_z1)
        deconv_z1=tensorflow.nn.selu(deconv_z1)

        gru_z2=convgru(deconv_z1,h_old2,'2',layer_dim2,layer_dim2)

        deconv_w2=tensorflow.get_variable('deconv_w2', [3,3,layer_dim2[1],layer_dim1[1]], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        deconv_b2=tensorflow.get_variable('deconv_b2', layer_dim1[1], initializer=tensorflow.constant_initializer(0))
        deconv_z2=tensorflow.nn.conv2d(tensorflow.image.resize_nearest_neighbor(gru_z2,[layer_dim1[0],layer_dim1[0]]),deconv_w2,[1,1,1,1],'SAME')+deconv_b2
        deconv_z2=tensorflow.contrib.layers.layer_norm(deconv_z2)
        deconv_z2=tensorflow.nn.selu(deconv_z2)

        gru_z3=convgru(deconv_z2,h_old3,'3',layer_dim1,layer_dim1) 

        deconv_w3=tensorflow.get_variable('deconv_w3', [3,3,layer_dim1[1],input_dim[1]], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        deconv_b3=tensorflow.get_variable('deconv_b3', input_dim[1], initializer=tensorflow.constant_initializer(0))
        deconv_z3=tensorflow.nn.conv2d(tensorflow.image.resize_nearest_neighbor(gru_z3,[input_dim[0],input_dim[0]]),deconv_w3,[1,1,1,1],'SAME')+deconv_b3
        deconv_z3=tensorflow.contrib.layers.layer_norm(deconv_z3)
        deconv_z3=tensorflow.nn.tanh(deconv_z3)

        deconv_z3=tensorflow.clip_by_value(deconv_z3,-0.5,1)
        k=tensorflow.constant(255,tensorflow.float32,[1,501,501,1])
        output_image=tensorflow.where(tensorflow.less(deconv_z3,0),k,deconv_z3*80)
    return gru_z1, gru_z2, gru_z3, deconv_z3, output_image

def process(input_image):
    init_hide1=numpy.zeros([1,251,251,layer_dim1[1]]).astype(numpy.float32)
    init_hide2=numpy.zeros([1,126,126,layer_dim2[1]]).astype(numpy.float32)
    init_hide3=numpy.zeros([1,63,63,layer_dim3[1]]).astype(numpy.float32)
    encode_output=[]
    for i in range(7):
        if i==0:
            output_hide=encode(input_image[i:i+1],init_hide1,init_hide2,init_hide3)
            encode_output.append(output_hide)
        else:
            output_hide=encode(input_image[i:i+1],output_hide[0],output_hide[1],output_hide[2])
            encode_output.append(output_hide)
    result=[]
    predict_output=[]
    for i in range(6):
        if i==0:
            output=decode(init_hide3,output_hide[2],output_hide[1],output_hide[0])
            predict_output.append(output[3])
            result.append(output[4])
        else:
            output=decode(init_hide3,output[0],output[1],output[2])
            predict_output.append(output[3])   
            result.append(output[4])  
    return predict_output, result

len([x.name for x in tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES)])

origin_image=tensorflow.placeholder(tensorflow.float32,[None,501,501,1],name='origin_image')
trans_image=tensorflow.placeholder(tensorflow.float32,[None,501,501,1],name='trans_image')
global_step = tensorflow.get_variable('global_step',initializer=0, trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step,decay_rate)

process_image, result=process(trans_image)

for i in range(6):
    locals()['loss%s'%i]=tensorflow.losses.mean_squared_error(process_image[i],trans_image[7+i:7+i+1])

AdamOptimizer=tensorflow.train.AdamOptimizer(learning_rate)
for i in range(6):
    if i==0:
        locals()['minimize%s'%i]=AdamOptimizer.minimize(locals()['loss%s'%i],global_step=global_step,name='minimize%s'%i)
    else:
        locals()['minimize%s'%i]=AdamOptimizer.minimize(locals()['loss%s'%i],name='minimize%s'%i)

Saver = tensorflow.train.Saver(max_to_keep=0,filename='convgru')

Session=tensorflow.Session()
if tensorflow.train.latest_checkpoint(model_dir):
    Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))
else:
    Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss1', loss0)
tensorflow.summary.scalar('loss2', loss1)
tensorflow.summary.scalar('loss3', loss2)
tensorflow.summary.scalar('loss4', loss3)
tensorflow.summary.scalar('loss5', loss4)
tensorflow.summary.scalar('loss6', loss5)
tensorflow.summary.image('true image1',origin_image[7:8])
tensorflow.summary.image('true image2',origin_image[8:9])
tensorflow.summary.image('true image3',origin_image[9:10])
tensorflow.summary.image('true image4',origin_image[10:11])
tensorflow.summary.image('true image5',origin_image[11:12])
tensorflow.summary.image('true image6',origin_image[12:13])
tensorflow.summary.image('predict image1',result[0])
tensorflow.summary.image('predict image2',result[1])
tensorflow.summary.image('predict image3',result[2])
tensorflow.summary.image('predict image4',result[3])
tensorflow.summary.image('predict image5',result[4])
tensorflow.summary.image('predict image6',result[5])
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
    all_image_dir=[all_image_dir[y] for y in [x*5 for x in range(13)]]
    all_image=[skimage.io.imread(x) for x in all_image_dir]
    all_image=numpy.array(all_image).reshape(-1,501,501,1)
    all_image_new=numpy.where(all_image>80,-40,all_image)/80

    for i in range(6):
        Session.run(locals()['minimize%s'%i],feed_dict={trans_image:all_image_new})
    if Session.run(global_step)%100==1:
        summary = Session.run(merge_all, feed_dict={trans_image:all_image_new,origin_image:all_image})
        FileWriter.add_summary(summary, Session.run(global_step))
        Saver.save(Session, model_dir, global_step)
    print(Session.run(global_step))