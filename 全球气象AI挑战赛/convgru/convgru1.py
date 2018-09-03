import tensorflow
import numpy
import os
import random
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
input_dim=1
hide_dim=4
output_dim=1

def convgru(h_old,x):
    with tensorflow.variable_scope('convgru', reuse=tensorflow.AUTO_REUSE):
        rxw=tensorflow.get_variable('rxw',[3,3,input_dim,hide_dim])
        rhw=tensorflow.get_variable('rhw',[3,3,hide_dim,hide_dim])
        rb=tensorflow.get_variable('rb',hide_dim)
        rxw_r=tensorflow.nn.conv2d(x,rxw,[1,1,1,1],'SAME')
        rhw_r=tensorflow.nn.conv2d(h_old,rhw,[1,1,1,1],'SAME')
        rz=rxw_r+rhw_r+rb
        rz=tensorflow.contrib.layers.layer_norm(rz)
        r=tensorflow.nn.sigmoid(rz)

        uxw=tensorflow.get_variable('uxw',[3,3,input_dim,hide_dim])
        uhw=tensorflow.get_variable('uhw',[3,3,hide_dim,hide_dim])
        ub=tensorflow.get_variable('ub',hide_dim)
        uxw_r=tensorflow.nn.conv2d(x,uxw,[1,1,1,1],'SAME')
        uhw_r=tensorflow.nn.conv2d(h_old,uhw,[1,1,1,1],'SAME')
        uz=uxw_r+uhw_r+ub
        uz=tensorflow.contrib.layers.layer_norm(uz)
        u=tensorflow.nn.sigmoid(uz)

        txw=tensorflow.get_variable('txw',[3,3,input_dim,hide_dim])
        thw=tensorflow.get_variable('thw',[3,3,hide_dim,hide_dim])
        tb=tensorflow.get_variable('tb',hide_dim)
        txw_r=tensorflow.nn.conv2d(x,txw,[1,1,1,1],'SAME')
        thw_r=tensorflow.nn.conv2d(r*h_old,thw,[1,1,1,1],'SAME')
        tz=txw_r+thw_r+tb
        tz=tensorflow.contrib.layers.layer_norm(tz)
        t=tensorflow.nn.tanh(tz)

        h_new=(1-u)*h_old+u*t
        return h_new

def predict(x):
    with tensorflow.variable_scope('predict',reuse=tensorflow.AUTO_REUSE):
        w=tensorflow.get_variable('w', [3,3,hide_dim,output_dim], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b=tensorflow.get_variable('b', output_dim, initializer=tensorflow.constant_initializer(0))
        z=tensorflow.nn.conv2d(x,w,[1,1,1,1],'SAME')+b
        z=tensorflow.contrib.layers.layer_norm(z)
        z=tensorflow.nn.tanh(z)
        z=tensorflow.clip_by_value(z*128+128,0,255)
    return z

def train_stage(input_image):
    all_output=[]
    init_hide=numpy.zeros([batch_size,501,501,hide_dim]).astype(numpy.float32)
    for i in range(61):
        if i==0:
            output_hide=convgru(init_hide,input_image[:,i,:,:,:])
            all_output.append(output_hide)
        else:
            output_hide=convgru(output_hide,input_image[:,i,:,:,:])
            all_output.append(output_hide)

    return all_output

def precidt_stage(input_image):
    all_output=[]
    init_hide=numpy.zeros([batch_size,501,501,hide_dim]).astype(numpy.float32)
    for i in range(31):
        if i==0:
            output_hide=convgru(init_hide,input_image[:,i,:,:,:])
            all_output.append(output_hide)
        else:
            output_hide=convgru(output_hide,input_image[:,i,:,:,:])
            all_output.append(output_hide)

    for i in range(31,61):
        output_hide=convgru(output_hide,predict(output_hide))
        all_output.append(output_hide)           

    return all_output

len([x.name for x in tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES)])

input_image=tensorflow.placeholder(tensorflow.float32,[batch_size,None,501,501,1],name='input_image')
global_step = tensorflow.get_variable('global_step',initializer=0, trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step,decay_rate)
which_minimize=tensorflow.placeholder(tensorflow.int32,name='which_minimize')

train_output=train_stage(input_image)
precidt_output=precidt_stage(input_image)
train_decode_image=tensorflow.map_fn(predict,tensorflow.stack(train_output,1),name='train_decode_image')
predict_decode_image=tensorflow.map_fn(predict,tensorflow.stack(precidt_output,1),name='predict_decode_image')

loss=tensorflow.losses.mean_squared_error(train_decode_image[:,which_minimize,:,:,:],input_image[:,which_minimize+1,:,:,:])

with tensorflow.control_dependencies(tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS)):
    minimize=tensorflow.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step,name='minimize')

Saver = tensorflow.train.Saver(max_to_keep=0,filename='convgru')

Session=tensorflow.Session()
if tensorflow.train.latest_checkpoint(model_dir):
    Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))
else:
    Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss', loss)
tensorflow.summary.image('true image',input_image[0,31:],10)
tensorflow.summary.image('predict image',predict_decode_image[0,30:],10)
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

    Session.run(minimize,feed_dict={input_image:all_image})
    if Session.run(global_step)%100==1:
        summary = Session.run(merge_all, feed_dict={input_image:all_image})
        FileWriter.add_summary(summary, Session.run(global_step))
        Saver.save(Session, model_dir, global_step)
        print(Session.run(loss,feed_dict={input_image:all_image}))

    print(Session.run(global_step))