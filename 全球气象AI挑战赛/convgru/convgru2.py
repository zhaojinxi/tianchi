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

def convgru(h_old,x):
    with tensorflow.variable_scope('convgru%s'%name,reuse=tensorflow.AUTO_REUSE):
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

def process(input_image):
    init_hide=numpy.zeros([batch_size,501,501,hide_dim]).astype(numpy.float32)
    all_output=[]
    for i in range(31):
        if i==0:
            output_hide=convgru(init_hide,input_image[:,i,:,:,:])
            all_output.append(predict(output_hide))
        else:
            output_hide=convgru(output_hide,input_image[:,i,:,:,:])
            all_output.append(predict(output_hide))
    for i in range(31,61):
        output_hide=convgru(output_hide,predict(output_hide))
        all_output.append(predict(output_hide))
    return all_output

len([x.name for x in tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES)])

input_image=tensorflow.placeholder(tensorflow.float32,[batch_size,None,501,501,1],name='input_image')
global_step = tensorflow.get_variable('global_step',initializer=0, trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step,decay_rate)

process_image=process(input_image)

for i in range(61):
    locals()['loss%s'%i]=tensorflow.losses.mean_squared_error(process_image[i],input_image[:,i+1,:,:,:])

AdamOptimizer=tensorflow.train.AdamOptimizer(learning_rate)
for i in range(61):
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

tensorflow.summary.scalar('loss1', loss4)
tensorflow.summary.scalar('loss2', loss9)
tensorflow.summary.scalar('loss3', loss14)
tensorflow.summary.scalar('loss4', loss19)
tensorflow.summary.scalar('loss5', loss24)
tensorflow.summary.scalar('loss6', loss29)
tensorflow.summary.image('true image1',input_image[0,35],batch_size)
tensorflow.summary.image('true image2',input_image[0,40],batch_size)
tensorflow.summary.image('true image3',input_image[0,45],batch_size)
tensorflow.summary.image('true image4',input_image[0,50],batch_size)
tensorflow.summary.image('true image5',input_image[0,55],batch_size)
tensorflow.summary.image('true image6',input_image[0,60],batch_size)
tensorflow.summary.image('predict image1',process_image[34],batch_size)
tensorflow.summary.image('predict image2',process_image[39],batch_size)
tensorflow.summary.image('predict image3',process_image[44],batch_size)
tensorflow.summary.image('predict image4',process_image[49],batch_size)
tensorflow.summary.image('predict image5',process_image[54],batch_size)
tensorflow.summary.image('predict image6',process_image[59],batch_size)
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

    for i in range(60):
        Session.run(locals()['minimize%s'%i],feed_dict={input_image:all_image})
    if Session.run(global_step)%100==1:
        summary = Session.run(merge_all, feed_dict={input_image:all_image})
        FileWriter.add_summary(summary, Session.run(global_step))
        Saver.save(Session, model_dir, global_step)
        print(Session.run(locals()['loss%s'%i],feed_dict={input_image:all_image}))

    print(Session.run(global_step))