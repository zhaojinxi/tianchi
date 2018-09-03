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

def convlstm(c_old,h_old,x):
    with tensorflow.variable_scope('convlstm', reuse=tensorflow.AUTO_REUSE):
        fxw=tensorflow.get_variable('fxw',[3,3,input_dim,hide_dim])
        fhw=tensorflow.get_variable('fhw',[3,3,hide_dim,hide_dim])
        fcw=tensorflow.get_variable('fcw',[batch_size,501,501,hide_dim])
        fb=tensorflow.get_variable('fb',hide_dim)
        fxw_r=tensorflow.nn.conv2d(x,fxw,[1,1,1,1],'SAME')
        fhw_r=tensorflow.nn.conv2d(h_old,fhw,[1,1,1,1],'SAME')
        fcw_r=c_old*fcw
        fz=fxw_r+fhw_r+fcw_r+fb
        fz=tensorflow.contrib.layers.layer_norm(fz)
        f=tensorflow.nn.sigmoid(fz)

        ixw=tensorflow.get_variable('ixw',[3,3,input_dim,hide_dim])
        ihw=tensorflow.get_variable('ihw',[3,3,hide_dim,hide_dim])
        icw=tensorflow.get_variable('icw',[batch_size,501,501,hide_dim])
        ib=tensorflow.get_variable('ib',hide_dim)
        ixw_r=tensorflow.nn.conv2d(x,ixw,[1,1,1,1],'SAME')
        ihw_r=tensorflow.nn.conv2d(h_old,ihw,[1,1,1,1],'SAME')
        icw_r=c_old*icw
        iz=ixw_r+ihw_r+icw_r+ib
        iz=tensorflow.contrib.layers.layer_norm(iz)
        i=tensorflow.nn.sigmoid(iz)

        txw=tensorflow.get_variable('txw',[3,3,input_dim,hide_dim])
        thw=tensorflow.get_variable('thw',[3,3,hide_dim,hide_dim])
        tb=tensorflow.get_variable('tb',hide_dim)
        txw_r=tensorflow.nn.conv2d(x,txw,[1,1,1,1],'SAME')
        thw_r=tensorflow.nn.conv2d(h_old,thw,[1,1,1,1],'SAME')
        tz=txw_r+thw_r+tb
        tz=tensorflow.contrib.layers.layer_norm(tz)
        t=tensorflow.nn.tanh(tz)

        c_new=f*c_old+i*t

        oxw=tensorflow.get_variable('oxw',[3,3,input_dim,hide_dim])
        ohw=tensorflow.get_variable('ohw',[3,3,hide_dim,hide_dim])
        ocw=tensorflow.get_variable('ocw',[batch_size,501,501,hide_dim])
        ob=tensorflow.get_variable('ob',hide_dim)
        oxw_r=tensorflow.nn.conv2d(x,oxw,[1,1,1,1],'SAME')
        ohw_r=tensorflow.nn.conv2d(h_old,ohw,[1,1,1,1],'SAME')
        ocw_r=c_old*ocw
        oz=oxw_r+ohw_r+ocw_r+ob
        oz=tensorflow.contrib.layers.layer_norm(oz)
        o=tensorflow.nn.sigmoid(oz)

        h_new=o*tensorflow.nn.tanh(c_new)
    
    return c_new,h_new

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
    init_cell=numpy.zeros([batch_size,501,501,hide_dim]).astype(numpy.float32)
    init_hide=numpy.zeros([batch_size,501,501,hide_dim]).astype(numpy.float32)
    for i in range(61):
        if i==0:
            output_cell, output_hide=convlstm(init_cell,init_hide,input_image[:,i,:,:,:])
            all_output.append(output_hide)
        else:
            output_cell, output_hide=convlstm(output_cell,output_hide,input_image[:,i,:,:,:])
            all_output.append(output_hide)

    return output_cell, all_output

def precidt_stage(input_image):
    all_output=[]
    init_cell=numpy.zeros([batch_size,501,501,hide_dim]).astype(numpy.float32)
    init_hide=numpy.zeros([batch_size,501,501,hide_dim]).astype(numpy.float32)
    for i in range(31):
        if i==0:
            output_cell, output_hide=convlstm(init_cell,init_hide,input_image[:,i,:,:,:])
            all_output.append(output_hide)
        else:
            output_cell, output_hide=convlstm(output_cell,output_hide,input_image[:,i,:,:,:])
            all_output.append(output_hide)

    for i in range(31,61):
        output_cell, output_hide=convlstm(output_cell,output_hide,predict(output_hide))
        all_output.append(output_hide)           

    return output_cell, all_output

len([x.name for x in tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES)])

train_image=tensorflow.placeholder(tensorflow.float32,[batch_size,61,501,501,1],name='train_image')
precidt_image=tensorflow.placeholder(tensorflow.float32,[batch_size,31,501,501,1],name='precidt_image')
global_step = tensorflow.get_variable('global_step',initializer=0, trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step,decay_rate)

train_output=train_stage(train_image)
# precidt_output=precidt_stage(precidt_image)
train_decode_image=tensorflow.map_fn(predict,tensorflow.stack(train_output[1],1),name='train_decode_image')
# predict_decode_image=tensorflow.map_fn(predict,tensorflow.stack(precidt_output[1],1),name='predict_decode_image')

loss=0
for i in range(60):
    loss=loss+tensorflow.losses.mean_squared_error(train_decode_image[:,i,:,:,:],train_image[:,i+1,:,:,:])
loss=loss/60

with tensorflow.control_dependencies(tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS)):
    minimize=tensorflow.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step,name='minimize')

Saver = tensorflow.train.Saver(max_to_keep=0,filename='convlstm')

Session=tensorflow.Session()
if tensorflow.train.latest_checkpoint(model_dir):
    Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))
else:
    Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss', loss)
tensorflow.summary.image('true image',train_image[0,31:],10)
tensorflow.summary.image('predict image',train_decode_image[0,30:],10)
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
        Session.run(minimize,feed_dict={train_image:all_image})
        if Session.run(global_step)%100==1:
            summary = Session.run(merge_all, feed_dict={train_image:all_image})
            FileWriter.add_summary(summary, Session.run(global_step))
            Saver.save(Session, model_dir, global_step)
            print(Session.run(loss,feed_dict={train_image:all_image}))
    except:
        with open('log/异常数据目录.txt','a') as f:
            f.write('异常数据:%s\n'%(rads))

    print(Session.run(global_step))