import numpy
import tensorflow
import pandas
import data_loading
import argparse
import sys

flag=None
is_train = tensorflow.placeholder(tensorflow.bool)
iteration = tensorflow.placeholder(tensorflow.int32)

def multilayer_perceptron(x):
    with tensorflow.name_scope('fc1'):
        w1 = tensorflow.Variable(tensorflow.truncated_normal([flag.input_dimension, flag.fc1_dimension], stddev=0.1),name='w1')
        b1 = tensorflow.Variable(tensorflow.constant(0.1,shape=[flag.fc1_dimension]),name='b1')
        z1 = tensorflow.matmul(x,w1) + b1
        fc1=tensorflow.nn.leaky_relu(z1)

    with tensorflow.name_scope('fc2'):
        w2 = tensorflow.Variable(tensorflow.truncated_normal([flag.fc1_dimension, flag.fc2_dimension],stddev=0.1),name='w2')
        b2 = tensorflow.Variable(tensorflow.constant(0.1,shape=[flag.fc2_dimension]),name='b2')
        z2 = tensorflow.matmul(fc1,w2) + b2
        fc2=tensorflow.nn.leaky_relu(z2)

    with tensorflow.name_scope('fc3'):
        w3 = tensorflow.Variable(tensorflow.truncated_normal([flag.fc2_dimension, flag.fc3_dimension], stddev=0.1),name='w3')
        b3 = tensorflow.Variable(tensorflow.constant(0.1,shape=[flag.fc3_dimension]),name='b3')
        z3 = tensorflow.matmul(fc2,w3) + b3
        fc3=tensorflow.nn.leaky_relu(z3)

    with tensorflow.name_scope('fc4'):
        w4 = tensorflow.Variable(tensorflow.truncated_normal([flag.fc3_dimension, flag.fc4_dimension], stddev=0.1),name='w4')
        b4 = tensorflow.Variable(tensorflow.constant(0.1,shape=[flag.fc4_dimension]),name='b4')
        z4 = tensorflow.matmul(fc3,w4) + b4
        fc4=tensorflow.nn.leaky_relu(z4)

    with tensorflow.name_scope('fc5'):
        w5 = tensorflow.Variable(tensorflow.truncated_normal([flag.fc4_dimension, flag.fc5_dimension], stddev=0.1),name='w5')
        b5 = tensorflow.Variable(tensorflow.constant(0.1,shape=[flag.fc5_dimension]),name='b5')
        z5 = tensorflow.matmul(fc4,w5) + b5
        fc5=tensorflow.nn.leaky_relu(z5)

    with tensorflow.name_scope('dropout1'):
        fc5_dropout = tensorflow.contrib.layers.dropout(fc5, flag.keep_prob, is_training=is_train)

    with tensorflow.name_scope('fc6'):
        w6 = tensorflow.Variable(tensorflow.truncated_normal([flag.fc5_dimension, flag.output_dimension], stddev=0.1),name='w6')
        b6 = tensorflow.Variable(tensorflow.constant(0.1,shape=[flag.output_dimension]),name='b6')
        y = tensorflow.matmul(fc5_dropout, w6) + b6

    tensorflow.add_to_collection('loss',tensorflow.contrib.layers.l2_regularizer(flag.L2_ratio)(w1))
    tensorflow.add_to_collection('loss',tensorflow.contrib.layers.l2_regularizer(flag.L2_ratio)(b1))
    tensorflow.add_to_collection('loss',tensorflow.contrib.layers.l2_regularizer(flag.L2_ratio)(w2))
    tensorflow.add_to_collection('loss',tensorflow.contrib.layers.l2_regularizer(flag.L2_ratio)(b2))
    tensorflow.add_to_collection('loss',tensorflow.contrib.layers.l2_regularizer(flag.L2_ratio)(w3))
    tensorflow.add_to_collection('loss',tensorflow.contrib.layers.l2_regularizer(flag.L2_ratio)(b3))
    tensorflow.add_to_collection('loss',tensorflow.contrib.layers.l2_regularizer(flag.L2_ratio)(w4))
    tensorflow.add_to_collection('loss',tensorflow.contrib.layers.l2_regularizer(flag.L2_ratio)(b4))
    tensorflow.add_to_collection('loss',tensorflow.contrib.layers.l2_regularizer(flag.L2_ratio)(w5))
    tensorflow.add_to_collection('loss',tensorflow.contrib.layers.l2_regularizer(flag.L2_ratio)(b5))
    tensorflow.add_to_collection('loss',tensorflow.contrib.layers.l2_regularizer(flag.L2_ratio)(w6))
    tensorflow.add_to_collection('loss',tensorflow.contrib.layers.l2_regularizer(flag.L2_ratio)(b6))

    return y

def main(_):
    x = tensorflow.placeholder(tensorflow.float32, [None, flag.input_dimension])
    y_ = tensorflow.placeholder(tensorflow.float32, [None, flag.output_dimension])
    keep_prob = tensorflow.placeholder(tensorflow.float32)
    lr = tensorflow.placeholder(tensorflow.float32)
    
    train_x, train_y, test_x=data_loading.get_data_1dimension()

    y=multilayer_perceptron(x, )

    with tensorflow.name_scope('total_loss'):
        loss = tensorflow.reduce_mean(tensorflow.square(y_ - y), reduction_indices=[0])
    losses = loss+tensorflow.add_n(tensorflow.get_collection('loss'))

    with tensorflow.name_scope('optimizer'):
        train_step = tensorflow.train.AdamOptimizer(lr).minimize(losses)

    with tensorflow.name_scope('accuracy'):
        accuracy = tensorflow.reduce_mean(tensorflow.reduce_sum(tensorflow.square(y_ - y), reduction_indices=[1]))

    with tensorflow.Session() as sess:
        sess.run(tensorflow.global_variables_initializer())
        num = train_x.shape[0] // flag.batch_size
        for i in range(flag.max_iteration):
            temp_x = train_x[i % num * flag.batch_size:i % num * flag.batch_size+flag.batch_size,:]
            temp_y = train_y[i % num * flag.batch_size:i % num * flag.batch_size+flag.batch_size,:]
            learning_rate = flag.max_learning_rate - (flag.max_learning_rate - flag.min_learning_rate)*(i / flag.max_iteration)
            sess.run(train_step, feed_dict={x:temp_x, y_:temp_y, lr:learning_rate, is_train:True})
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:temp_x, y_:temp_y, is_train:False})
                print('step %d, training loss %g' % (i, train_accuracy))

        score = accuracy.eval(feed_dict={x:train_x, y_:train_y, is_train:False})
        print('final loss %g' % (score))

        predict_resault=sess.run(y, feed_dict={x:test_x, is_train:False})
        pandas.DataFrame(predict_resault).to_csv('mlp %s.csv'%numpy.around(score, decimals=1), index=False, header=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_dir', default='summary/')
    parser.add_argument('--batch_size', default=500)
    parser.add_argument('--max_iteration', default=30000)
    parser.add_argument('--max_learning_rate', default=0.01)
    parser.add_argument('--min_learning_rate', default=0.00001)
    parser.add_argument('--input_dimension', default=40)
    parser.add_argument('--fc1_dimension', default=50)
    parser.add_argument('--fc2_dimension', default=40)
    parser.add_argument('--fc3_dimension', default=30)
    parser.add_argument('--fc4_dimension', default=20)
    parser.add_argument('--fc5_dimension', default=10)
    parser.add_argument('--output_dimension', default=1)
    parser.add_argument('--L2_ratio', default=1e-3)
    parser.add_argument('--keep_prob', default=0.8)
    flag, unparsed = parser.parse_known_args()
    tensorflow.app.run(main=main, argv=[sys.argv[0]] + unparsed)