import numpy
import tensorflow
import pandas
import argparse
import sys
import data_loading

tensorflow.contrib.eager.enable_eager_execution()

flag = None
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='/gpu:0')
parser.add_argument('--summary_dir', default='summary/')
parser.add_argument('--model_dir', default='model/')
parser.add_argument('--batch_size', default=500)
parser.add_argument('--repeat', default=40)
parser.add_argument('--max_iteration', default=30000)
parser.add_argument('--max_learning_rate', default=0.01)
parser.add_argument('--min_learning_rate', default=0.0001)
parser.add_argument('--input_dimension', default=[None,1,4,4,4])
parser.add_argument('--cnn1_dimension', default=[2,2,2,1,8])
parser.add_argument('--cnn2_dimension', default=[2,2,2,8,16])
parser.add_argument('--cnn3_dimension', default=[2,2,2,16,32])
parser.add_argument('--cnn4_dimension', default=[2,2,2,32,16])
parser.add_argument('--fc5_dimension', default=20)
parser.add_argument('--output_dimension', default=1)
parser.add_argument('--keep_prob', default=0.8)
flag, unparsed = parser.parse_known_args()

train_x, train_y, test_x = data_loading.get_data_3dimension()

with tensorflow.device(flag.device):
    w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal(flag.cnn1_dimension, stddev=0.1), name='w1')
    b1 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.cnn1_dimension[4]), name='b1')
    w2 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal(flag.cnn2_dimension, stddev=0.1), name='w2')
    b2 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.cnn2_dimension[4]), name='b2')
    w3 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal(flag.cnn3_dimension, stddev=0.1), name='w3')
    b3 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.cnn3_dimension[4]), name='b3')
    w4 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal(flag.cnn4_dimension, stddev=0.1), name='w4')
    b4 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.cnn4_dimension[4]), name='b4')
    w5 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal((128, flag.fc5_dimension), stddev=0.1), name='w5')
    b5 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.fc5_dimension), name='b5')
    w6 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal((flag.fc5_dimension, flag.output_dimension), stddev=0.1), name='w6')
    b6 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.output_dimension), name='b6')

def model(x, is_train):
    h1 = tensorflow.nn.leaky_relu(tensorflow.nn.conv3d(x, w1, strides=[1,1,1,1,1], padding='SAME') + b1)

    h2 = tensorflow.nn.leaky_relu(tensorflow.nn.conv3d(h1, w2, strides=[1,1,1,1,1], padding='VALID') + b2)

    h3 = tensorflow.nn.leaky_relu(tensorflow.nn.conv3d(h2, w3, strides=[1,1,1,1,1], padding='SAME') + b3)

    h4 = tensorflow.nn.leaky_relu(tensorflow.nn.conv3d(h3, w4, strides=[1,1,1,1,1], padding='VALID') + b4)
    h4 = tensorflow.reshape(h4, (x.shape[0], -1))

    h5 = tensorflow.nn.leaky_relu(tensorflow.matmul(h4, w5) + b5)
    h5 = tensorflow.contrib.layers.dropout(h5, flag.keep_prob, is_training=is_train)

    out = tensorflow.matmul(h5, w6) + b6

    return out

@tensorflow.contrib.eager.implicit_value_and_gradients
def loss_fun(x, y_, is_train):
    y = model(x, is_train)
    loss = tensorflow.reduce_mean(tensorflow.square(y_ - y), reduction_indices=[0])
    return loss

def score(x, y_, is_train):
    y = model(x, is_train)
    accuracy = tensorflow.reduce_mean(tensorflow.square(y_ - y), reduction_indices=[0])
    return accuracy

def main(_):
    optimizer = tensorflow.train.AdamOptimizer()

    with tensorflow.device(flag.device):
        num = train_x.shape[0] // flag.batch_size
        for i in range(flag.max_iteration):
            x = train_x[i % num * flag.batch_size:i % num * flag.batch_size+flag.batch_size,:]
            y_ = train_y[i % num * flag.batch_size:i % num * flag.batch_size+flag.batch_size,:]
            learning_rate = flag.max_learning_rate - (flag.max_learning_rate - flag.min_learning_rate)*(i / flag.max_iteration)
            optimizer._lr = learning_rate
            _, gradient = loss_fun(x, y_, True)
            optimizer.apply_gradients(gradient)
            if i % 100 == 0:
                loss = score(train_x, train_y, False)
                print("step: {}  score: {}".format(i, loss.numpy()))
    with tensorflow.device(flag.device):
        accuracy = score(x, y_, False)
    print("final loss: {}".format(accuracy.numpy()))

    with tensorflow.device(flag.device):
        predict=model(test_x, False)
    pandas.DataFrame(predict.numpy()).to_csv('cnn_3d %s.csv'%numpy.around(accuracy.numpy()[0], decimals=1), index=False, header=False)

    Saver = tensorflow.contrib.eager.Saver([w1, w2, w3, w4, w5, w6, b1, b2, b3, b4, b5, b6])
    Saver.save(flag.model_dir + 'cnn_3d')

if __name__ == '__main__':
    tensorflow.app.run(main=main, argv=[sys.argv[0]] + unparsed)