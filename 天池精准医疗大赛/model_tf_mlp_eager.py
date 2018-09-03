import numpy
import tensorflow
import pandas
import argparse
import sys
import data_loading

tensorflow.contrib.eager.enable_eager_execution()

flag = None
parser = argparse.ArgumentParser()
parser.add_argument('--summary_dir', default='summary/')
parser.add_argument('--batch_size', default=500)
parser.add_argument('--repeat', default=300)
parser.add_argument('--max_iteration', default=30000)
parser.add_argument('--max_learning_rate', default=0.01)
parser.add_argument('--min_learning_rate', default=0.0001)
parser.add_argument('--input_dimension', default=40)
parser.add_argument('--fc1_dimension', default=50)
parser.add_argument('--fc2_dimension', default=40)
parser.add_argument('--fc3_dimension', default=30)
parser.add_argument('--fc4_dimension', default=20)
parser.add_argument('--fc5_dimension', default=10)
parser.add_argument('--output_dimension', default=1)
parser.add_argument('--keep_prob', default=0.8)
flag, unparsed = parser.parse_known_args()

train_x, train_y, test_x = data_loading.get_data_1dimension()

w1 = tensorflow.get_variable(initializer = tensorflow.truncated_normal([flag.input_dimension, flag.fc1_dimension], stddev=0.1), name='w1')
b1 = tensorflow.get_variable(initializer = tensorflow.constant(0.1, shape = flag.fc1_dimension), name='b1')
w2 = tensorflow.get_variable(initializer = tensorflow.truncated_normal([flag.fc1_dimension, flag.fc2_dimension], stddev=0.1), name='w2')
b2 = tensorflow.get_variable(initializer = tensorflow.constant(0.1, shape = flag.fc2_dimension), name='b2')
w3 = tensorflow.get_variable(initializer = tensorflow.truncated_normal([flag.fc2_dimension, flag.fc3_dimension], stddev=0.1), name='w3')
b3 = tensorflow.get_variable(initializer = tensorflow.constant(0.1, shape = flag.fc3_dimension), name='b3')
w4 = tensorflow.get_variable(initializer = tensorflow.truncated_normal([flag.fc3_dimension, flag.fc4_dimension], stddev=0.1), name='w4')
b4 = tensorflow.get_variable(initializer = tensorflow.constant(0.1, shape = flag.fc4_dimension), name='b4')
w5 = tensorflow.get_variable(initializer = tensorflow.truncated_normal([flag.fc4_dimension, flag.fc5_dimension], stddev=0.1), name='w5')
b5 = tensorflow.get_variable(initializer = tensorflow.constant(0.1, shape = flag.fc5_dimension), name='b5')
w6 = tensorflow.get_variable(initializer = tensorflow.truncated_normal([flag.fc5_dimension, flag.output_dimension], stddev=0.1), name='w6')
b6 = tensorflow.get_variable(initializer = tensorflow.constant(0.1, shape = flag.output_dimension), name='b6')

def model(x, y_, is_train):
    fc1 = tensorflow.nn.leaky_relu(tensorflow.matmul(x, w1) + b1)

    fc2 = tensorflow.nn.leaky_relu(tensorflow.matmul(fc1, w2) + b2)

    fc3 = tensorflow.nn.leaky_relu(tensorflow.matmul(fc2, w3) + b3)

    fc4 = tensorflow.nn.leaky_relu(tensorflow.matmul(fc3, w4) + b4)

    fc5 = tensorflow.nn.leaky_relu(tensorflow.matmul(fc4, w5) + b5)
    fc5 = tensorflow.contrib.layers.dropout(fc5, flag.keep_prob, is_training=is_train)

    y = tensorflow.matmul(fc5, w6) + b6

    return y

@tensorflow.contrib.eager.implicit_value_and_gradients
def loss_fun(x, y_, is_train):
    y=model(x, y_, is_train)
    loss = tensorflow.reduce_mean(tensorflow.square(y_ - y), reduction_indices=[0])
    return loss

def score(x, y_, is_train):
    y=model(x, y_, is_train)
    loss = tensorflow.reduce_mean(tensorflow.square(y_ - y), reduction_indices=[0])
    return loss

def main(_):
    optimizer = tensorflow.train.AdamOptimizer()

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

    loss = score(train_x, train_y, False)
    print("final loss: {}".format(loss.numpy()))

    predict = model(test_x, 0, False)
    pandas.DataFrame(predict.numpy()).to_csv('mlp %s.csv'%numpy.around(loss.numpy()[0], decimals=1), index=False, header=False)

if __name__ == '__main__':
    tensorflow.app.run(main=main, argv=[sys.argv[0]] + unparsed)