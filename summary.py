# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_step = 1000
learning_rate = 0.001
dropout = 0.9

data_dir = 'datasets'
log_dir = 'log'

mnist = input_data.read_data_sets(data_dir, one_hot=True)

sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    print(x)
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_input')
    print(y_)
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))

    tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)

        with tf.name_scope('bias'):
            bias = bias_variable([output_dim])
            variable_summaries(bias)

        with tf.name_scope('linear_compute'):
            preactive = tf.matmul(input_tensor, weights) + bias
            tf.summary.histogram('linear', preactive)

        activation = act(preactive, name='activation')
        tf.summary.histogram('activations', activation)
        return activation


hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    print(keep_prob)
    tf.summary.scalar('dropout_keep_probility', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('loss'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correction_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                  name='result')
        print(accuracy)
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(log_dir + 'Train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + 'Test', sess.graph)
saver = tf.train.Saver(max_to_keep=1)

tf.global_variables_initializer().run()


def feed_dict(train):
    if (train):
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0

    return {x: xs, y_: ys, keep_prob: k}


for i in range(max_step):
    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('accuracy at step %s: %s' % (i, acc))
    else:
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)

saver.save(sess, './save/')
train_writer.close()
test_writer.close()