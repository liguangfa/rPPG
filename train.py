from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np
import tensorflow as tf
from tflearn.layers.conv import conv_2d,conv_3d

import y
from y import *
from tflearn.data_utils import image_preloader
import tflearn.datasets.imdb

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 閫夋嫨ID涓?鐨凣PU
print('1:',tf.test.is_gpu_available())


def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2, activation='relu', batch_norm=True,
                             bias=True, weights_init='variance_scaling',
                             bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                             trainable=True, restore=True, reuse=False, scope=None,
                             name="ResidualBlock"):
    # residual shrinkage blocks with channel-wise thresholds

    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1] #计算出输入通道

    # Variable Scope fix for older TF
    try:
        vscope = tf.variable_scope(scope, default_name=name, values=[incoming],
                                   reuse=reuse)
    except Exception:
        vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)

    with vscope as scope:
        name = scope.name  # TODO

        for i in range(nb_blocks):

            identity = residual

            if not downsample:
                downsample_strides = 1

            if batch_norm:
                residual = tflearn.batch_normalization(residual)
            residual = tflearn.activation(residual, activation)
            residual = conv_2d(residual, out_channels, 3,
                               downsample_strides, 'same', 'linear',
                               bias, weights_init, bias_init,
                               regularizer, weight_decay, trainable,
                               restore)

            if batch_norm:
                residual = tflearn.batch_normalization(residual)
            residual = tflearn.activation(residual, activation)
            residual = conv_2d(residual, out_channels, 3, 1, 'same',
                               'linear', bias, weights_init,
                               bias_init, regularizer, weight_decay,
                               trainable, restore)

            # get thresholds and apply thresholding
            abs_mean = tf.reduce_mean(tf.reduce_mean(tf.abs(residual), axis=2, keep_dims=True), axis=1, keep_dims=True)
            scales = tflearn.fully_connected(abs_mean, out_channels // 4, activation='linear', regularizer='L2',
                                             weight_decay=0.0001, weights_init='variance_scaling')
            scales = tflearn.batch_normalization(scales)
            scales = tflearn.activation(scales, 'relu')
            scales = tflearn.fully_connected(scales, out_channels, activation='linear', regularizer='L2',
                                             weight_decay=0.0001, weights_init='variance_scaling')
            scales = tf.expand_dims(tf.expand_dims(scales, axis=1), axis=1)
            thres = tf.multiply(abs_mean, tflearn.activations.sigmoid(scales))
            # soft thresholding
            residual = tf.multiply(tf.sign(residual), tf.maximum(tf.abs(residual) - thres, 0))

            # Downsampling
            if downsample_strides > 1:
                identity = tflearn.avg_pool_2d(identity, 1,
                                               downsample_strides)

            # Projection to new dimension
            if in_channels != out_channels:
                if (out_channels - in_channels) % 2 == 0:
                    ch = (out_channels - in_channels) // 2
                    identity = tf.pad(identity,
                                      [[0, 0], [0, 0], [0, 0], [ch, ch]])
                else:
                    ch = (out_channels - in_channels) // 2
                    identity = tf.pad(identity,
                                      [[0, 0], [0, 0], [0, 0], [ch, ch + 1]])
                in_channels = out_channels

            residual = residual + identity

    return residual

def deep_net(x):
    # Building Deep Residual Shrinkage Network
    #x = tflearn.input_data(shape=[64,128,128,3])
    with tf.Graph().as_default():
        net = tflearn.conv_3d(x, 16, 3, strides=1,regularizer='L2', weight_decay=0.0001) #(64,128,128,3) > (64,128,128,16)
        net=tflearn.batch_normalization(net)
        net=tflearn.relu(net)

        net = tflearn.avg_pool_3d(net, kernel_size=1,strides=(1,2,2)) #(64,128,128,16) > (64, 64, 64, 16)

        net = tflearn.conv_3d(net, 1, 16, strides=1,regularizer='L2', weight_decay=0.0001) #(64,64,64,16) > (64,64,64,1)
        net=tflearn.batch_normalization(net)
        net=tflearn.relu(net)

        net = tflearn.avg_pool_3d(net, kernel_size=1,strides=(1,2,2)) #(64,64,64,1) > (64,32,32,1)
        net=tf.squeeze(net,-1) #(64,32,32,1) > (64,32,32)
        net=tf.transpose(net,perm=[0,2,3,1]) #(N,64,32,32) > (N,32,32,64)

        net = residual_shrinkage_block(net, 1, 64) #1是blocks,16是output_channels,(32,32,64) > (32,32,64)
        net = residual_shrinkage_block(net, 1, 64, downsample=True) #(32,32,64) > (32,32,64)
        net = residual_shrinkage_block(net, 1, 64, downsample=True) #(32,32,64) >(32,32,64)
        net = tflearn.batch_normalization(net) #(32,32,64) > (32,32,64),对输入进行归一化
        net = tflearn.activation(net, 'relu') #
        net = tflearn.global_avg_pool(net) #输入4维，输出2维，(N,32,32,64) > (N,64)
        # Regression，回归
        net = tflearn.fully_connected(net, 64, activation='softmax') #10表示number of units for this layer

    return net

def neg_pearson(m,n): #皮尔森相关性系数是协方差与标准差的比值
    loss=0
    for i in range(m.shape[0]):
        a = m[i, :]
        b = n[i, :]
        sum_x = tf.reduce_sum(a)  # x
        sum_y = tf.reduce_sum(b)  # y
        sum_xy = tf.reduce_sum(tf.multiply(a, b))  # xy
        sum_x2 = tf.reduce_sum(tf.multiply(a, a))  # x^2
        sum_y2 = tf.reduce_sum(tf.multiply(b, b))  # y^2
        N = m.shape[1]
        N = tf.to_float(N)  # 不加这一步，整个运算就会出错，tensorflow要求同一类型数据进行运算
        q = tf.abs((N * sum_x2 - sum_x * sum_x) * (N * sum_y2 - sum_y * sum_y))
        pearson1 = (N * sum_xy - sum_x * sum_y) / (tf.rsqrt(q) + 0.01)
        loss += 1 - pearson1
    loss = loss / tf.to_float(m.shape[0])
    return loss

#load_data
dataset1,images,labels=y.get_batch_data()
X=tf.placeholder(shape=(images.shape[0], 64,128,128,3), dtype=tf.float32)
Y=tf.placeholder(shape=(images.shape[0],64), dtype=tf.float32)
dataset=tf.data.Dataset.from_tensor_slices((X,Y))
dataset=dataset.shuffle(20).batch(3).repeat()
iterator = dataset.make_initializable_iterator()
data_element = iterator.get_next()

Y2=tf.placeholder(shape=(3,64), dtype=tf.float32)
Z=tf.placeholder(shape=(3, 64,128,128,3), dtype=tf.float32)
y_pred = deep_net(Z)
train_loss = neg_pearson(y_pred, Y2)

optimizer = tf.train.AdamOptimizer(1e-4).minimize(train_loss)


avg_cost=0
epochs=3
num_batches=3

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    print('2:', tf.test.is_gpu_available())
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={X: images.eval(), Y: labels.eval()})
    for epoch in range(epochs):
        print ("EPOCH = ", epoch+1)
        for i in range(num_batches):
            batch_xs, batch_ys = sess.run(data_element) #第一个维度由dataset中的batch决定
            feed_dict={Z: batch_xs, Y2: batch_ys}
            sess.run(optimizer,feed_dict=feed_dict)
            cost = sess.run(train_loss,feed_dict=feed_dict)
            avg_cost += cost / num_batches  # total_batch个循环中的平均值
            print('cost:', cost)
        loss = avg_cost / num_batches
        print('loss:', loss)
