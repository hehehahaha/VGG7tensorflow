import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#import os
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 64
n_batch = mnist.train.num_examples // batch_size
keep_prop = tf.placeholder(tf.float32)

def batch_norm_layer(value,is_training=True,name='batch_norm'):

    if is_training is True:
        #训练模式 使用指数加权函数不断更新均值和方差
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = True)
    else:
        #测试模式 不更新均值和方差，直接使用
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = False)

is_training = tf.placeholder(dtype=tf.bool)                       #设置为True，表示训练 Flase表示测试

#训练数据
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
#训练数据的标签
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
#把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽，第4维代表通道数
x_image = tf.reshape(x, shape=[-1, 28, 28, 1])

#第一层：卷积层
conv1_weights = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 16], mean=0, stddev=1.0))
conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding="VALID")
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
bn1 = batch_norm_layer(pool1,is_training=True)
relu1 = tf.nn.relu(bn1)

conv2_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], mean=0, stddev=1.0))
conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding="VALID")
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
bn2 = batch_norm_layer(pool2,is_training=True)
relu2 = tf.nn.relu(bn2)

#第五层：全连接层
fc1_weights = tf.Variable(tf.truncated_normal(shape=[5 * 5 * 32, 1024], mean=0, stddev=1.0))
pool2_vector = tf.reshape(relu2, shape=[-1, 5 * 5 * 32])
#binarize3=binarize(pool2_vector)
fc1 = tf.matmul(pool2_vector, fc1_weights)  
bn3 = batch_norm_layer(fc1,is_training=True)
relu3 = tf.nn.relu(bn3)
#第六层：全连接层
fc2_weights = tf.get_variable('fc2_weights', [1024, 10],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
fc2 = tf.matmul(relu3, fc2_weights)

#第七层：输出层
y_conv = tf.nn.softmax(fc2)

#定义损失函数
learning_rate = 0.5

#输出交叉熵损失
loss_label = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
#权值正则化

#优化
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_label)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_label)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(fc2, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(init)
#    tf.train.start_queue_runners()
    for epoch in range(1000):

        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = np.round(batch_xs)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, is_training:True})

        mnist.test.images1 = np.round(mnist.test.images)    
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images1, y: mnist.test.labels, is_training:False})
        print("after " + str(epoch) + " test accuracy: " + str(acc))
#    conv1_weights_save=sess.run(conv1_weights)
#    conv2_weights_save=sess.run(conv2_weights)
#    fc1_weights_save=sess.run(fc1_weights)
#    fc2_weights_save=sess.run(fc2_weights)
#
#print(max_accuracy)      
#np.save("conv1_weights.npy",conv1_weights_save)
#np.save("conv2_weights.npy",conv2_weights_save)    
#np.save("fc1_weights.npy",fc1_weights_save)    
#np.save("fc2_weights.npy",fc2_weights_save)
