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

def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        #x=tf.clip_by_value(x,-1,1)
        with g.gradient_override_map({"Sign": "Identity"}):
            return tf.sign(x)

def batch_norm_layer(value,is_training=True,name='batch_norm'):
    '''
    批量归一化  返回批量归一化的结果
    
    args:
        value:代表输入，第一个维度为batch_size
        is_training:当它为True，代表是训练过程，这时会不断更新样本集的均值与方差。当测试时，要设置成False，这样就会使用训练样本集的均值和方差。
              默认测试模式
        name：名称。
    '''
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
conv1_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 32], mean=0, stddev=1.0))
#conv1_weights = tf.get_variable('conv1_weights', [3, 3, 1, 16],
#                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
#conv1_weights = tf.clip_by_value(conv1_weights,-1,1)
#conv1_weights=binarize(conv1_weights)
conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding="VALID")
bn1 = batch_norm_layer(conv1,is_training=True)
#bn1=conv1*0.25
#conv1=conv1*0.5
pool1 = tf.nn.max_pool(bn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
relu1 = tf.clip_by_value(pool1,-1,1)

#第二层：最大池化层

#第三层：卷积层
conv2_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=1.0) / 27.0)
#conv2_weights = tf.get_variable('conv2_weights', [3, 3, 16, 32],
#                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
#conv2_weights = tf.clip_by_value(conv2_weights,-1,1)
#aaaaa=binarize(conv2_weights)
binarize2=binarize(relu1)
conv2 = tf.nn.conv2d(binarize2, conv2_weights, strides=[1, 1, 1, 1], padding="VALID")
bn2 = batch_norm_layer(conv2,is_training=True)
#bn2=bn2*0.25
#conv2=conv2*0.5
pool2 = tf.nn.max_pool(bn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
relu2 = tf.clip_by_value(pool2,-1,1)

#第四层：最大池化层
pool2_dropout = tf.nn.dropout(relu2, keep_prop)

#第五层：全连接层
fc1_weights = tf.Variable(tf.truncated_normal(shape=[6 * 6 * 64, 128], mean=0, stddev=2.0) / 120.0)
#fc1_weights = tf.get_variable('fc1_weights', [7*7*32, 512],
#                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
#fc1_weights = tf.clip_by_value(fc1_weights,-1,1)
#bbbbb=binarize(fc1_weights)
pool2_vector = tf.reshape(pool2_dropout, shape=[-1, 6 * 6 * 64])
binarize3=binarize(pool2_vector)
fc1 = tf.matmul(binarize3, fc1_weights)  
bn3 = batch_norm_layer(fc1,is_training=True)
#bn3=bn3*0.25
#fc1=fc1*0.5
fc11 = tf.clip_by_value(bn3,-1,1)
#为了减少过拟合，加入Dropout层
fc11_dropout = tf.nn.dropout(fc11, keep_prop)

#第六层：全连接层
fc2_weights = tf.get_variable('fc2_weights', [128, 10],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
#fc2_weights = tf.clip_by_value(fc2_weights,-1,1)
#ccccc=binarize(fc2_weights)
binarize4=binarize(fc11_dropout)
fc2 = tf.matmul(binarize4, fc2_weights)

#第七层：输出层
y_conv = tf.nn.softmax(fc2)

#定义损失函数
learning_rate = 0.5
learning_rate1 = 0.5
#输出交叉熵损失
loss_label = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
#权值正则化
th1 = 0.15
th2 = 1   
loss2 = tf.reduce_sum(tf.where(tf.greater(conv1_weights, th1), tf.square(conv1_weights - th2),
                               tf.where(tf.greater(conv1_weights, -th1),
                                        0.0 * conv1_weights, tf.square(conv1_weights + th2))))
loss3 = tf.reduce_sum(tf.where(tf.greater(conv2_weights, th1), tf.square(conv2_weights - th2),
                               tf.where(tf.greater(conv2_weights, -th1),
                                        0.0 * conv2_weights, tf.square(conv2_weights + th2))))
loss4 = tf.reduce_sum(tf.where(tf.greater(fc1_weights, th1), tf.square(fc1_weights - th2),
                               tf.where(tf.greater(fc1_weights, -th1),
                                        0.0 * fc1_weights, tf.square(fc1_weights + th2))))
loss5 = tf.reduce_sum(tf.where(tf.greater(fc2_weights, th1), tf.square(fc2_weights - th2),
                               tf.where(tf.greater(fc2_weights, -th1),
                                        0.0 * fc2_weights, tf.square(fc2_weights + th2))))
loss_w = loss2 + loss3 + loss4 + loss5
#loss_w = loss3 + loss4 

loss21 = tf.reduce_sum(tf.where(tf.greater(conv1_weights, th1), tf.square(conv1_weights - th2),
                               tf.where(tf.greater(conv1_weights, -th1),
                                        tf.square(conv1_weights), tf.square(conv1_weights + th2))))
loss31 = tf.reduce_sum(tf.where(tf.greater(conv2_weights, th1), tf.square(conv2_weights - th2),
                               tf.where(tf.greater(conv2_weights, -th1),
                                        tf.square(conv2_weights), tf.square(conv2_weights + th2))))
loss41 = tf.reduce_sum(tf.where(tf.greater(fc1_weights, th1), tf.square(fc1_weights - th2),
                               tf.where(tf.greater(fc1_weights, -th1),
                                        tf.square(fc1_weights), tf.square(fc1_weights + th2))))
loss51 = tf.reduce_sum(tf.where(tf.greater(fc2_weights, th1), tf.square(fc2_weights - th2),
                               tf.where(tf.greater(fc2_weights, -th1),
                                        tf.square(fc2_weights), tf.square(fc2_weights + th2))))
loss_w1 = loss21 + loss31 + loss41 + loss51
#loss_w1 = loss31 + loss41 

#优化
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_label)
train_step_w = tf.train.GradientDescentOptimizer(learning_rate1).minimize(loss_w)

train_step_w1 = tf.train.GradientDescentOptimizer(0.5).minimize(loss_w1)


init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(fc2, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

max_accuracy=0
with tf.Session() as sess:
    sess.run(init)
#    tf.train.start_queue_runners()
    for epoch in range(5000):
        if (epoch + 1) % 100 == 0:
            learning_rate = learning_rate / 2

        #if((epoch + 1) % 10) == 0:
        #    learning_rate1 = learning_rate1 + 0.1

        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = np.round(batch_xs)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prop: 1.0, is_training:False})
            if batch % 30 == 0:
                sess.run(train_step_w)
        if epoch>=4999:
            sess.run(train_step_w1)
        #else:
        #    sess.run(train_step_w)

        mnist.test.images1 = np.round(mnist.test.images)    
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images1, y: mnist.test.labels, keep_prop: 1.0, is_training:False})
        #if epoch>1000 and acc>0.9955:
        #    sess.run(train_step_w1)
        #    acc = sess.run(accuracy, feed_dict={x: mnist.test.images1, y: mnist.test.labels, keep_prop: 1.0, is_training:False})
        #    print("after " + str(epoch) + " test accuracy: " + str(acc))
        #    break
        if acc>max_accuracy :
            max_accuracy=acc    
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
