import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

D_input = 2
D_label = 1
D_hidden = 1
lr = 1e-4

X = tf.placeholder(tf.float32,[None,D_input],name="X")
Y = tf.placeholder(tf.float32,[None,D_label],name="Y")

w1 = tf.Variable(tf.truncated_normal([D_input,D_hidden],stddev=0.1),name="w1")
b1 = tf.Variable(tf.constant(0.1,shape=[D_hidden]),name="b1")
pre_act1 = tf.matmul(X,w1)+b1
f1 = tf.nn.relu(pre_act1,name='f1')

w2 = tf.Variable(tf.truncated_normal([D_hidden, D_label], stddev=0.1),name="w2")
b2 = tf.Variable(tf.constant(0.1, shape=[D_label]), name="b2")
pre_act2 = tf.matmul(f1, w2) + b2
f2 = tf.nn.relu(pre_act2, name='f2')

loss = tf.reduce_sum(tf.pow(pre_act2-Y, 2))
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

train_X=[[0,0],[0,1],[1,0],[1,1]]
train_Y=[[0],[1],[1],[0]]
train_X=np.array(train_X).astype('float')
train_Y=np.array(train_Y).astype('float')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    T=10000
    for i in range(T):
        sess.run(train_step,feed_dict={X:train_X,Y:train_Y})
    print("w1:",sess.run(w1),",b1:",sess.run(b1),",w2:",sess.run(w2),",b2:",sess.run(b2))
    
