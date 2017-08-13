import tensorflow as tf

with tf.device('/cpu:0'):
    matrix = tf.constant([[1., 2.],[2.,4.]])
    vector = tf.constant([1.,1.])
product = tf.mul(vector,matrix)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess.run(product))
sess.close()
