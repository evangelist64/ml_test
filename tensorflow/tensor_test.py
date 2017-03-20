import tensorflow as tf

with tf.device('/cpu:0'):
    matrix1 = tf.constant([[3., 3.],[2.,1.]])
    matrix2 = tf.constant([[2.,2.],[2.,2.]])
product = tf.matmul(matrix1, matrix2)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess.run(product))
sess.close()
