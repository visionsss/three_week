import tensorflow as tf

string = tf.constant('hello world')
sess = tf.Session()
print(sess.run(string))
sess.close()
