import tensorflow as tf

# 1 定义计算图
a2 = tf.constant([2, 3])
b2 = tf.constant([4, 5])
my_sum = a2 + b2  # Tensor
# 2 定义执行计算图：会话
sess = tf.Session()
print(sess.run(my_sum))
sess.close()
