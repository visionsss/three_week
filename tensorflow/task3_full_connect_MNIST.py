from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets(
    './data/MNIST_data/', one_hot=True
)
tf.reset_default_graph()  # 重设计算图
# Tensor与参数
input_size = 784  # 输入层神经元个数
output_size = 10  # 输出层神经元个数
learing_rate = 0.01  # 学习率
batch_size = 200  # 批处理样本数量
x = tf.placeholder(
    dtype=tf.float32, shape=[None, input_size]
)  # 模型输入
y = tf.placeholder(
    dtype=tf.float32, shape=[None, output_size]
)  # 模型输出
W = tf.Variable(tf.truncated_normal(
    [input_size, output_size], dtype=tf.float32
))  # 权重矩阵
b = tf.Variable(tf.truncated_normal(
    [output_size, ], dtype=tf.float32
))  # 阈值、偏置项

# softmax层前向传播
y_pre = tf.nn.softmax(tf.matmul(x, W) - b)
# loss：交叉熵
loss = tf.reduce_mean(
    -tf.reduce_sum(y * tf.log(y_pre), axis=1)
)
# 优化器
opts = tf.train.GradientDescentOptimizer(
    learing_rate
)
# 目标
train_op = opts.minimize(loss)
# 计算正确率
ind = tf.equal(tf.argmax(y, axis=1),
               tf.argmax(y_pre, axis=1))
acc = tf.reduce_mean(tf.cast(ind, tf.float32))
# 执行计算图：训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(5000):
    train = mnist.train.next_batch(batch_size)
    _, cost, acc2 = sess.run([train_op, loss, acc],
                             feed_dict={
                                 x: train[0],
                                 y: train[1]
                             })
    if i % 200 == 0:
        print(f'''第{i}次，
        损失值为{cost:.2f},
        正确率为{acc2:.2f}''')
# 测试
y_, score = sess.run([y_pre, acc], feed_dict={
    x: mnist.test.images, y: mnist.test.labels
})
sess.close()
y_predict = np.argmax(y_, axis=1)
y_real = np.argmax(mnist.test.labels, axis=1)
np.mean(y_real == y_predict)
