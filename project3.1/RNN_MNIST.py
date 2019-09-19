# %% 文件读取
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)

# %% 设置基本参数
learning_rate = 0.001
input_size = 28
hidden_size = 60
output_size = 10
sequence_length = 28

# %% 计算图搭建
tf.reset_default_graph()
x_data = tf.placeholder(tf.float32, shape=[None, 784])
x = tf.reshape(x_data, shape=[-1, sequence_length, input_size])
y_data = tf.placeholder(tf.float32, shape=[None, 10])
# 定义RNN单元
# rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)

#   RNN单元放之到动态RNN单元中
# hidden_out, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
hidden_out, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
# hidden_out, states = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell2, x, dtype=tf.float32)

w = tf.Variable(tf.truncated_normal([hidden_size, output_size], dtype=tf.float32))
b = tf.Variable(tf.zeros(output_size, tf.float32))
y_ = tf.matmul(hidden_out[:, -1, :], w) + b
y = tf.nn.softmax(y_)

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y_data)
)
acc = tf.reduce_mean(tf.cast(
    tf.equal(tf.argmax(y, axis=1), tf.argmax(y_data, axis=1)),
    tf.float32
))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# %% 执行训练图
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(500):
    batch_x, batch_y = mnist.train.next_batch(100)
    _, accuary, cast, y_pre_prob, y_pre = sess.run(
        [train_op, acc, loss, y, tf.argmax(y, axis=1)],
        feed_dict={x_data: batch_x, y_data: batch_y}
    )
    if i % 10 == 0:
        print(f'i: {i}accuary: {accuary}, cast: {cast}')

# sess.close()
