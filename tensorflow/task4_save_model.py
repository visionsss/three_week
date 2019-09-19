from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

# 导入数据
iris = load_iris()
# 数据划分
x_train, x_test, y_train, y_test = \
    train_test_split(iris['data'], iris['target'], test_size=0.2)
# tensor与参数设置
input_size = 4  # 输入层神经元个数
output_size = 3  # 输出层神经元个数
hidden_size = 8  # 隐层神经元个数(自行设置)
lr = 0.01  # 学习率

tf.reset_default_graph()
# 占位符：输入、输出
x = tf.placeholder(tf.float32, [None, input_size], name='input')
y = tf.placeholder(tf.int32, [None, ], name='output')
y_real = tf.one_hot(y, depth=output_size, dtype=tf.float32)

# 权重矩阵与阈值(变量)
# input layer --> hidden layer
v = tf.Variable(tf.truncated_normal(
    dtype=tf.float32, shape=[input_size, hidden_size]
))  # 权重矩阵
b1 = tf.Variable(tf.zeros(
    shape=(hidden_size,), dtype=tf.float32
))  # 阈值
# hidden layer --> output layer
w = tf.Variable(tf.truncated_normal(
    dtype=tf.float32, shape=[hidden_size, output_size]
))  # 权重矩阵
b2 = tf.Variable(tf.zeros(
    shape=(output_size,), dtype=tf.float32
))  # 阈值

# 构建前向传播计算图
# input layer --> hidden layer
hidden_output = tf.nn.sigmoid(tf.matmul(x, v) - b1)
# hidden layer --> output layer
out = tf.nn.softmax(tf.matmul(hidden_output, w) - b2,
                    name='predict_prob')

# 更新
loss = tf.reduce_mean(
    -tf.reduce_sum(y_real * tf.log(out), axis=1)
)  # 损失函数：交叉熵
opt = tf.train.GradientDescentOptimizer(lr)
train_op = opt.minimize(loss)

# 计算正确率
y_pre = tf.argmax(out, axis=1, output_type=tf.int32)
acc = tf.reduce_mean(
    tf.cast(tf.equal(y, y_pre), tf.float32),
    name='accuracy'
)

saver = tf.train.Saver()
ckpt_path = './tmp'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
# 在会话中执行训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(5000):
    _, score, cost = sess.run([train_op, acc, loss],
                              feed_dict={
                                  x: x_train, y: y_train
                              })
    if i % 100 == 0:
        print(f"第{i}次 损失值为{cost} 训练正确率为{score}")
saver.save(sess, os.path.join(ckpt_path, 'model-train'))
sess.close()
