# 读取文件
import pandas as pd

lines = pd.read_csv('./data/lines.csv',
                    index_col=0)
# 构造一个线性模型 y = a*x1 + b*x2 + c
import tensorflow as tf

tf.reset_default_graph()  # 重设计算图
y = tf.placeholder(tf.float32)
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
a = tf.Variable(initial_value=[0.])
b = tf.Variable([0.])
c = tf.Variable([0.])
y_pre = a * x1 + b * x2 + c

# 最小化方差
loss = (y - y_pre) ** 2
opts = tf.train.GradientDescentOptimizer(
    learning_rate=0.001
)
train_op = opts.minimize(loss)

# 初始化变量
sess = tf.Session()  # 启动图
sess.run(tf.global_variables_initializer())
# 拟合平面(开始训练)
for i in range(5000):
    _, cost = sess.run([train_op, loss],
                       feed_dict={
                           x1: lines.loc[i % 100, 'X_1'],
                           x2: lines.loc[i % 100, 'X_2'],
                           y: lines.loc[i % 100, 'Y']
                       })
    if i % 200 == 0:
        print(cost)
print(sess.run([a, b, c]), cost)
sess.close()
