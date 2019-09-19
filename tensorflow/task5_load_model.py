import tensorflow as tf
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

# 导入数据
iris = load_iris()
# 数据划分
x_train, x_test, y_train, y_test = \
    train_test_split(iris['data'], iris['target'], test_size=0.2)
ckpt_path = './tmp'
sess = tf.Session()
# 导入计算图
saver = tf.train.import_meta_graph(
    os.path.join(ckpt_path, 'model-train.meta')  # 计算图
)
# 加载数据
saver.restore(sess,
              os.path.join(ckpt_path, 'model-train'))
# 加载所需节点
out = sess.graph.get_tensor_by_name('predict_prob:0')
acc = sess.graph.get_tensor_by_name('accuracy:0')
# 执行计算
y_predict, score = sess.run([out, acc], feed_dict={
    'input:0': x_test, 'output:0': y_test
})
print(y_predict, score)
sess.close()  # 关闭会话
