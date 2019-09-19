构建数据存储位置：
checkpoints：Seq2Seq模型运行时保存的模型参数保存路径
dialog：语料库储存路径
fromids：问题的id向量储存路径
toids：回答的id向量储存路径
tmp: 临时文件保存路径
logs: 记录图的日志的保存路径（TensorBroad）

main.py:主函数，直接运行
process.py——语料库预处理
train.py —— 模型训练
test.py —— 模型测试
main.py —— 主程序

model.py —— 模型计算图
myfun.py —— 自定义函数汇总