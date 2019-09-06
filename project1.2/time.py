import pandas as pd
import jieba
import os
import re
from glob import glob
import matplotlib.pyplot as plt


def get_all_data():
    # all_data = pd.DataFrame()
    # for i in glob(r'../data/tmp/*'):
    #     data = pd.read_csv(i,encoding='gbk')
    #     all_data = pd.concat([all_data, data])
    all_data = pd.read_excel(r'./temp.xlsx')
    return all_data


if __name__ == '__main__':
    all_data = get_all_data()
    all_data['评论时间'] = pd.to_datetime(all_data['评论时间'])
    # print(all_data['评论时间'])
    all_data.sort_values(by='评论时间', inplace=True)
    all_data['评论时间'] = all_data['评论时间'].dt.date
    num = all_data['评论时间'].value_counts().sort_index()  # 统计数量并排序
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 字体
    plt.style.use('ggplot')  # 绘图风格
    plt.plot(range(len(num)), num)  # 绘制折线图
    plt.title('评论数量随日期变化情况')  # 设置标题
    plt.ylabel('数量')  # Y轴名称
    plt.xlabel('日期')  # X轴名称
    plt.xticks(range(len(num)), num.index, rotation=90)  # 坐标须
    plt.show()  # 图表展示
    num = pd.crosstab(all_data['评论时间'], all_data['评分'])  # 交叉表
    num.sort_index(inplace=True)  # 按照index进行排序
    print(num)
