import pandas as pd
import jieba
import os
import re
from glob import glob
import matplotlib.pyplot as plt


def get_all_data():
    all_data = pd.DataFrame()
    for i in glob(r'../data/tmp/*'):
        data = pd.read_csv(i, encoding='gbk')
        all_data = pd.concat([all_data, data])
    return all_data


if __name__ == '__main__':
    all_data = get_all_data()
    num = pd.to_datetime(all_data['times'])
    print(num)
    print(pd.to_datetime('2014/3/31') - num[0])
