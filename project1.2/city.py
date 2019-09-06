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
    citys = all_data['常居'].value_counts()
    provices = []
    for i in all_data['常居']:
        try:
            if len(i) > 2:
                if re.findall('[a-z]', i) == []:
                    provices.append(i[2:4])
                else:
                    provices.append(i.split(', ')[-1])
            else:
                provices.append(i)
        except:
            pass
    provices = pd.DataFrame(provices)
    a = provices[0].value_counts()
    print(a)
    plt.rcParams['font.sans-serif'] = 'Simhei'
    plt.bar(range(10), a[:10])
    plt.xticks(range(10), a.index[:10], rotation=45)
    plt.show()
