import pandas as pd
import jieba
import os
import re
from glob import glob


def get_all_data():
    # all_data = pd.DataFrame()
    # for i in glob(r'../data/tmp/*'):
    #     data = pd.read_csv(i,encoding='gbk')
    #     all_data = pd.concat([all_data, data])
    all_data = pd.read_excel(r'./temp.xlsx')
    return all_data


if __name__ == '__main__':
    all_data = get_all_data()
    print(all_data.shape)
    all_data['短评'] = all_data['短评'].apply(lambda x: re.sub(r'\n', '', x))
    jieba.load_userdict('../data/dict.txt')
    a = all_data['短评'].apply(jieba.lcut)
    with open('../data/stoplist.txt', 'r', encoding='utf-8') as f:
        stop = f.read()
    stop = stop.split()
    stop = [' ', '\n', '\t', '\r'] + stop
    a = a.apply((lambda x: [i for i in x if i not in stop]))
    from tkinter import _flatten

    data_after = pd.Series(_flatten(list(a))).value_counts()
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    pic = plt.imread('../data/aixin.jpg')
    wc = WordCloud(
        mask=pic, font_path='../data/simhei.ttf', background_color='white'
    )
    wc.fit_words(data_after)
    plt.imshow(wc)
    plt.show()
