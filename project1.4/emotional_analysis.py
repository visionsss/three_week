# -*- coding: utf-8 -*-
from project4.read_processing import *
import gensim


def analysis(data):
    # 读取词的信息
    feeling = pd.read_csv('./data/BosonNLP_sentiment_score.txt', sep=' ', header=None)
    feeling.columns = ['word', 'score']
    degree = pd.read_csv('./data/degree.csv')
    degree['score'] = -degree['score'] / 100
    not_word = pd.read_csv('./data/not.csv')

    # 停用词表需不包含上面3个表的信息
    w = list(feeling) + list(degree) + list(not_word)
    jieba.load_userdict('./data/dic1.txt')
    comment = data['评论'].astype(str).apply(jieba.lcut)
    with open('./data/stoplist.txt', 'r', encoding='utf-8') as f:
        stop = f.read()
    stop = stop.split()
    stop = [' ', '\n', '\t', '\r'] + stop
    stop = [i for i in stop if i not in w]
    # comment = comment.apply((lambda x: [i for i in x if i not in stop]))
    a = comment[14] + ['不是', '不', '好']
    print(a)
    tmp = pd.merge(pd.DataFrame(a), feeling, how='left', left_on=0, right_on='word')
    tmp = pd.merge(tmp, degree, how='left', left_on=0, right_on='term')
    tmp = pd.merge(tmp, not_word, how='left', left_on=0, right_on='term')
    ind = tmp.index[tmp['score_y'].notnull()]
    for i in ind:
        if i != len(tmp) - 1:
            tmp.loc[i + 1, 'score_x'] = \
                tmp.loc[i + 1, 'score_x'] * tmp.loc[i, 'score_y']
            tmp.loc[i, 'score_x'] = 0
    tmp3 = pd.merge(tmp, not_word, how='left', left_on=0, right_on='term')
    ind = tmp3.index[tmp3['term'].notnull()]
    k = 0
    while k < len(ind):
        # 否定、否定词在句末、双重否定
        if ind[k] != len(tmp3) - 1:
            if ind[k + 1] == ind[k] + 1:  # 双重否定
                tmp3.loc[ind[k]:ind[k] + 1, 'score_x'] = 0
                k += 2
            else:
                tmp3.loc[ind[k] + 1, 'score_x'] = -tmp3.loc[ind[k] + 1, 'score_x']
                tmp3.loc[ind[k], 'score_x'] = 0
                k += 1
        else:  # 否定词在句末
            tmp3.loc[ind[k], 'score_x'] = 0
            k += 1
    a = tmp3['score_x'].sum()
    print(tmp3)
    print(a)


if __name__ == '__main__':
    data = read_data()
    data = processing(data)
    analysis(data)
