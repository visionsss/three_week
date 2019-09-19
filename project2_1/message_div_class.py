import pandas as pd
import re
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tkinter import _flatten


def read_csv():
    data = pd.read_csv('./data/message80.csv', index_col=0, header=None)
    # 查看垃圾短信与正常短信数量
    # a = data.iloc[:, 0].value_counts()
    # print(a)

    # 抽取20000条短信
    ind = data.iloc[:, 0] == 0
    n = 10000
    normal = data.loc[-ind].sample(n, random_state=123)
    abnormal = data.loc[ind].sample(n, random_state=123)
    data2w = pd.concat([normal, abnormal], axis=0)
    data2w.columns = ['labels', 'content']
    return data2w


def process(data2w):
    # 去除空格
    data2w['content'] = data2w['content'].apply(
        lambda x: re.sub('\u3000| ', '', x)
    )
    # 重新脱敏
    data2w['content'] = data2w['content'].apply(
        lambda x: re.sub('[0-9]', 'x', x)
    )
    ind = data2w['content'].apply(lambda x: re.findall(
        '银行|工行|农行|邮政|中行|建行', x
    ) != [])
    pattern = re.compile('x{16,}|x{3,}[^\u4E00-\u9FD5A-Za-wyz]{10,}x{3,}')
    ind = data2w['content'].apply(lambda x: pattern.findall(x) != [])
    a = data2w.loc[ind, 'content']
    data2w['content'] = data2w['content'].apply(
        lambda x: pattern.sub('C账号C', x)
    )
    # 电话号码
    pattern = re.compile(
        'x{11,}|x{7,8}|x{3,4}.?x{6,}|x{3,}[^\u4E00-\u9FD5A-Za-wyz]+x{3,}'
    )
    data2w['content'] = data2w['content'].apply(
        lambda x: pattern.sub('T联系T', x)
    )
    # 价格
    pattern = re.compile(
        'x[x.]+[元块钱万]|[满每买]?x{1,}[送减反返]{1,2}x{1,}'
    )
    ind = data2w['content'].apply(lambda x: pattern.findall(x) != [])
    data2w.loc[ind, 'content']
    data2w['content'] = data2w['content'].apply(lambda x: pattern.sub('P价格P', x))
    # 日期
    pattern = re.compile(
        'x{2,}年x{1,2}月x{1,2}[天日号]|x{1,2}月x{1,2}[天日号]|x{1,2}[天日号]'
    )
    ind = data2w['content'].apply(lambda x: pattern.findall(x) != [])
    data2w.loc[ind, 'content']
    data2w['content'] = data2w['content'].apply(lambda x: pattern.sub('D日期D', x))
    # 删除剩下没有被提取出来的x
    data2w['content'] = data2w['content'].apply(lambda x: re.sub('x', '', x))
    data2w.drop_duplicates(subset='content', inplace=True, keep='first')

    return data2w


def draw_wc(data_after):
    num = pd.Series(_flatten(list(data_after))).value_counts()  # 统计词频
    pic = plt.imread('./data/aixin.jpg')  # 读取背景图片
    wc = WordCloud(mask=pic, font_path='C:/Windows/Fonts/simkai.ttf',
                   background_color='white')
    wc.fit_words(num)  # 传入词频
    plt.imshow(wc)
    plt.axis('off')
    #     plt.show()
    return num


def draw(data2w):
    for i in ['C账号C', 'T联系T', 'P价格P', 'D日期D']:
        jieba.add_word(i)
    jieba.load_userdict('./data/newdic1.txt')
    data_cut = data2w['content'].apply(jieba.lcut)
    with open('./data/stoplist.txt', 'r', encoding='utf-8') as f:
        stop = f.read()
    stop = stop.split()
    data_after = data_cut.apply(lambda x: [i for i in x if i not in stop])

    # 查看词云
    ind = data2w['labels'] == 0
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    a = draw_wc(data_after[ind])
    plt.title('非垃圾短信')
    plt.subplot(1, 2, 2)
    b = draw_wc(data_after[-ind])
    plt.title('垃圾短信')
    plt.show()
    return data_after


if __name__ == '__main__':
    data = read_csv()

    data = process(data)
    data_after = draw(data)
    data2w = data
    # 文本向量化
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    x = data_after.apply(lambda x: ' '.join(x))
    x_train, x_test, y_train, y_test = \
        train_test_split(x, data2w['labels'], test_size=0.2, random_state=123)
    tfidf = TfidfVectorizer().fit(x_train)
    x_train_tv = tfidf.transform(x_train)
    x_test_tv = tfidf.transform(x_test)
    from sklearn.svm import LinearSVC, SVC

    model_svm = LinearSVC().fit(x_train_tv, y_train)
    s = model_svm.score(x_test_tv, y_test)
    from sklearn.metrics import confusion_matrix, classification_report, \
        recall_score, precision_score

    y_pre = model_svm.predict(x_test_tv)
    print(classification_report(y_test, y_pre))
    print(recall_score(y_test, y_pre))
    print(precision_score(y_test, y_pre))
    confusion_matrix(y_true=y_test, y_pred=y_pre)
