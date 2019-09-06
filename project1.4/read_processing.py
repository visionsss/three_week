import pandas as pd
import matplotlib.pyplot as plt
import re
import jieba


def read_data():
    file_path = './data/comment.csv'
    data = pd.read_csv(file_path)
    return data


def draw1(num):
    """
    :param num: data['品牌'].value_counts(),<class 'pandas.core.series.Series'>
    :return:
    """
    plt.figure(figsize=(4, 4))  # 设置size
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 显示中文
    plt.pie(num, autopct='%.2f', labels=num.index, explode=[0.1] + [0] * 5)  # 数据，占百分比，标签， 突出显示，第一个0.1，后面不分离
    plt.show()


def draw2(num):
    num.sort_index(inplace=True)
    # print(num)
    plt.plot(num.index, num)
    plt.xticks(num.index)
    plt.show()


def processing(data):
    # 删除评论换行，以及产品类型，&[a-z]+，文字重复,评论重复
    # 提取正常样本
    data = data.loc[data['已采'] != False]
    # 提取AO热水器
    data = data.loc[data['品牌'] == 'AO']
    # 删除换行符号
    data['评论'] = data['评论'].astype(str).apply(lambda x: re.sub(r'\\n', '', x))
    # 删除产品类型、
    # print(data['型号'].value_counts()) # 查看产品型号的不同
    data['评论'] = data['评论'].astype(str).apply(lambda x:
                                              re.sub('AO史密斯（A.O.Smith） ET[0-9]{3}J-[0-9]{2} 电热水器 [0-9]{2}升', '', x))
    # 删除&[a-z]+
    data['评论'] = data['评论'].astype(str).apply(lambda x:
                                              re.sub('&[a-z]+', '', x))

    # print(data['评论'].apply(len).sum())
    # 文字重复
    data['评论'] = data['评论'].astype(str).apply(del_com)
    # print(data['评论'].apply(len).sum())
    # 评论重复
    # print(data.shape)
    data.drop_duplicates(inplace=True, subset='评论')
    # print(data.shape)
    # 去除空值
    data = data[data['评论'] != 'nan']
    # print(data.shape)
    # print(data.shape)
    # 剔除平台提示文字
    ind = data['评论'].astype(str).apply(lambda x: re.findall('[0-9]+[^,]{,5}字', x) != [])
    data = data[-ind]

    return data


def del_com(string='今天天气天气天气天气很好哈哈哈哈哈哈哈哈哈'):
    for i in [1, 2]:
        for j in range(len(string)):
            if string[j: j + i] == string[j + i: j + 2 * i] and string[j + i: j + 2 * i] == string[
                                                                                            j + 2 * i: j + 3 * i]:
                k = j + 2 * i
                while string[k: k + i] == string[k + i: k + 2 * i] and k < len(string):
                    k += i
                string = string[: j] + string[k:]
    # print(string)
    for i in range(3, int(len(string) / 2) + 1):
        for j in range(len(string)):
            if string[j: j + i] == string[j + i: j + 2 * i]:
                k = j + 2 * i
                while string[k: k + i] == string[k + i: k + 2 * i] and k < len(string):
                    k += i
                string = string[: j] + string[k:]
    return string


def div_word(data):
    jieba.load_userdict('./data/dic1.txt')
    comment = data['评论'].astype(str).apply(jieba.lcut)
    with open('./data/stoplist.txt', 'r', encoding='utf-8') as f:
        stop = f.read()
    stop = stop.split()
    stop = [' ', '\n', '\t', '\r'] + stop
    comment = comment.apply((lambda x: [i for i in x if i not in stop]))
    print(comment.head())

    from tkinter import _flatten
    data_after = pd.Series(_flatten(list(comment))).value_counts()
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    pic = plt.imread('./data/aixin.jpg')
    wc = WordCloud(
        mask=pic, font_path='./data/simhei.ttf', background_color='white', max_words=100
    )
    wc.fit_words(data_after)
    plt.imshow(wc)
    plt.show()


if __name__ == '__main__':
    data = read_data()
    # draw1(data['品牌'].value_counts())
    # draw2(pd.to_datetime(data['时间']).dt.month.value_counts())
    # 查看那个容量比较多人买
    # tmp = data['型号'].astype(str).apply(lambda x: re.findall('([0-9]{2})[升L]', x))
    # tmp = tmp.loc[tmp.apply(lambda x: x != [])].str[0].value_counts()
    # print(tmp)
    data = processing(data)
    div_word(data)
