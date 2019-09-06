# coding=utf8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import Lasso
from project2_2.GM11 import GM11
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor


def read_csv():
    data = pd.read_csv('./data/data.csv')
    data.index = range(1994, 2014)

    num = data.corr(method='pearson')  # 判断相关性
    # seaborn.heatmap(num, annot=True)    # 画热力图
    # plt.show()
    return data


def draw1(data):
    plt.figure(figsize=(10, 10))
    for i in range(14):
        plt.subplot(3, 5, i + 1)
        plt.plot(data.index, data.iloc[:, i])
    plt.show()


def LR(data):
    losso = Lasso(alpha=1000, max_iter=50000)  # 设置惩罚因子,最大训练次数
    losso.fit(data.iloc[:, :-1], data['y'])
    y_ = losso.predict(data.iloc[:, :-1])
    # print(data.iloc[:, -1])
    # print(y_)
    # print(f'a is {losso.coef_}')
    cols = data.iloc[:, losso.coef_ != 0]
    cols = pd.concat([cols, data['y']], axis=1)
    # print(cols)
    return cols


def predict_x(data_new):
    cols = data_new.columns
    print(cols)
    data_new.loc[2014] = None
    data_new.loc[2015] = None
    for i in range(len(cols) - 1):
        f, a, b, x0, C, P = GM11(data_new.loc[range(1994, 2014), cols[i]].as_matrix())  # C<0.35,0.5,0.65, P>0.95,0.8
        data_new.loc[2014, cols[i]] = f(len(data_new) - 1)
        data_new.loc[2015, cols[i]] = f(len(data_new))
    return data_new


def finall_model(data):
    # 数据标准化
    y = pd.DataFrame(data.iloc[:, -1])
    data = pd.DataFrame(data.iloc[:, :-1])
    ss = StandardScaler().fit(data.loc[range(1994, 2014), :])
    data.loc[:, :] = ss.transform(data.loc[:, :])
    x_train = data.iloc[:-2, :]
    x_test = data.iloc[-2:, :]
    y_train = y.iloc[:-2, :]
    ss = StandardScaler()
    y_train = ss.fit_transform(y_train)
    model1 = LinearRegression()
    model2 = SVR()
    model3 = LinearSVR()
    model4 = MLPRegressor(hidden_layer_sizes=(100, 2))
    model1.fit(x_train, y_train)
    model2.fit(x_train, y_train)
    model3.fit(x_train, y_train)
    model4.fit(x_train, y_train)
    print(model1.score(x_train, y_train))
    print(model2.score(x_train, y_train))
    print(model3.score(x_train, y_train))
    print(model4.score(x_train, y_train))
    y_ = model1.predict(x_test)
    yy = np.sqrt(ss.var_) * y_ + ss.mean_

    plt.plot(y.loc[range(1994, 2014), "y"])
    plt.scatter([2014, 2015], yy, marker='*')
    plt.show()
    return data


if __name__ == '__main__':
    data = read_csv()
    # draw1(data)
    # print(data.head())
    data = LR(data)
    data = predict_x(data)
    data = finall_model(data)
