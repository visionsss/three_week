import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def read_csv():
    """
    读取文件
    :return:
    """
    file_path = r'./air_data.csv'
    df = pd.read_csv(file_path)
    return df


def process(data):
    """
    预处理文件
    :param data:
    :return:
    """
    # 票价为缺失值--删除
    ind = data['SUM_YR_1'].notnull()
    ind2 = data['SUM_YR_2'].notnull()
    data2 = data.loc[ind & ind2, :]
    # 票价为零--删除
    ind = data2['SUM_YR_1'] != 0
    ind2 = data2['SUM_YR_2'] != 0
    data3 = data2.loc[ind & ind2, :]
    # 平均折扣为零--删除
    data4 = data3.loc[
            data3['avg_discount'] != 0, :
            ]
    # 飞行里程为零--删除
    data_process = data4.loc[
                   data4['SEG_KM_SUM'] != 0, :
                   ]
    return data_process


def get_features(df):
    """
    预处理特征
    """
    data_process = df
    # LRFMC
    L = pd.to_datetime('2014/3/31') \
        - pd.to_datetime(data_process['FFP_DATE'])
    L.dtype  # 查看数据类型
    L = L.dt.days / 30
    R = data_process['LAST_TO_END']
    F = data_process['FLIGHT_COUNT']
    M = data_process['SEG_KM_SUM']
    C = data_process['avg_discount']
    data_new = pd.concat([L, R, F, M, C], axis=1)
    return data_new


def Standard(df):
    """
    标准化特征
    :param df:
    :return:
    """
    df = np.array(df)
    ss = StandardScaler().fit_transform(df)
    return ss


def cluster(df):
    model = KMeans(n_clusters=5)  # 模型构建
    model.fit(df)  # 模型训练
    cluster_centers = model.cluster_centers_  # 聚类中心
    labels = model.labels_  # 聚类标签
    N = 5  # 纬度
    M = 5  # 聚类个数

    angles = np.linspace(0, 2 * np.pi, N,
                         endpoint=False)
    angles = np.concatenate([angles,
                             [angles[0]]])

    features = list("LRFMC")  # 维度名称
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = 'Simhei'
    plt.style.use('ggplot')
    fig = plt.figure(dpi=90)  # 设置画布
    # 声明使用极坐标
    ax = fig.add_subplot(111, polar=True)
    # 设置雷达图特征参数
    ax.set_thetagrids(angles * 180 / np.pi,
                      features)
    # 设置雷达图取值范围
    ax.set_ylim(cluster_centers.min(),
                cluster_centers.max())
    plt.title('聚类结果雷达图展示')
    plt.grid(True)
    cols = ['r-', 'g-', 'b-', 'p-', 'o-']
    for i in range(M):
        value = cluster_centers[i]
        value = np.concatenate([value,
                                [value[0]]])
        ax.plot(angles, value, cols[i])
        ax.fill(angles, value, cols[i], alpha=0.2)
    plt.legend(['客户群' + str(i) for i in range(M)])
    plt.show()


if __name__ == '__main__':
    df = read_csv()
    print(df.shape)
    df = process(df)
    print(df.shape)
    df = get_features(df)
    print(df.head())
    df = Standard(df)
    print(df)
    cluster(df)
