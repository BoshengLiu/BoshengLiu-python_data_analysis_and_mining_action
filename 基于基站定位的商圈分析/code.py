import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

'''
session_1           获取数据，进行离差标准化
session_2           画谱系聚类图
session_3           层次聚类算法，并将每类进行可视化操作
'''


def session_1():
    data_file = 'business_circle.xls'
    standardize_file = 'standardized.csv'

    data = pd.read_excel(data_file, index_col='基站编号')
    data = (data - data.min()) / (data.max() - data.min())
    data = data.reset_index()

    data.to_csv(standardize_file, index=False)


def session_2():
    standardize_file = 'standardized.csv'
    data = pd.read_csv(standardize_file, index_col='基站编号')

    plt.figure(figsize=(20, 10))
    Z = linkage(data, method='ward', metric='euclidean')
    P = dendrogram(Z, 0)
    plt.show()

    return P


def session_3():
    standardize_file = 'standardized.csv'
    k = 3
    data = pd.read_csv(standardize_file, index_col='基站编号')

    # 层次聚类
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    model.fit(data)

    # 详细输入原始数据及对应类别
    r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
    r.columns = list(data.columns) + ['聚类类别']

    # 绘制聚类图，并且用不同样式进行画图
    style = ['ro-', 'go-', 'b0-']
    xlabels = ['工作日人均停留时间', '凌晨人均停留时间', '周末人均停留时间', '日均人流量']
    pic_output = 'type_'

    for i in range(k):
        plt.figure()
        tmp = r[r['聚类类别'] == i].iloc[:, :4]

        for j in range(len(tmp)):
            plt.plot(range(1, 5), tmp.iloc[j], style[i])

        plt.xticks(range(1, 5), xlabels, rotation=20)

        plt.title('商圈类别%s' % (i + 1))

        # 调整底部
        plt.subplots_adjust(bottom=0.15)
        plt.savefig('%s%s.png' % (pic_output, i+1))


if __name__ == '__main__':
    # session_1()
    # session_2()
    session_3()