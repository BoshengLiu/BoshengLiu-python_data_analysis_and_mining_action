import pandas as pd
from sklearn.cluster import KMeans

'''
session_1()         提取原始数据的一些特征描述并保存为新表
session_2()         对原始数据进行清理，提取其中的一些数据并保存到新的文件
session_3()         标准化数据并进行保存
session_4()         使用K-means对数据进行聚类分析
'''


def session_1():
    data = pd.read_csv('air_data.csv')

    # 对原始数据基本描述，percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）；T是转置，转置后更方便查阅
    explorer = data.describe(percentiles=[], include='all').T
    # print(explorer)

    explorer['null'] = len(data) - explorer['count']
    explorer = explorer[['null', 'max', 'min']]
    explorer.columns = ['空值', u'最大值', u'最小值']

    explorer.to_csv('explorer.csv', index=False)


def session_2():
    data = pd.read_csv('air_data.csv')

    # 使用乘法运算非空数值的数据，因为numpy不支持*运算，在这里换做&运算
    data = data[data['SUM_YR_1'].notnull() & data['SUM_YR_2'].notnull()]

    # 只保留票价非零的，或者平均折扣率与总飞行公里数同时为0的记录
    index_1 = data['SUM_YR_1'] != 0
    index_2 = data['SUM_YR_2'] != 0
    index_3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)   # 这里是'与'
    data = data[index_1 | index_2 | index_3]    # 这里是'或'

    data.to_csv('data_clean.csv', index=False)


def session_3():
    data = pd.read_excel('zscoredata.xls')

    # 核心语句，实现标准化变换，类似的可以实现任何想要的变换
    data = (data - data.mean(axis=0)) / (data.std(axis=0))
    data.columns = ['Z' + i for i in data.columns]

    data.to_csv('zs_core_data.csv', index=False)


def session_4():
    k = 5
    data = pd.read_excel('zscoreddata.xls')

    k_model = KMeans(n_clusters=k, n_jobs=4)
    k_model.fit(data)

    print(k_model.cluster_centers_)  # 查看聚类中心
    print(k_model.labels_)  # 查看各样本对应的类别


if __name__ == '__main__':
    # session_1()
    # session_2()
    # session_3()
    session_4()
