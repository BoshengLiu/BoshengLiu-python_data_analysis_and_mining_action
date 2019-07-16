import time

import pandas as pd
from sklearn.cluster import KMeans

'''
session_1()         进行聚类离散化
session_2()         
find_rule()         定义关联规则
connect_string()    字符串连接
'''


def session_1():
    type_label = {
        "肝气郁结证型系数": "A",
        "热毒蕴结证型系数": "B",
        "冲任失调证型系数": "C",
        "气血两虚证型系数": "D",
        "脾胃虚弱证型系数": "E",
        "肝肾阴虚证型系数": "F",
    }
    k = 4  # 需要进行的聚类类别数

    data = pd.read_excel('data.xls')
    result = pd.DataFrame()

    for key, item in type_label.items():
        print('正在进行“%s”的聚类...' % key)

        # 进行聚类离散化
        k_model = KMeans(n_clusters=k, n_jobs=4)
        k_model.fit(data[[key]].values)  # 训练模型

        # 聚类中心
        r1 = pd.DataFrame(k_model.cluster_centers_, columns=[item])

        # 分类统计
        r2 = pd.Series(k_model.labels_).value_counts()
        r2 = pd.DataFrame(r2, columns=[item + 'n'])  # 记录各个类别的数目

        # 合并为一个DataFrame
        r = pd.concat([r1, r2], axis=1).sort_values(item)  # 匹配聚类中心和类别数目
        r.index = [1, 2, 3, 4]

        # 用来计算相邻两列的均值，以此作为边界点
        r[item] = pd.Series.rolling(r[item], 2).mean()

        # 将NaN值转换为0.0，将原来的聚类中心改为边界点
        r.loc[1, item] = 0.0
        result = result.append(r.T)

    # 以ABCDEF排序
    result = result.sort_index()
    result.to_csv('data_process.csv')


# 自定义连接函数
def connect_string(x, y):
    x = list(map(lambda i: sorted(i.split(y)), x))
    r = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i][:-1] == x[j][:-1] and x[i][-1] != x[j][:-1]:
                r.append(x[i][:-1] + sorted([x[j][-1], x[i][-1]]))
    return r


# 寻找关联规则函数
def find_rule(x, support, confidence, ms='--'):
    result = pd.DataFrame(index=['support', 'confidence'])

    # 第一批支持度筛选
    support_series = 1.0 * x.sum() / len(x)

    column = list(support_series[support_series > support].index)
    k = 0

    while len(column) > 1:
        k = k + 1
        print('\n正在进行第%s次搜索...' % k)

        column = connect_string(column, ms)
        print('数目%s...' % len(column))
        index_list = [ms.join(i) for i in column]

        # 新的支持度函数
        sf = lambda i: x[i].prod(axis=1, numeric_only=True)

        # 计算连接后的支持度，开始筛选
        x_1 = pd.DataFrame(list(map(sf, column)), index=index_list).T
        support_series_1 = 1.0 * x_1[index_list].sum() / len(x_1)
        column = list(support_series_1[support_series_1 > support].index)

        support_series = support_series.append(support_series_1)
        column_1 = []

        # 遍历所有可能的情况
        for i in column:
            i = i.split(ms)
            for j in range(len(i)):
                column_1.append(i[:j] + i[j + 1:] + i[j:j + 1])

        # 置信度序列
        confidence_series = pd.Series(index=[ms.join(i) for i in column_1])

        for i in column_1:
            confidence_series[ms.join(i)] = support_series[
                                                ms.join(sorted(i))] / support_series[ms.join(i[:-1])]

        # 置信度筛选
        for i in confidence_series[confidence_series > confidence].index:
            result[i] = 0.0
            result[i]['confidence'] = confidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(['confidence', 'support'], ascending=False)
    print(u'\nresult:')
    print(result)

    return result


def session_2():
    data = pd.read_csv('apriori.txt', header=None, dtype=object)

    # 计时
    start = time.perf_counter()
    print('\n转换原始数据至0-1矩阵...')

    # 0-1矩阵的转换
    ct = lambda x: pd.Series(1, index=x[pd.notnull(x)])
    b = list(map(ct, data.values))
    data = pd.DataFrame(b).fillna(0)
    end = time.perf_counter()
    print('\n转换完成，用时: %0.2f s' % (end - start))

    # 删除中间变量b，节省内存
    del b

    # 定义支持度，置信度，连接符号
    support = 0.06
    confidence = 0.75
    ms = '---'

    # 计时
    start = time.perf_counter()
    print('\n开始搜素关联规则...')
    find_rule(data, support, confidence, ms)
    end = time.perf_counter()
    print('\n搜素完成，用时%0.2f s' % (end - start))


if __name__ == '__main__':
    # session_1()
    session_2()
