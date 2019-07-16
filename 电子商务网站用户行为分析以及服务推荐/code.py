import pandas as pd
import numpy as np

from sqlalchemy import create_engine
import pymysql

'''
sql_process         连接并访问数据库，进行分块统计
count107            统计107类别情况
session_1           大概了解了处理数据意图
session_2           提取所需数据，并且保存到数据库中
session_3           进行数据筛选，保存到数据库中
session_4           合并某些特征为一个特征，保存到数据库
Recommender         推荐矩阵
'''


def count107(i):
    j = i[['fullURL']][i['fullURLId'].str.contains('107')].copy()

    # 添加空列
    j['type'] = None

    # 利用正则进行匹配，并重命名
    j['type'][j['fullURL'].str.contains('info/.+?/')] = "知识首页"
    j['type'][j['fullURL'].str.contains('info/.+?/.+?')] = "知识列表页"
    j['type'][j['fullURL'].str.contains('/\d+?_*\d+?\.html')] = "知识内容页"

    return j['type'].value_counts()


def session_1():
    '''
    用pymysql连接本地数据库
    engine表示连接数据的引擎，chunksize表示每次读取的数据量
    注：这里分三个部分，每次只运行一个部分
    '''
    engine = create_engine('mysql+mysqlconnector://root@localhost/7law')
    sql = pd.read_sql('all_gzdata', engine, chunksize=10000)

    # 第一部分
    # 逐块统计，合并统计结果，把相同的统计项合并（即按index分组并求和）
    counts = [i['fullURLId'].value_counts() for i in sql]
    counts = pd.concat(counts).groupby(level=0).sum()
    # 重新设置index，将原来的index作为columns
    counts = counts.reset_index()
    counts.columns = ['index', 'num']
    # 修改列名，提取前3个数字作为类别id，按类别合并
    counts['type'] = counts['index'].str.extract('(\d{3})')
    counts_ = counts[['type', 'num']].groupby('type').sum()
    # 按类别排列
    counts_.sort_values('num', ascending=False)

    # 第二部分，同counts
    sql = pd.read_sql('all_gzdata', engine, chunksize=10000)
    counts1 = [count107(i) for i in sql]
    counts1 = pd.concat(counts1).groupby(level=0).sum()

    # 第三部分
    # 统计次数，同上分块统计结果并合并t
    c = [i['realIP'].value_counts() for i in sql]
    counts2 = pd.concat(c).groupby(level=0).sum()
    counts2 = pd.DataFrame(counts2)
    # 添加新列，全为1，添加某特征分别出现的次数
    counts2[1] = 1
    counts2.groupby(level=0).sum()


def session_2():
    engine = create_engine('mysql+mysqlconnector://root@localhost/7law')
    sql = pd.read_sql('all_gzdata', engine, chunksize=10000)

    for i in sql:
        # 只要网址列
        d = i[['realIP', 'fullURL']]

        # 只要含有.html的网址
        d = d[d['fullURL'].str.contains('\.html')].copy()

        # 保存数据到表中，如果不存在则创建表
        d.to_sql('cleaned_gzdata', engine, index=False, if_exists='append')


def session_3():
    engine = create_engine('mysql+mysqlconnector://root@localhost/7law')
    sql = pd.read_sql('cleaned_gzdata', engine, chunksize=10000)

    for i in sql:
        d = i.copy()

        # 替换关键词
        d['fullURL'] = d['fullURL'].str.replace('_\d{0,2}.html', '.html')

        # 去除重复数据
        d = d.drop_duplicates()
        d.to_sql('changed_gzdata', engine, index=False, if_exists='append')


def session_4():
    engine = create_engine('mysql+mysqlconnector://root@localhost/7law')
    sql = pd.read_sql('changed_gzdata', engine, chunksize=10000)

    for i in sql:
        d = i.copy()
        d['type_1'] = d['fullURL']
        d['type_1'][d['fullURL'].str.extract('(ask)|(askzt)')] = 'zixun'
        d.to_sql('splited_gzdata', engine, index=False, if_exists='append')


# 自定义杰卡德相似系数函数，仅对0-1矩阵有效
def Jaccard(a, b):
    return 1.0 * (a * b).sum() / (a + b - a * b).sum()


class Recommender():
    sim = None

    # 判断距离（相似性）
    def similarity(self, x, distance):
        y = np.ones((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                y[i, j] = distance(x[i], x[j])

        return y

    def fit(self, x, distance=Jaccard):
        self.sim = self.similarity(x, distance)

    # 推荐矩阵
    def recommend(self, a):
        return np.dot(self.sim, a) * (1 - a)


if __name__ == '__main__':
    # session_1()
    # session_2()
    # session_3()
    # session_4()
    Recommender()
