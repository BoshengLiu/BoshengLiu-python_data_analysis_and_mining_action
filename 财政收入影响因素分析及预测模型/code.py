import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from sklearn.linear_model import Lasso

'''
GM1             自定义的灰度预测函数
session_1       读取文件，提取基本信息
session_2       用灰度预测函数进行预测
session_3       建立神经网络模型，进行预测并绘制预测图
session_4       用灰度月模型进行预测一组数据，并绘制预测图
'''


def GM1(x0):
    # 1-AGO序列，累计求和
    x1 = np.cumsum(x0)

    # 紧邻均值(MEAN)生成序列
    z1 = (x1[:-1] + x1[1:]) / 2.0
    z1 = z1.reshape(len(z1), 1)
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Yn = x0[1:].reshape((len(x0) - 1, 1))

    # 矩阵计算，计算参数
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn)

    # 还原值
    f = lambda k: (x0[0] - b / a) * np.exp(-a * (k - 1)) - (x0[0] - b / a) * np.exp(-a * (k - 2))

    delta = np.abs(x0 - np.array([f(i) for i in range(1, len(x0) + 1)]))
    C = delta.std() / x0.std()
    P = 1.0 * (np.abs(delta - delta.mean()) < 0.6745 * x0.std()).sum() / len(x0)

    # 灰度预测函数、a、b、首项、方差小残差概率
    return f, a, b, x0[0], C, P


def session_1(input_file, data_range):
    data = pd.read_csv(input_file)

    '''
    原始方法，替代方法可以使用describe()方法，然后进行筛选
    r = [data.min(), data.max(), data.mean(), data.std()]
    r = pd.DataFrame(r, index=['Min', 'Max', 'Mean', 'STD'])
    '''
    r = pd.DataFrame(data.describe()).T
    np.round(r, 2)

    # 计算相关系数矩阵
    np.round(data.corr(method='pearson'), 2)

    '''
    原代码使用AdaptiveLasso,现更新为Lasso
    参数也由gamma变为tol
    '''
    model = Lasso(tol=1)
    model.fit(data.iloc[:, 0:data_range], data['y'])

    # 各个特征的系数
    print(model.coef_)


def session_2(input_file, output_file, start_year, feature_list, round_num):
    '''
    start_year: 开始年份
    feature_list: 特征列
    round_num: 四舍五入保留的位数
    '''

    data = pd.read_csv(input_file)
    data.index = range(start_year, 2014)

    data.loc[2014] = None
    data.loc[2015] = None

    for i in feature_list:
        f = GM1(data[i][list(range(start_year, 2014))].values)[0]
        # 2014年预测结果
        data[i][2014] = f(len(data) - 1)
        # 2015年预测结果
        data[i][2015] = f(len(data) - 1)
        data[i] = data[i].round(round_num)

    print(data[feature_list + ['y']])
    data[feature_list + ['y']].to_csv(output_file)


def session_3(input_file, output_file, model_file, feature_list, start_year, input_dim_1,
              units1, input_dim_2, units2, epochs_num=10000, round_num=0):
    '''
    feature_list: 特征列
    input_dim_1: 第一层训练模型层数
    units1: 第一层神经元个数
    input_dim_2: 第二层训练模型层数
    units2: 第二层神经元个数
    epochs_num:训练轮数
    round_num: 四舍五入
    '''
    data = pd.read_csv(input_file)

    # 特征列，取start_year年以前的数据
    data_train = data.loc[range(start_year, 2014)].copy()
    data_mean = data_train.mean()
    data_std = data.std()

    # 数据标准化
    data_train = (data_train - data_mean) / data_std

    # 特征数据
    x_train = data_train[feature_list].values

    # 标签数据
    y_train = data_train['y'].values

    model = Sequential()
    model.add(Dense(input_dim=input_dim_1, units=units1))
    model.add(Activation('relu'))
    model.add(Dense(input_dim=input_dim_2, units=units2))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs_num, batch_size=16)
    model.save_weights(model_file)

    # 预测，并还原结果
    x = ((data[feature_list] - data_mean[feature_list]) / data_std[feature_list]).values
    data['y_pred'] = model.predict(x) * data_std['y'] + data_mean['y']
    data['y_pred'] = data['y_pred'].round(round_num)

    data.to_csv(output_file)

    # 画出预测结果图
    data[['y', 'y_pred']].plot(subplots=True, style=['b-o', 'r-*'])
    plt.show()


def session_4():
    x0 = np.array([3152063, 2213050, 4050122, 5265142,
                   5556619, 4772843, 9463330])
    f, a, b, x00, C, P = GM1(x0)
    print(a, b, x00, C, P)
    print('2014年、2015年的预测结果分别为：\n%0.2f万元和%0.2f万元' % (f(8), f(9)))
    print('后验差比值为：%0.4f' % C)

    p = pd.DataFrame(x0, columns=['y'], index=range(2007, 2014))
    p.loc[2014] = None
    p.loc[2015] = None
    p['y_pred'] = [f(i) for i in range(1, 10)]
    p['y_pred'] = p['y_pred'].round(2)
    p.index = pd.to_datetime(p.index, format='%Y')

    p.plot(style=['b-0', 'r-*'], xticks=p.index)
    plt.show()


if __name__ == '__main__':
    # session_1(input_file='data1.csv', data_range=13)
    # session_2(input_file='data1.csv',
    #           output_file='data1_GM1.csv',
    #           start_year=1994,
    #           feature_list=['x1', 'x2', 'x3', 'x4', 'x5', 'x7'],
    #           round_num=2)
    # session_3(input_file='data1_GM1.csv',
    #           output_file='revenue.csv',
    #           model_file='1-net.model',
    #           feature_list=['x1', 'x2', 'x3', 'x4', 'x5', 'x7'],
    #           start_year=1994,
    #           input_dim_1=6, units1=6,
    #           input_dim_2=12, units2=1,
    #           round_num=1)

    # session_1(input_file='data2.csv', data_range=6)
    # session_2(input_file='data2.csv',
    #           output_file='data2_GM1.csv',
    #           start_year=1999,
    #           feature_list=['x1', 'x3', 'x5'],
    #           round_num=6)
    # session_3(input_file='data2_GM1.csv',
    #           output_file='VAT.csv',
    #           model_file='2-net.model',
    #           feature_list=['x1', 'x3', 'x5'],
    #           start_year=1999,
    #           input_dim_1=3, units1=6,
    #           input_dim_2=6, units2=1,
    #           round_num=2)

    # session_1(input_file='data3.csv', data_range=10)
    # session_2(input_file='data3.csv',
    #           output_file='data3_GM1.csv',
    #           start_year=1999,
    #           feature_list=['x3', 'x4', 'x6', 'x8'],)
    # session_3(input_file='data3_GM1.csv',
    #           output_file='sales_tax.csv',
    #           model_file='3-net.model',
    #           feature_list=['x3', 'x4', 'x6', 'x8'],
    #           start_year=1999,
    #           input_dim_1=4, units1=8,
    #           input_dim_2=8, units2=1,
    #           round_num=2)

    # session_1(input_file='data4.csv', data_range=10)
    # session_2(input_file='data4.csv',
    #           output_file='data4_GM1.csv',
    #           start_year=2002,
    #           feature_list=["x1", "x2", "x3", "x4", "x6", "x7", "x9", "x10"],
    #           round_num=2)
    # session_3(input_file='data4_GM1.csv',
    #           output_file='enterprise.csv',
    #           model_file='4-net.model',
    #           feature_list=["x1", "x2", "x3", "x4", "x6", "x7", "x9", "x10"],
    #           start_year=2002,
    #           input_dim_1=8, units1=6,
    #           input_dim_2=6, units2=1,
    #           round_num=2)

    # session_1(input_file='data5.csv', data_range=7)
    # session_2(input_file='data5.csv',
    #           output_file='data5_GM1.csv',
    #           start_year=2000,
    #           feature_list=["x1", "x4", "x5", "x7"])
    # session_3(input_file='data5_GM1.csv',
    #           output_file='enterprise.csv',
    #           model_file='5-net.model',
    #           feature_list=["x1", "x4", "x5", "x7"],
    #           start_year=2000,
    #           input_dim_1=4, units1=8,
    #           input_dim_2=8, units2=1,
    #           epochs_num=15000)

    session_4()
