import pandas as pd
import numpy as np

from keras.layers.core import Activation, Dense
from keras.models import Sequential

'''
session_1()         数据筛选
session_2()         阈值寻优
session_3()         建立神经网络模型，并对模型进行检验
session_4()         根据特征推算是否满足某项条件
'''


def session_1():
    # 阈值
    threshold = pd.Timedelta('4 min')

    data = pd.read_excel('water_heater.xls')

    # DataFrame处理
    data['发生时间'] = pd.to_datetime(data['发生时间'], format='%Y%m%d%H%M%S')
    data = data[data['水流量'] > 0]  # 只记录流量大于0的数据
    d = data['发生时间'].diff() > threshold  # 相邻时间做差分，大于threshold
    data['事件编号'] = d.cumsum() + 1  # 通过累计求和的方式为事件编号

    data.to_csv('divid_sequence.csv')


def session_2():
    # 使用之后四个点的平均斜率
    n = 4

    # 专家阈值
    threshold = pd.Timedelta(minutes=5)
    data = pd.read_excel('water_heater.xls')
    data['发生时间'] = pd.to_datetime(data['发生时间'], format='%Y%m%d%H%M%S')
    data = data[data['水流量'] > 0]

    # 定义阈值列
    dt = [pd.Timedelta(minutes=i) for i in np.arange(1, 9, 0.25)]
    h = pd.DataFrame(dt, columns=['阈值'])

    # 相邻时间的差分，比较是否大于阈值
    def event_num(ts):
        d = data['发生时间'].diff() > ts

        # 返回事件数
        return d.sum() + 1

    # 计算每个阈值对应的事件数
    h['事件数'] = h['阈值'].apply(event_num)

    # 计算每两个相邻点对应的斜率
    h['斜率'] = h['事件数'].diff() / 0.25

    # 采用后n个斜率的绝对值的平均作为斜率指标
    h['斜率指标'] = pd.Series.rolling(h['斜率'].abs(), n).mean()
    ts = h['阈值'][h['斜率指标'].idxmin() - n]

    if ts > threshold:
        ts = pd.Timedelta(minutes=4)

    print(ts)


def session_3():
    # 读取训练集和测试集，并且划分样本特征和标签
    data_train = pd.read_excel('train_neural_network_data.xls')
    data_test = pd.read_excel('test_neural_network_data.xls')
    y_train = data_train.iloc[:, 4].values
    x_train = data_train.iloc[:, 5:17].values
    y_test = data_test.iloc[:, 4].values
    x_test = data_test.iloc[:, 5:17].values

    # 建立神经网络模型
    model = Sequential()
    model.add(Dense(17, input_shape=(11,)))
    model.add(Activation('relu'))
    model.add(Dense(10, input_shape=(17,)))
    model.add(Activation('relu'))
    model.add(Dense(1, input_shape=(10,)))
    model.add(Activation('sigmoid'))

    # 编译模型
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        sample_weight_mode='binary'
    )

    # 训练模型
    model.fit(x_train, y_train, epochs=100, batch_size=1)

    # 保存模型
    model.save_weights('net.model')

    # 进行预测
    r = pd.DataFrame(model.predict_classes(x_test), columns=['预测结果'])
    pd.concat([data_test.iloc[:, :5], r], axis=1).to_csv('test_output_data.csv')
    model.predict(x_test)
    return y_test


def session_4():
    threshold = pd.Timedelta('4 min')
    data = pd.read_excel('water_heater.xls')

    data['发生时间'] = pd.to_datetime(data['发生时间'], format='%Y%m%d%H%M%S')
    data = data[data['水流量'] > 0]
    d = data['发生时间'].diff() > threshold
    data['事件编号'] = d.cumsum() + 1

    data_g = data.groupby('事件编号')
    result = pd.DataFrame()
    dt = pd.Timedelta(seconds=2)

    for _, g in data_g:
        temp = pd.DataFrame(index=[0])

        # 根据用水时长、开关机切换次数、总用水量退出是否洗澡
        t_start = g['发生时间'].min()
        t_end = g['发生时间'].max()
        temp['用水事件时长(M)'] = (dt+t_end-t_start).total_seconds()/60
        temp['开关机切换次数'] = (pd.Series.rolling(g['开关机状态'] == '关',2).sum() == 1).sum()
        temp['总用水量(L)'] = g['水流量'].sum()
        t_diff = g['发生时间'].diff()

        if len(g['发生时间']) == 1:
            temp["总用水时长（Min）"] = dt.total_seconds() / 60
        else:
            temp["总用水时长（Min）"] = (
                t_diff.sum() - t_diff.iloc[1] / 2 -
                t_diff.iloc[len(t_diff) - 1] / 2).total_seconds() / 60

        temp["平均水流量（L/min）"] = temp["总用水量(L)"] / temp["总用水时长（Min）"]
        result = result.append(temp, ignore_index=True)

    result.to_csv('attribute_extract.csv')


if __name__ == '__main__':
    # session_1()
    # session_2()
    # session_3()
    session_4()
