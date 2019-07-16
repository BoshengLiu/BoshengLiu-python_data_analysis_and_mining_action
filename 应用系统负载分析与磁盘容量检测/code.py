import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller as ADF

'''
attr_trans          属性变换
session_1()         数据筛选
session_2()         平稳性检测
session_3()         白噪声检测
session_4()         确定最佳p、d、q值
session_5()         模型检验
session_6()         计算预测误差
'''


# 属性变换，改变列名
def attr_trans(x):
    result = pd.Series(
        index=['SYS_NAME', 'CWXT_DB:184:C:\\', 'CWXT_DB:184:D:\\', 'COLLECTTIME'])
    result['SYS_NAME'] = x['SYS_NAME'].iloc[0]
    result['COLLECTTIME'] = x['COLLECTTIME'].iloc[0]
    result['CWXT_DB:184:C:\\'] = x['VALUE'].iloc[0]
    result['CWXT_DB:184:D:\\'] = x['VALUE'].iloc[1]

    return result


def session_1():
    data = pd.read_excel('discdata.xls')

    # 提取某部分数据
    data = data[data['TARGET_ID'] == 184].copy()

    # 以某字段进行分组
    data_group = data.groupby('COLLECTTIME')

    # 逐组处理
    data_processed = data_group.apply(attr_trans)
    data_processed.to_csv('discdata_processed.csv', index=False)


def session_2():
    data = pd.read_csv('discdata_processed.csv')

    # 去除最后5个数据，不使用最后5个数据
    predict_num = 5
    data = data.iloc[:len(data) - predict_num]

    # 平稳性检测
    diff = 0
    adf = ADF(data['CWXT_DB:184:D:\\'])
    while adf[1] > 0.05:  # adf[1]为p值，p值小于0.05可认为是平稳的
        diff = diff + 1
        adf = ADF(data['CWXT_DB:184:D:\\'].diff(diff).dropna())

    print('原始序列经过%s阶差分后归于平稳，p值为%s' % (diff, adf[1]))


def session_3():
    data = pd.read_csv('discdata_processed.csv')
    data = data.iloc[:len(data) - 5]

    # 白噪声检测
    [[lb], [p]] = acorr_ljungbox(data['CWXT_DB:184:D:\\'], lags=1)

    if p < 0.05:
        print('原始序列为非白噪声序列，对应的p值为：%s' % p)
    else:
        print('原始序列为白噪声序列，对应的p值为：%s' % p)

    # 一阶差分后的白噪声
    [[lb], [p]] = acorr_ljungbox(data['CWXT_DB:184:D:\\'].diff().dropna(), lags=1)

    if p < 0.05:
        print('原始序列为非白噪声序列，对应的p值为：%s' % p)
    else:
        print('原始序列为白噪声序列，对应的p值为：%s' % p)

    print(lb)


def session_4():
    data = pd.read_csv('discdata_processed.csv', index_col='COLLECTTIME')
    # 不使用最后5个数据
    data = data.iloc[:len(data) - 5]
    x_data = data['CWXT_DB:184:D:\\']

    # 定阶，一般阶数不超过length/10
    p_max = int(len(x_data) / 10)
    q_max = int(len(x_data) / 10)

    # 定义bic矩阵
    bic_matrix = []
    for p in range(p_max + 1):
        temp = []
        for q in range(q_max + 1):
            # 存在部分错误，所以通过try跳过报错
            try:
                temp.append(ARIMA(x_data, (p, 1, q)).fit().bic)
            except:
                temp.append(None)
        bic_matrix.append(temp)

    bic_matrix = pd.DataFrame(bic_matrix)

    # 找出最小值，先用stack展平，然后用idxmin找出最小值位置，
    p, q = bic_matrix.stack().astype(float).idxmin()
    print('BIC最小的p值和q值为：%s、%s' % (p, q))


def session_5():
    # 残差延迟个数
    lag_num = 5

    data = pd.read_csv('discdata_processed.csv', index_col='COLLECTTIME')
    data = data.iloc[:len(data) - 5]
    x_data = data['CWXT_DB:184:D:\\']

    # 训练模型并预测，计算残差
    model = ARIMA(x_data, (0, 1, 1)).fit(disp=-1)
    x_data_pre = model.predict(typ='levels')
    pre_error = (x_data_pre - x_data).dropna()

    lb, p = acorr_ljungbox(pre_error, lags=lag_num)
    h = (p < 0.05).sum()
    if h > 0:
        print('模型ARIMA(0,1,1)不符合白噪声检验')
    else:
        print('模型ARIMA(0,1,1)符合白噪声检验')

    print(lb)


def session_6():
    data = pd.read_excel('predictdata.xls')

    # 计算误差
    abs_ = (data['预测值']-data['实际值']).abs()
    mae_ = abs_.mean()
    rmse_ = ((abs_**2).mean())**0.5
    mape_ = (abs_/data['实际值']).mean()

    print('平均误差为：%0.4f'
          '\n均方根误差为：%0.4f'
          '\n平均绝对百分误差：%0.6f'%(mae_,rmse_,mape_))


if __name__ == '__main__':
    # session_1()
    # session_2()
    # session_3()
    # session_4()
    # session_5()
    session_6()
