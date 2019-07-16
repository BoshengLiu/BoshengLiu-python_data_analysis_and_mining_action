import xlrd
from random import shuffle

import pandas as pd
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier

import warnings

warnings.filterwarnings("ignore", module="matplotlib")

'''
cm_plot         自定义混淆矩阵可视化
session_1       使用拉格朗日插值法进行插值
session_2       使用神经网络模型，进行预测给出训练结果，并绘制ROC曲线
session_3       构建CART决策树模型，进行预测给出训练结果，并绘制ROC曲线
session_4       读取pkl文件
'''


def cm_plot(x, y):
    cm = confusion_matrix(x, y)
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar()

    for i in range(len(cm)):
        for x in range(len(cm)):
            plt.annotate(
                cm[i, x],
                xy=(i, x),
                horizontalalignment='center',
                verticalalignment='center'
            )

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    return plt


def session_1():
    data = pd.read_excel('missing_data.xls')

    # 自定义列向量插值函数
    # s为列向量，n为被插值的位置，k为取值前后的数据个数，默认为5
    def ployinterp_column(s, n, k=5):
        y = s.reindex(list(range(n - k, n)) + list(range(n + 1, n + 1 + k)))
        y = y[y.notnull()]

        return lagrange(y.index, list(y))(n)        # 插值并返回结果

    # 诸葛判断元素是否需要插值
    for i in data.columns:
        for j in range(len(data)):
            if (data[i].isnull())[j]:
                data[i][j] = ployinterp_column(data[i], j)

    data.to_csv('missing_data_process.csv', index=False)


def session_2():
    data_model = pd.read_excel('model.xls')
    data_model = data_model.values
    shuffle(data_model)

    p = 0.8
    train = data_model[:int(len(data_model) * p), :]
    test = data_model[int(len(data_model) * p):, :]

    # 构建LM神经网络模型
    net = Sequential()                              # 建立神经网络

    # net.add(Dense(input_dim=3,units=10))          # 添加输入层（3节点）到隐藏层（10节点）的连接
    net.add(Dense(10, input_shape=(3,)))
    net.add(Activation('relu'))                     # 隐藏层使用relu激活函数

    # net.add(Dense(input_dim=10, uints=1))
    net.add(Dense(1, input_shape=(10,)))            # 添加隐藏层（10节点）到输出层（1节点）的连接
    net.add(Activation('sigmoid'))                  # 输出层使用sigmoid激活函数

    net.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])                                               # 编译模型，使用adam方法求解

    net.fit(train[:, :3], train[:, 3], epochs=1000, batch_size=1)           # 训练模型，循环1000次
    net.save_weights('model.csv')                                           # 保存模型在本地
    predict_result = net.predict_classes(train[:, :3]).reshape(len(train))  # 预测结果变形
    ''' 这里要提醒的是，keras用predict给出预测概率，predict_classes才是给出的预测类别，而且两者的预测结果都是    
        n * 1 维的数组，而不是 1 * n
    '''
    cm_plot(train[:, 3], predict_result).show()                             # 显示混淆化矩阵可视化结果

    predict_result = net.predict(test[:, :3]).reshape(len(test))
    fpr, tpr, thresholds = roc_curve(test[:, 3], predict_result, pos_label=1)

    plt.plot(fpr, tpr, linewidth=2, label='ROC of LM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # 设定边界
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(loc=4)
    plt.show()

    print(thresholds)


def session_3():
    data_model = pd.read_excel('model.xls')
    data_model = data_model.values                      # 将表格转换为矩阵
    shuffle(data_model)                                 # 随机打乱数据

    p = 0.8  # 设置训练数据比列
    train = data_model[:int(len(data_model) * p), :]    # 前80%为训练数据
    test = data_model[int(len(data_model) * p):, :]     # 后20%为测试数据

    # 构建CART决策树模型
    tree = DecisionTreeClassifier()                     # 建立决策树模型
    tree.fit(train[:, :3], train[:, 3])

    # 保存模型
    joblib.dump(tree, 'tree.pkl')

    # 显示混淆化矩阵可视化结果
    cm_plot(train[:, 3], tree.predict(train[:, :3])).show()
    # 注意scikit-learn使用predict方法直接给出预测结果

    predict_result = tree.predict_proba(test[:, :3])[:, 1]
    fpr, tpr, thresholds = roc_curve(test[:, 3], predict_result, pos_label=1)

    plt.plot(fpr, tpr, linewidth=2, label='ROC of CART', color='green')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(loc=4)
    plt.show()

    print(thresholds)


def session_4():
    pkl_file = open('tree.pkl', 'rb')
    file_list = joblib.load(pkl_file)
    pkl_file.close()
    print(file_list)

if __name__ == '__main__':
    # session_1()
    # session_2()
    # session_3()
    session_4()