import pickle

import pandas as pd
from numpy.random import shuffle
from sklearn import metrics
from sklearn.svm import SVC


def main():
    data = pd.read_csv('moment.csv', encoding='gbk')
    data = data.values

    # 随机抽取训练集和验证集
    shuffle(data)
    data_train = data[:int(0.8 * len(data)), :]
    data_test = data[int(0.8 * len(data)):, :]

    # 训练集/验证集的训练数据和结果数据的抽取
    x_train = data_train[:, 2:] * 30
    y_train = data_train[:, 0].astype(int)
    x_test = data_test[:, 2:] * 30
    y_test = data_test[:, 0].astype(int)

    # 训练支持向量机的SVC
    model = SVC(gamma='auto')  # 这里有警告，加上gamma
    model.fit(x_train, y_train)

    # 保存模型
    pickle.dump(model, open('svm.model', 'wb'))

    # 混淆矩阵，评估模型的准确性
    cm_train = metrics.confusion_matrix(y_train, model.predict(x_train))
    cm_test = metrics.confusion_matrix(y_test, model.predict(x_test))
    print(cm_train, '\n', cm_test)

    # 保存训练数据集
    pd.DataFrame(
        cm_train, index=list(range(1, 6)),
        columns=list(range(1, 6))
    ).to_csv('cm_train.csv')

    # 保存测试数据集
    pd.DataFrame(
        cm_test, index=list(range(1, 6)),
        columns=list(range(1, 6))
    ).to_csv('cm_test.csv')


# 读取model文件
def file_load():
    model_file = open('svm.model','rb')
    file_list = pickle.load(model_file)
    model_file.close()
    print(file_list)


if __name__ == '__main__':
    main()
    file_load()