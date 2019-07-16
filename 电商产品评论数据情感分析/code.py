import pandas as pd
import jieba
from gensim import corpora, models

'''
输出的文件存放在data目录下
session_1           提取数据
session_2           数据去重
session_3           利用正则去除一些数据
session_4           使用jieba分词
session_5           分词后进行语义分析，LDA模型分析正面负面情感
'''


def session_1():
    input_file = 'huizong.csv'
    output_file = 'data/meidi_jd.txt'

    data = pd.read_csv(input_file)
    data = data[['评论']][data['品牌'] == '美的']
    data.to_csv(output_file, index=False)


def session_2():
    input_file = 'data/meidi_jd.txt'
    output_file = 'data/meidi_jd_process_1.txt'

    data = pd.read_csv(input_file, header=None)

    l1 = len(data)
    data = pd.DataFrame(data[0].unique())
    l2 = len(data)
    data.to_csv(output_file, index=False, header=False)

    print('删除了%s条评论...' % (l1 - l2))


def session_3():
    input_file1 = 'meidi_jd_process_end_正面情感结果.txt'
    input_file2 = 'meidi_jd_process_end_负面情感结果.txt'
    output_file1 = 'data/meidi_jd_pos.txt'
    output_file2 = 'data/meidi_jd_neg.txt'

    data1 = pd.read_csv(input_file1, header=None)
    data2 = pd.read_csv(input_file2, header=None)

    data1 = pd.DataFrame(data1[0].str.replace('.*?\d+?\\t', ''))
    data2 = pd.DataFrame(data2[0].str.replace('.*?\d+?\\t', ''))

    data1.to_csv(output_file1, index=False, header=False)
    data2.to_csv(output_file2, index=False, header=False)


def session_4():
    input_file1 = 'data/meidi_jd_pos.txt'
    input_file2 = 'data/meidi_jd_neg.txt'
    output_file1 = 'data/meidi_jd_pos_cut.txt'
    output_file2 = 'data/meidi_jd_neg_cut.txt'

    data1 = pd.read_csv(input_file1, header=None)
    data2 = pd.read_csv(input_file2, header=None)

    def mycut(s):
        return ' '.join(jieba.cut(s))

    data1 = data1[0].apply(mycut)
    data2 = data2[0].apply(mycut)

    data1.to_csv(output_file1, index=False, header=False)
    data2.to_csv(output_file2, index=False, header=False)


def session_5():
    pos_file = 'data/meidi_jd_pos_cut.txt'
    neg_file = 'data/meidi_jd_neg_cut.txt'
    stop_list = 'stoplist.txt'

    pos = pd.read_csv(pos_file, header=None)
    neg = pd.read_csv(neg_file, header=None)

    '''
    sep设置分割词，由于csv默认半角逗号为分割词，而且该词恰好位于停用词表中，所以会导致读取错误
    解决办法是手动设置一个不存在的分割词，这里使用的是tipdm
    参数engine加上指定引擎，避免警告
    '''

    stop = pd.read_csv(stop_list, header=None, sep='tipdm', engine='python')

    # pandas自动过滤了空格，这里手动添加
    step = [' ', ''] + list(stop[0])

    # 定义分割词，然后用apply进行广播
    neg[1] = neg[0].apply(lambda s: s.split(' '))
    neg[2] = neg[1].apply(lambda x: [i for i in x if i not in stop])
    pos[1] = pos[0].apply(lambda s: s.split(' '))
    pos[2] = pos[1].apply(lambda x: [i for i in x if i not in stop])

    # 负面主题分析
    # 建立词库
    neg_dict = corpora.Dictionary(neg[2])
    # 建立语料库
    neg_corpus = [neg_dict.doc2bow(i) for i in neg[2]]
    # LDA模型训练
    neg_lda = models.LdaModel(neg_corpus, num_topics=3, id2word=neg_dict)

    for i in range(3):
        print(neg_lda.print_topic(i))

    # 正面主题分析
    # 建立词库
    pos_dict = corpora.Dictionary(pos[2])
    # 建立语料库
    pos_corpus = [pos_dict.doc2bow(i) for i in pos[2]]
    # LDA模型训练
    pos_lda = models.LdaModel(pos_corpus, num_topics=3, id2word=pos_dict)

    for i in range(3):
        print(pos_lda.print_topic(i))


if __name__ == '__main__':
    # session_1()
    # session_2()
    # session_3()
    # session_4()
    session_5()
