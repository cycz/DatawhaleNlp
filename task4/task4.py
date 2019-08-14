import pandas as pd
import numpy as np
from tqdm import tqdm

'''
利用单层textcnn 做的一次情感分析
成功率90+
'''

tqdm.pandas(desc='pandas')

maxlen = 200  # 截断词数
min_count = 3  # 出现次数少于该值的词扔掉

pos = pd.read_pickle('./preData/pos.pkl')
neg = pd.read_pickle('./preData/neg.pkl')

neg['label'] = 0
pos['label'] = 1

all_ = neg.append(pos, ignore_index=True)

stopset = {}


def getstopword():
    stop_words = list()
    stop_f = open('./data/stopword.txt', "r", encoding='gbk')
    for line in stop_f.readlines():
        line = line.strip()
        if not len(line):
            continue
        stop_words.append(line)
        stopset[line] = 1
    stop_f.close
    return stop_words


getstopword()


def mycut(s):
    bufencitemp = ''
    for _a in list(s):
        try:
            if stopset[_a] == 1:
                continue
        except BaseException as e:
            bufencitemp += _a
    result = list(bufencitemp)
    return result


all_['words'] = all_[0].apply(lambda s: mycut(s))  # 调用mycut分词
content = []
for i in all_['words']:
    content.extend(i)

abc = pd.Series(content).value_counts()
abc = abc[abc >= min_count]
abc[:] = list(range(1, len(abc) + 1))
abc[''] = 0  # 添加空字符串用来补全
word_set = set(abc.index)


#
def doc2num(s, maxlen):
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + [''] * max(0, maxlen - len(s))
    return list(abc[s])


def getxy(all_):
    print('开始转换训练编码集')
    all_['doc2num'] = all_[0].progress_apply(lambda s: doc2num(mycut(s), maxlen))

    idx = list(range(len(all_)))
    np.random.shuffle(idx)
    all_ = all_.loc[idx]
    pd.to_pickle(all_, "doc2num.pkl")
    x = np.array(list(all_['doc2num']), dtype=np.int32)
    y = np.array(list(all_['label']))
    y = y.reshape((-1, 1))
    return x, y


x, y = getxy(all_)

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout


def baseline_model(max_features, embedding_dims, filters):
    kernel_size = 3

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims))  # 使用Embedding层将每个词编码转换为词向量
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # 池化
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))  # 第一个参数units: 全连接层输出的维度，即下一层神经元的个数。
    model.add(Dropout(0.2))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    return model


model = baseline_model(len(abc) + 1, 256, 256)
history = model.fit(x, y, batch_size=20,
                    epochs=2,
                    # validation_data=(x_val, y_val),
                    verbose=1)

model.save('model.h5')


def readFile(filepath):
    x = ''
    with open(filepath, 'r', encoding='UTF-8') as f:
        for line in f:
            line = str(line).replace('\n', "")
            x += line;
        x = x.strip().replace('\n', '').replace('\t', '').replace(' ', '').replace(' ', '')
    return x


def predict_one(s):  # 单个句子的预测函数
    s = np.array(doc2num(mycut(s), maxlen))
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]


while (1):
    print('Please input your sentence:')
    demo_sent = input()
    text = readFile('test.txt')
    if demo_sent == '' or demo_sent.isspace():
        print('See you next time!')
        break
    else:
        print(predict_one(text))
