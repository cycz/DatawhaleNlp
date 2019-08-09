'''
由于时间的关系只实现了python代码。
后续将其他补充
'''
# import modules & set up logging
import logging
import os
from gensim.models import word2vec
import jieba
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(content)
                    labels.append(label)
            except:
                pass
    return contents, labels


def predata():
    alltext, label = read_file('cnews.val.txt')
    # 词与词需要用空格隔开
    traintext = ''
    for _a in tqdm(alltext):
        cutword = jieba.cut(_a)
        for _b in cutword:
            traintext += _b
            traintext += ' '
    with open('resultword.txt', 'w', encoding='utf-8') as f:
        f.write(traintext)


def train():
    if not os.path.exists('resultword.txt'):  # 判断文件是否存在
        predata()
    else:
        print('已有分词的文本')
    sentences = word2vec.LineSentence('resultword.txt')

    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=100)

    model.save('word2vec.model')


if not os.path.exists('word2vec.model'):  # 判断文件是否存在
    train()
else:
    print('此训练模型已经存在，不用再次训练')

model_1 = word2vec.Word2Vec.load('word2vec.model')
# 计算两个词的相似度/相关程度
y1 = model_1.similarity("韦德", "詹姆斯")
print(u"詹姆斯和韦德的相似度为：", y1)
print("-------------------------------\n")

# 计算某个词的相关词列表
y2 = model_1.most_similar("韦德", topn=10)  # 10个最相关的
print(u"和韦德最相关的词有：\n")
for item in y2:
    print(item[0], item[1])
print("-------------------------------\n")
