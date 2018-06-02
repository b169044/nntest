#-*- coding: UTF-8 -*-
# https://blog.csdn.net/m0_37650263/article/details/77343220
# https://blog.csdn.net/leiting_imecas/article/details/71246541
#https://www.jianshu.com/p/c474f875fc96
#https://www.cnblogs.com/Newsteinwell/p/6034747.html
#http://www.open-open.com/lib/view/open1444351655682.html
import csv
from gensim.models import word2vec
from pyltp import Segmentor, Postagger
import os
import numpy as np
seg = Segmentor()
seg.load('baidu_analysis/cws.model')
poser = Postagger()
poser.load('baidu_analysis/pos.model')
real_dir_path = os.path.split(os.path.realpath(__file__))[0]  # 文件所在路径
stop_words_file = os.path.join(real_dir_path, 'baidu_analysis/stopwords.txt')
# 定义允许的词性
allow_pos_ltp = ('a', 'i', 'j', 'n', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'v', 'ws')


def cut_stopword_pos(s):
    words = seg.segment(''.join(s.split()))
    poses = poser.postag(words)
    stopwords = {}.fromkeys([line.rstrip() for line in open(stop_words_file, 'r', encoding='UTF-8')])
    sentence = []
    for i, pos in enumerate(poses):
        if (pos in allow_pos_ltp) and (words[i] not in stopwords):
            sentence.append(words[i])
    return sentence

file = os.path.join(real_dir_path, 'baidu_analysis/data_train.csv')
v = []
f2 =open("fenci_result5.txt", 'a',encoding='utf-8')
with open(file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if (line.split('\t')[1] == '物流快递'):
            sentence = cut_stopword_pos(line.split('\t')[2].replace('hellip',''))
            f2.write(" ".join(sentence))
            f2.write(" ")
            v.append(' '.join(sentence))

print(v)
exit()
sentences =word2vec.Text8Corpus("baidu_analysis/fenci_result1.txt")

model = word2vec.Word2Vec(sentences, sg=1, size=100,  window=5,  min_count=1,  negative=3, sample=0.001, hs=1)
print(model)
print(model['烤鸭'])

print(model.most_similar(u"烤鸭"))

# def getWordVecs(wordList):
#     vecs = []
#     for word in wordList:
#         word = word.replace('\n', '')
#         try:
#             # only use the first 500 dimensions as input dimension
#             vecs.append(model[word])
#         except KeyError:
#             continue
#     # vecs = np.concatenate(vecs)
#     return np.array(vecs, dtype = 'float')

def getWordVecs(wordList,size):#获得评论中所有词向量的平均值
    vecs = np.zeros(size).reshape((1,size))
    count = 0
    for word in wordList:
        word = word.replace('\n', '')
        try:
            vecs += model[word].reshape((1,size))
            count +=1
        except KeyError:
            continue
    if count!=0:
        vecs /=count
    return vecs

print(getWordVecs(['买', '系统', '做', '公司', '公众', '号', '平台', '代', '运营', '想到', 'app', '方便', '管理', '找', '久', '程序'],100))





