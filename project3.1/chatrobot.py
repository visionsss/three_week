# %% 语料库预处理
from gensim.models import Word2Vec
from tkinter import _flatten
import jieba
import os
from glob import glob

# 读取语料库文件
corpus = []
for file in glob(r'D:\Note\大四\three_week\project3.1\dialog/*.txt'):
    with open(file, 'r', encoding='utf-8') as f:
        corpus.extend(f.readlines())
corpus = [i.replace('\n', '') for i in corpus]
print(corpus)

# 分词
jieba.load_userdict(r'D:\Note\大四\three_week\project3.1\mydict.txt')
corpus_cut = [jieba.lcut(i) for i in corpus]
tmp = _flatten(corpus_cut)
all_dict = ['_BOS', '_ROS', 'PAD', '_UNK'] + list(set(tmp))
id2word = {j: i for i, j in enumerate(all_dict)}

# 语料转为id向量
ids = [[id2word.get(i, id2word['_UNK']) for i in w] for w in corpus_cut]
# 拆分source与target
fromids = ids[::2]
toids = ids[1::2]

# 词向量训练
emb_size = 50  # 词向量大小
tmp = [list(map(str, i)) for i in ids]
emb_path = r'D:\Note\大四\three_week\project3.1\tmp\word2vec.model'
if not os.path.exists(emb_path):
    model = Word2Vec(tmp, size=emb_size, window=5, min_count=1)
    model.save(emb_path)
else:
    print('模型已存在')

# 保存文件fromids toids all_dict
# import json
# with open(r'D:\Note\大四\three_week\project3.1\ids\ids.json', 'w', encoding='utf-8') as f:
#     f.writelines([''])
# %% 模型计算图构建


# 文件读取
import json

with open(r'D:\Note\大四\three_week\project3.1\ids\ids.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
# all_dict = d['dict']
fromids = d['fromids']
toids = d['toids']

# 词向量矩阵
from gensim.models import Word2Vec

model = Word2Vec.load(r'D:\Note\大四\three_week\project3.1\tmp\word2vec.model')
emb_size = model.layer1_size
print(emb_size)
import numpy as np

vocab_size = len(all_dict)
embedding_matrix = np.zeros((vocab_size, emb_size))
tmp = np.diag([1] * emb_size)
