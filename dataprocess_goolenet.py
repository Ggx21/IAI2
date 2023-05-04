import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import Counter
import os
import json


data_path="Dataset/"
input_path = "input_google/"
train_path=data_path+"train.txt"
test_path=data_path+"test.txt"
validation_path=data_path+"validation.txt"
freq_treshold=10
average_len=96



train_data = pd.read_csv(train_path,names=["label","comment"],sep="\t")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

def print_basic_info():
    # print basic information of the GPU
    print("name of the GPU:", torch.cuda.get_device_name(0))
    print("device:", device)

Embedding_size = 96
Batch_Size = 10
Kernel = 10
Filter_num = 10#卷积核的数量。
Epoch = 60
Dropout = 0.5
Learning_rate = 1e-3

class TextCNNDataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.LongTensor(data_inputs)
        self.label = torch.LongTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)

with open(os.path.join(data_path,"word_freq.txt"), encoding='utf-8') as fin:
    vocab = [i.split("\t")[0] for i in fin]
vocab=set(vocab)
with open(os.path.join(input_path,"vocab.txt"),'w', encoding='utf-8') as fout:
    for i in vocab:
        fout.write(i+"\n")

word2idx = {i:index for index, i in enumerate(vocab)}
with open(os.path.join(input_path,"word2idx.json"), "w", encoding='utf-8') as f:
    json.dump(word2idx, f)
idx2word = {index:i for index, i in enumerate(vocab)}
with open(os.path.join(input_path,"idx2word.json"), "w", encoding='utf-8') as f:
    json.dump(idx2word, f)
vocab_size = len(vocab)
print("vocab_size:",vocab_size)

pad_id=word2idx["把"]
print(pad_id)

sequence_length = 96
#对输入数据进行预处理,主要是对句子用索引表示且对句子进行截断与padding，将填充使用”把“来。

def tokenizer(train_data=train_data):
    inputs = []
    sentence_char = [i.split() for i in train_data["comment"]]
    # 将输入文本进行padding
    for index,i in enumerate(sentence_char):
        temp=[word2idx.get(j,pad_id) for j in i]#表示如果词表中没有这个稀有词，无法获得，那么就默认返回pad_id。
        if len(temp)==0:
            temp=[pad_id]
        while len(temp)<sequence_length:
            temp.extend(temp)
        temp=temp[:sequence_length]
        inputs.append(temp)
    return {"comment":inputs,"label":train_data["label"]}
train_data = tokenizer(train_data)
test_data = pd.read_csv(test_path,names=["label","comment"],sep="\t")
test_data = tokenizer(test_data)
validation_data = pd.read_csv(validation_path,names=["label","comment"],sep="\t")
validation_data = tokenizer(validation_data)

from gensim.models import keyedvectors
w2v=keyedvectors.load_word2vec_format(os.path.join(data_path,"wiki_word2vec_50.bin"),binary=True)

vocab_l={}

for word in vocab:
    word_idx=word2idx[word]
    try:
        vocab_l[word_idx]=w2v[word]
    except KeyError:
        vocab_l[word_idx]=np.random.uniform(-0.25,0.25,50)
    # resize to 96,pad with 0
    # duplicate the vector if the word vector is shorter than 96
    if len(vocab_l[word_idx])<96:
        vocab_list=vocab_l[word_idx].tolist()
        while len(vocab_list)<96:
            vocab_list.extend(vocab_list)
    vocab_l[word_idx]=vocab_list[:96]


with open(os.path.join(input_path,"word2vec.txt"), "w",encoding='utf-8') as fout:
    for key,value in vocab_l.items():
        fout.write(str(key)+"\t")
        for i in value:
            fout.write(" "+str(i))
        fout.write("\n")

word2vec={}
with open(os.path.join(input_path,"word2vec.txt"), encoding='utf-8') as f:
    for line in f:
        line=line.strip().split(" ")
        word2vec[int(line[0].strip())]=[float(i) for i in line[1:]]


for key,value in word2vec.items():
    word2vec[key]=[float(i) for i in value]

import pickle

train_input={}
train_input["label"]=[i for i in train_data["label"]]
train_input["comment"]=[]
for index,i in enumerate(train_data["comment"]):
    temp=[]
    for j in i:
        temp.append(word2vec[j])
    train_input["comment"].append(temp)
with open(os.path.join(input_path,"train_input.pkl"), "wb") as fout:
    pickle.dump(train_input, fout)

validation_input={}
validation_input["label"]=[i for i in validation_data["label"]]
validation_input["comment"]=[]
for index,i in enumerate(validation_data["comment"]):
    temp=[]
    for j in i:
        temp.append(word2vec[j])
    validation_input["comment"].append(temp)
with open(os.path.join(input_path,"validation_input.pkl"), "wb") as fout:
    pickle.dump(validation_input, fout)

test_input={}
test_input["label"]=[i for i in test_data["label"]]
test_input["comment"]=[]
for index,i in enumerate(test_data["comment"]):
    temp=[]
    for j in i:
        temp.append(word2vec[j])
    test_input["comment"].append(temp)
with open(os.path.join(input_path,"test_input.pkl"), "wb") as fout:
    pickle.dump(test_input, fout)