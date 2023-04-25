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
train_path=data_path+"train.txt"
test_path=data_path+"processec.txt"
validation_path=data_path+"validation.txt"
freq_treshold=10
average_len=62
Embedding_size = 60
Batch_Size = 10
Kernel = 10
Filter_num = 10#卷积核的数量。
Epoch = 60
Dropout = 0.5
Learning_rate = 1e-3#学习率
sequence_length = 62#每个句子的长度

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_basic_info():
    # print basic information of the GPU
    print("name of the GPU:", torch.cuda.get_device_name(0))
    print("device:", device)


class TextCNNDataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.LongTensor(data_inputs)
        self.label = torch.LongTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)

# 从文件中读取预处理好的数据
input_path="input/"
with open(os.path.join(input_path,"vocab.txt"), encoding='utf-8') as fin:
    vocab = [i.split('\n')[0] for i in fin]
word2idx = {i:index for index, i in enumerate(vocab)}

word2vec={}
with open(os.path.join(input_path,"word2vec.txt"), encoding='utf-8') as f:
    for line in f:
        line=line.strip().split(" ")
        word2vec[int(line[0].strip())]=[float(i) for i in line[1:]]

for key,value in word2vec.items():
    word2vec[key]=[float(i) for i in value]

with open(os.path.join(input_path,"train_input.txt"), "r", encoding='utf-8') as fin:
    train_data = pd.read_csv(os.path.join(input_path,"train_input.txt"),names=["label","comment"],sep="\t")
    for i in range(len(train_data)):
        comment=train_data["comment"][i].strip().split(" ")
        train_data["comment"][i]=comment
        train_data["comment"][i]=[word2vec[int(j)] for j in train_data["comment"][i]]


with open(os.path.join(input_path,"test_input.txt"), "r", encoding='utf-8') as fin:
    test_data = pd.read_csv(os.path.join(input_path,"test_input.txt"),names=["label","comment"],sep="\t")
    for i in range(len(test_data)):
        test_data["comment"][i]=test_data["comment"][i].strip().split(" ")
        test_data["comment"][i]=[word2vec[int(j)] for j in test_data["comment"][i]]

# with open(os.path.join(input_path,"validation_input.txt"), "r", encoding='utf-8') as fin:
#     validation_data = pd.read_csv(os.path.join(input_path,"validation_input.txt"),names=["label","comment"],sep="\t")
#     for i in range(len(validation_data)):
#         validation_data["comment"][i]=validation_data["comment"][i].strip().split(" ")
#         validation_data["comment"][i]=[word2vec[int(j)] for j in validation_data["comment"][i]]

class TextCNNDataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.LongTensor(data_inputs)
        self.label = torch.LongTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)

train_dataset = TextCNNDataSet(train_data['comment'], list(train_data["label"]))
test_dataset = TextCNNDataSet(test_data['comment'], list(test_data["label"]))
# validation_dataset = TextCNNDataSet(validation_data['comment'], list(validation_data["label"]))

TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
TestDataLoader=Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=True)

#使用word2vec版本的。
num_classs = 2#2分类问题。

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
#               super(TextCNN, self).__init__()
        self.W = nn.Embedding(num_embeddings=8072,embedding_dim=Embedding_size)
        out_channel = Filter_num
        self.conv = nn.Sequential(
                    nn.Conv2d(1, out_channel, (2, Embedding_size)),#卷积核大小为2*Embedding_size
                    nn.ReLU(),
                    nn.MaxPool2d((sequence_length-1,1)),
        )
        self.dropout = nn.Dropout(Dropout)
        self.fc = nn.Linear(out_channel, num_classs)

    def forward(self, X):
        batch_size = X.shape[0]
        #x:batch_size*seq_len
        embedding_X = self.W(X)
        # batch_size, sequence_length, embedding_size
        embedding_X = embedding_X.unsqueeze(1)
        # batch_size, 1,sequence_length, embedding_size
        conved = self.conv(embedding_X)
        #batch_size,10,seq_len-1,1
        #batch_size,10,seq_len-1,1
        #batch_size,10,1,1########直接被maxpooliing了，从一个序列变成一个向量，表示将整个句子选出一个最关键的情感分类词来。
        conved = self.dropout(conved)
        flatten = conved.view(batch_size, -1)
        # [batch_size, 10]
        output = self.fc(flatten)
        #2分类问题，往往使用softmax，表示概率。
        return F.log_softmax(output)

model = TextCNN().to(device)
optimizer = optim.Adam(model.parameters(),lr=Learning_rate)

def binary_acc(pred, y):
    """
    计算模型的准确率
    :param pred: 预测值
    :param y: 实际真实值
    :return: 返回准确率
    """
    correct = torch.eq(pred, y).float()
    acc = correct.sum() / len(correct)
    return acc.item()

def train():
    """
    模型训练
    """
    avg_acc = []
    model.train()
    for index, (batch_x, batch_y) in enumerate(TrainDataLoader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = F.nll_loss(pred, batch_y)
        acc = binary_acc(torch.max(pred, dim=1)[1], batch_y)
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_acc = np.array(avg_acc).mean()
    return avg_acc



def evaluate():
    """
    模型评估
    :param model: 使用的模型
    :return: 返回当前训练的模型在测试集上的结果
    """
    avg_acc = []
    model.eval()  # 进入测试模式
    with torch.no_grad():
        for x_batch, y_batch in TestDataLoader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)#预测值
            acc = binary_acc(torch.max(pred, dim=1)[1], y_batch)
            avg_acc.append(acc)
    return np.array(avg_acc).mean()

def cal_presion(pred, y):
    """
    计算模型的准确率
    :param pred: 预测值
    :param y: 实际真实值
    :return: 返回准确率
    """
    correct = torch.eq(pred, y).float()
    acc = correct.sum() / len(correct)
    return float(acc.item())

def cal_recall(pred, y):
    """
    计算模型的召回率
    :param pred: 预测值
    :param y: 实际真实值
    :return: 返回召回率
    """
    correct = torch.eq(pred, y).float()
    recall = correct.sum() / len(y)
    return float(recall.item())

def f1_score():
    """
    计算模型的F1值
    :return: 返回F1值
    """
    model.eval()  # 进入测试模式
    with torch.no_grad():
        for x_batch, y_batch in TestDataLoader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            presion = cal_presion(torch.max(pred, dim=1)[1], y_batch)
            recall = cal_recall(torch.max(pred, dim=1)[1], y_batch)
            f1 = float(2 * presion * recall / (presion + recall))
    return f1

# Training cycle
model_train_acc, model_test_acc = [], []
model_train_f1=[]
for epoch in range(Epoch):
    train_acc = train()
    test_acc=evaluate()
    train_f1=f1_score()
    print("epoch = {}, 训练准确率={}".format(epoch + 1, train_acc))
    print("epoch = {}, 测试准确率={}".format(epoch + 1, test_acc))
    print("epoch = {}, F1值={}".format(epoch + 1, train_f1))
    model_train_acc.append(train_acc)
    model_test_acc.append(test_acc)
    model_train_f1.append(train_f1)

import matplotlib.pyplot as plt

plt.plot(model_train_acc)
plt.ylim(ymin=0.5, ymax=1)
plt.plot(model_test_acc)
plt.ylim(ymin=0.5, ymax=1)
plt.plot(model_train_f1)
plt.ylim(ymin=0.5, ymax=1)
plt.title("The accuracy of textCNN model")
# 添加图例
plt.legend(['train', 'test','f1'], loc='upper left')
plt.savefig('testCNN.png')






