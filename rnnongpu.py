import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

# ---------------------------data process parameters---------------------------
freq_treshold=10
average_len=62

# ----------------------------hyperparameters----------------------------
Embedding_size = 50
sequence_length = 62#每个句子的长度
Learning_rate = 1e-5#学习率
num_epochs = 50#训练的轮数
Batch_Size = 16#批处理的大小
Filter_num = 100#卷积核的数量
Dropout = 0.5#dropout的大小



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_basic_info():
    # print basic information of the GPU
    print("name of the GPU:", torch.cuda.get_device_name(0))
    print("device:", device)

print_basic_info()


# 从文件中读取预处理好的数据
input_path="input/"
with open(os.path.join(input_path,"vocab.txt"), encoding='utf-8') as fin:
    vocab = [i.split('\n')[0] for i in fin]
word2idx = {i:index for index, i in enumerate(vocab)}

# 读取预处理好的数据
import pickle
with open(os.path.join(input_path,"train_input.pkl"), "rb") as fin:
    train_data = pickle.load(fin)
with open(os.path.join(input_path,"test_input.pkl"), "rb") as fin:
    test_data = pickle.load(fin)
with open(os.path.join(input_path,"validation_input.pkl"), "rb") as fin:
    val_data = pickle.load(fin)

class MyDataset(Dataset):
    def __init__(self, data_set):
        self.comment = torch.FloatTensor(data_set["comment"])
        self.label = torch.LongTensor(data_set["label"])

    def __getitem__(self, index):
        return self.comment[index], self.label[index]

    def __len__(self):
        return len(self.label)


# 2. 数据集的加载
train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)
val_dataset = MyDataset(val_data)

train_data_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

class BiRNN(nn.Module):
    """_summary_

    Args:
        embed_size (int): 词向量的维度
        num_hiddens (int): 隐藏层的维度
        num_layers (int): 隐藏层的层数
    """
    def __init__(self, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                                hidden_size=num_hiddens,
                                num_layers=num_layers,
                                bidirectional=True,
                                dropout=0.5)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        # embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        # 将input的前两维交换
        inputs = inputs.permute(1, 0, 2)
        outputs, _ = self.encoder(inputs) # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs



model=BiRNN(embed_size=Embedding_size,num_hiddens=256,num_layers=4).to(device)
print(model)
for name, param in model.named_parameters():
    print(name, param.shape)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)

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
    from sklearn.metrics import f1_score
    avg_acc = []
    f1_score_list = []
    model.train()
    from tqdm import tqdm
    for batch_x, batch_y in tqdm(train_data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_size = batch_x.size(0)
        if batch_size != Batch_Size:
            continue
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        acc = binary_acc(torch.max(pred, dim=1)[1], batch_y)
        avg_acc.append(acc)
        f1=f1_score(batch_y.cpu(),torch.max(pred, dim=1)[1].cpu(),average='macro')
        f1_score_list.append(f1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return  np.array(avg_acc).mean() ,np.array(f1_score_list).mean()

def evaluate():
    """
    模型评估
    :param model: 使用的模型
    :return: 返回当前训练的模型在测试集上的结果
    """
    from sklearn.metrics import f1_score
    f1_score_list = []
    avg_acc = []
    model.eval()  # 进入测试模式
    with torch.no_grad():
        for x_batch, y_batch in test_data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            # print("pred:",torch.max(pred, dim=1)[1])
            acc = binary_acc(torch.max(pred, dim=1)[1], y_batch)
            avg_acc.append(acc)
            f1=f1_score(torch.max(pred, dim=1)[1].cpu(), y_batch.cpu(), average='macro')
            f1_score_list.append(f1)
    return np.array(avg_acc).mean(),np.array(f1_score_list).mean()

# Training cycle
model_train_acc, model_test_acc = [], []
model_train_f1, model_test_f1 = [], []

for epoch in range(num_epochs):
    train_acc,train_f1 = train()
    print("epoch = {}, 训练准确率={},训练f值={}".format(epoch + 1, train_acc,train_f1))
    model_train_acc.append(train_acc)
    model_train_f1.append(train_f1)
    test_acc,test_f1 = evaluate()
    print("epoch = {}, 测试准确率={},测试f值{}".format(epoch + 1, test_acc,test_f1))
    model_test_acc.append(test_acc)
    model_test_f1.append(test_f1)

from gensim.models import keyedvectors
w2v=keyedvectors.load_word2vec_format(os.path.join(data_path,"wiki_word2vec_50.bin"),binary=True)


# -----------------------------------测试模型-----------------------------------
import jieba
user_input = ""
while user_input != "exit":
    user_input = input("请输入一句话,输入exit退出：\n")
    input_1 = jieba.lcut(user_input)
    for word in input_1:
        if word not in w2v or word == " ":
            input_1.remove(word)
    print("分词结果：", input_1)
    if len(input_1) == 0:
        print("输入的句子没有一个词在词向量中，请重新输入！")
        continue
    while (len(input_1) < sequence_length):
        # input_1 duplicate itself
        input_1.extend(input_1)
    if len(input_1) > sequence_length:
        input_1 = input_1[:sequence_length]
    input_1 = torch.tensor([w2v[word] for word in input_1]).unsqueeze(0).to(device)
    pred = model(input_1)
    pred = torch.max(pred, dim=1)[1]
    if pred == 1:
        print("这是一句负面评价！")
    else:
        print("这是一句正面评价！")



import matplotlib.pyplot as plt
plt.plot(model_train_acc)
plt.ylim(ymin=0.5, ymax=1)
plt.plot(model_test_acc)
plt.ylim(ymin=0.5, ymax=1)
plt.legend(["train_acc", "test_acc"])
plt.title("The accuracy of textCNN model")
plt.show()
plt.savefig("The accuracy of textCNN model.png")
plt.close()

plt.plot(model_train_f1)
plt.ylim(ymin=0.5, ymax=1)
plt.plot(model_test_f1)
plt.ylim(ymin=0.5, ymax=1)
plt.legend(["train_acc", "test_acc"])
plt.title("The f of textCNN model")
plt.show()
plt.savefig("The f of textCNN model.png")
plt.close()





# 训练模型
model_accuracy = []
model_loss = []
for epoch in range(num_epochs):
    avg_acc = []
    avg_loss = []
    # 训练模式
    model.train()
    train_loss = 0.0
    from tqdm import tqdm
    for inputs, labels in tqdm(train_data_loader):
        inputs =inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs.to(device))
        loss = F.nll_loss(outputs, labels)
        # 反向传递和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        avg_loss.append(loss.item())

    # 验证模式
    model.eval()
    val_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in val_data_loader:
            inputs =inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs.to(device))
            acc= binary_acc(torch.max(outputs, dim=1)[1], labels)
            avg_acc.append(acc)


    train_loss = np.array(avg_loss).mean()
    val_accuracy = np.array(avg_acc).mean()


    # 测试模型
    test_accuracy = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_data_loader:
                inputs =inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs.to(device))
                acc= binary_acc(torch.max(outputs, dim=1)[1], labels)
                avg_acc.append(acc)

    test_accuracy = np.array(avg_acc).mean()
    model_accuracy.append(test_accuracy)
    model_loss.append(train_loss)
    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Accuracy: {:.2f}%, Test Accuracy: {:.2f}%.'
          .format(epoch+1, num_epochs, train_loss, val_accuracy*100, test_accuracy*100))









