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
Learning_rate = 1e-3#学习率
num_epochs = 100#训练的轮数
Batch_Size = 16#批处理的大小
Filter_num = 20#卷积核的数量
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
with open(os.path.join(input_path,"train_input.json"), "r", encoding='utf-8') as fin:
    train_data = json.load(fin)

with open(os.path.join(input_path,"test_input.json"), "r", encoding='utf-8') as fin:
    test_data = json.load(fin)
with open(os.path.join(input_path,"validation_input.json"), "r", encoding='utf-8') as fin:
    val_data = json.load(fin)

class MyDataset(Dataset):
    def __init__(self, data_set):
        self.inputs = []
        self.label = []
        for i in data_set:
            self.inputs.append(i['comment'])
            self.label.append(i['label'])
        self.inputs = torch.FloatTensor(self.inputs)
        self.label = torch.LongTensor(self.label)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


# 2. 数据集的加载
train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)
val_dataset = MyDataset(val_data)

train_data_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=True)

# 3. 构建模型
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        out_channel = Filter_num
        self.conv = nn.Sequential(
                    nn.Conv2d(1, out_channel, (2, Embedding_size)),#卷积核大小为2*Embedding_size,默认步长为1
                    nn.ReLU(),
                    nn.MaxPool2d((sequence_length-1,1)),
        )
        self.dropout = nn.Dropout(Dropout)
        self.fc = nn.Linear(out_channel, 2)

    def forward(self, X):
        batch_size = X.shape[0]
        embedding_X = X.unsqueeze(1)
        conved = self.conv(embedding_X)
        conved = self.dropout(conved)
        flatten = conved.view(batch_size, -1)
        output = self.fc(flatten)
        return F.log_softmax(output)



model=TextCNN().to(device)

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
    avg_acc = []
    model.train()
    for index, (batch_x, batch_y) in enumerate(train_data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_size = batch_x.size(0)
        if batch_size != Batch_Size:
            continue
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
        for x_batch, y_batch in test_data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            acc = binary_acc(torch.max(pred, dim=1)[1], y_batch)
            avg_acc.append(acc)
    return np.array(avg_acc).mean()

# Training cycle
model_train_acc, model_test_acc = [], []
for epoch in range(num_epochs):
    train_acc = train()
    print("epoch = {}, 训练准确率={}".format(epoch + 1, train_acc))
    model_train_acc.append(train_acc)
    test_acc = evaluate()
    print("epoch = {}, 测试准确率={}".format(epoch + 1, test_acc))
    model_test_acc.append(test_acc)

import matplotlib.pyplot as plt
plt.plot(model_train_acc)
plt.ylim(ymin=0.5, ymax=1)
plt.plot(model_test_acc)
plt.ylim(ymin=0.5, ymax=1)
plt.legend(["train_acc", "test_acc"])
plt.title("The accuracy of textCNN model")
plt.show()
plt.savefig("The accuracy of textCNN model.png")

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

import matplotlib.pyplot as plt

plt.plot(model_loss)
plt.ylim(ymin=0, ymax=0.5)
plt.plot(model_accuracy)
plt.ylim(ymin=0.5, ymax=1)
plt.title("The accuracy of textCNN model")
# 添加图例
plt.legend(['loss','acc'], loc='upper left')
plt.savefig('testCNN.png')




