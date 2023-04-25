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

with open(os.path.join(input_path,"validation_input.txt"), "r", encoding='utf-8') as fin:
    val_data = pd.read_csv(os.path.join(input_path,"validation_input.txt"),names=["label","comment"],sep="\t")
    for i in range(len(val_data)):
        val_data["comment"][i]=val_data["comment"][i].strip().split(" ")
        val_data["comment"][i]=[word2vec[int(j)] for j in val_data["comment"][i]]


#使用word2vec版本的。
num_classs = 2#2分类问题。


# 2. 数据拆分
train_inputs = torch.FloatTensor(train_data['comment'])
train_labels = torch.FloatTensor(train_data['label'])
test_inputs = torch.FloatTensor(test_data['comment'])
test_labels = torch.FloatTensor(test_data['label'])
val_inputs = torch.FloatTensor(val_data['comment'])
val_labels = torch.FloatTensor(val_data['label'])


class CommentDataset(Dataset):
    def __init__(self, data_input,data_label):
        self.data = {'comment': data_input, 'label': data_label}
        print("data_input_shape:",data_input.shape)
        print("data_label_shape:",data_label.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        comment = torch.tensor(self.data['comment'][index], dtype=torch.float32)
        print("comment_shape:",comment.shape)
        label = torch.tensor(self.data['label'][index], dtype=torch.long)
        return comment, label

# 构建训练、验证和测试集

train_dataset = CommentDataset(train_inputs,train_labels)
val_dataset = CommentDataset(val_inputs,val_labels)
test_dataset = CommentDataset(test_inputs,test_labels)

# 使用DataLoader加载数据
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 构建模型
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(16*30, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = x.view(-1, 16*30)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

input_dim = 50*62  # 输入特征的维度（50维向量的总数）
output_dim = 2     # 输出的数量（即类别的数量）
# model = LinearModel(input_dim, output_dim)
model=MyModel(input_dim,output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        # 前向传递
        inputs, labels = batch
        print("input.shape:",inputs.shape)
        # 改变形状以适应模型
        inputs = inputs.view(-1, 50, 62).transpose(1, 2)
        print("tranposed_input.shape:",inputs.shape)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传递和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 验证模式
    model.eval()
    val_accuracy = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs, labels = batch
            inputs = inputs.view(-1, 50, 62).transpose(1, 2)  # 改变形状以适应模型

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            val_accuracy += (predicted == labels).sum().item() / labels.size(0)

    train_loss /= len(train_dataloader)
    val_accuracy /= len(val_dataloader)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Accuracy: {:.2f}%'
          .format(epoch+1, num_epochs, train_loss, val_accuracy*100))

# 测试模型
test_accuracy = 0.0
model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels = batch
        inputs = inputs.view(-1, 50, 62).transpose(1, 2)  # 改变形状以适应模型

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        test_accuracy += (predicted == labels).sum().item() / labels.size(0)

test_accuracy /= len(test_dataloader)
print('Test Accuracy: {:.2f}%'.format(test_accuracy*100))





