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

#使用word2vec版本的。
num_classs = 2#2分类问题。


class MyDataset(Dataset):
    def __init__(self, data_set):
        self.inputs = []
        self.label = []
        for i in data_set:
            self.inputs.append(i['comment'])
            self.label.append(i['label'])
        self.inputs = torch.FloatTensor(self.inputs)
        self.label = torch.FloatTensor(self.label)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


# 2. 数据拆分
train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)
val_dataset = MyDataset(val_data)




# train_inputs = torch.FloatTensor(train_data['comment'])
# train_labels = torch.FloatTensor(train_data['label'])
# test_inputs = torch.FloatTensor(test_data['comment'])
# test_labels = torch.FloatTensor(test_data['label'])
# val_inputs = torch.FloatTensor(val_data['comment'])
# val_labels = torch.FloatTensor(val_data['label'])

# 使用DataLoader加载数据
batch_size = 32


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
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2,padding=1)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = x.reshape(-1, input_dim)
        x = x.unsqueeze(-1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x)



input_dim = 50*62  # 输入特征的维度（50维向量的总数）
output_dim = 2     # 输出的数量（即类别的数量）
# model = LinearModel(input_dim, output_dim)
model=MyModel(input_dim,output_dim).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    train_loss = 0.0
    from tqdm import tqdm
    for i in tqdm(range(len(train_data))):
        # 前向传递
        inputs, labels = train_dataset[i]
        label=labels.long()
        # convert label to 1d tensor
        label = label.view(-1)
        # 改变形状以适应模型

        outputs = model(inputs.to(device))
        loss = criterion(outputs.to(device), label.to(device))

        # 反向传递和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 验证模式
    model.eval()
    val_accuracy = 0.0
    with torch.no_grad():
        for i in tqdm(range(len(val_data))):
            inputs, labels = val_dataset[i]
            inputs = inputs.view(-1, 50, 62).transpose(1, 2)  # 改变形状以适应模型

            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            # get the value of predicted as a int

            predicted = predicted.item()
            predicted=float(predicted)
            #get the value of labels as a int
            labels=labels.long()
            labels=labels.item()
            labels=float(labels)

            # val_accuracy += (predicted == labels).sum().item()/len(val_data)
            val_accuracy += (predicted == labels)

    train_loss /= len(train_dataset)
    val_accuracy /= len(val_dataset)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Accuracy: {:.2f}%'
          .format(epoch+1, num_epochs, train_loss, val_accuracy*100))

# 测试模型
test_accuracy = 0.0
model.eval()
with torch.no_grad():
    for i in range(len(test_data)):
        inputs, labels = test_dataset[i]
        inputs = inputs.view(-1, 50, 62).transpose(1, 2)  # 改变形状以适应模型

        outputs = model(inputs.to(device))
        _, predicted = torch.max(outputs.data, 1)

        test_accuracy += (predicted == labels).sum().item()/len(test_data)

test_accuracy /= len(test_dataset)
print('Test Accuracy: {:.2f}%'.format(test_accuracy*100))





