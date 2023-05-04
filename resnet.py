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
import random


data_path="Dataset/"
model_path="g_Model/"
train_path=data_path+"train.txt"
test_path=data_path+"processec.txt"
validation_path=data_path+"validation.txt"

# ---------------------------data process parameters---------------------------
freq_treshold=10
average_len=96

# ----------------------------hyperparameters----------------------------
Embedding_size = 96
sequence_length = 96#每个句子的长度
Learning_rate = 1e-5#学习率
num_epochs = 1#训练的轮数
Batch_Size = 16#批处理的大小
Filter_num = 100#卷积核的数量
Dropout = 0.2#dropout的大小

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(3407)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_basic_info():
    # print basic information of the GPU
    print("name of the GPU:", torch.cuda.get_device_name(0))
    print("device:", device)

print_basic_info()


# 从文件中读取预处理好的数据
input_path="input"
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

# 3. 构建模型
class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))
class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10),nn.Sigmoid(),nn.Linear(10,2)))

model=net.to(device)

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
    from tqdm import tqdm
    avg_acc = []
    f1_score_list = []
    model.train()
    for batch_x, batch_y in tqdm(train_data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_size = batch_x.size(0)
        # 添加channel维度
        batch_x = batch_x.unsqueeze(1)
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
    from tqdm import tqdm
    f1_score_list = []
    avg_acc = []
    model.eval()  # 进入测试模式
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_data_loader):
            x_batch = x_batch.unsqueeze(1)
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

# save model
torch.save(model.state_dict(), os.path.join(model_path, "cnn_model.pkl"))


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








