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
model_path="Model/"
train_path=data_path+"train.txt"
test_path=data_path+"processec.txt"
validation_path=data_path+"validation.txt"

# ---------------------------data process parameters---------------------------
freq_treshold=10
average_len=62

# ----------------------------hyperparameters----------------------------
Embedding_size = 50
sequence_length = 62#每个句子的长度
Learning_rate = 1e-4#学习率
num_epochs = 20#训练的轮数
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

# 3. 构建模型
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class TextCNN1(nn.Module):
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
        return F.softmax(output, dim=1)
        # return output

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
        # x shape: (batch_size, channel, seq_len)
        # return shape: (batch_size, channel, 1)
        return F.max_pool1d(x, kernel_size=x.shape[2])

class TextCNN(nn.Module):
    def __init__(self, vocab_len, embed_size, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_len, embed_size)
        # 不参与训练的嵌入层
        self.constant_embedding = nn.Embedding(vocab_len, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = 2*embed_size,
                                        out_channels = c,
                                        kernel_size = k))

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = torch.cat((inputs,inputs), dim=2) # (batch, seq_len, 2*embed_size)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维(即词向量那一维)，变换到前一维
        embeddings = embeddings.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


with open(os.path.join(input_path,"word2vec.txt"), "r",encoding='utf-8') as fin:
    pretrained_embedding = np.array([line.strip().split()[1:] for line in fin.readlines()], dtype="float32")
    pretrained_embedding = torch.from_numpy(pretrained_embedding)
    print(pretrained_embedding.shape)

vocab_len = int(pretrained_embedding.shape[0])
print(vocab_len)

kernel_sizes, nums_channels = [3, 4, 5], [100, 100, 100]
model=TextCNN(vocab_len=vocab_len,embed_size=Embedding_size,kernel_sizes=kernel_sizes,num_channels=nums_channels).to(device)

print("embeddingshape",model.embedding.weight.data.shape)
model.embedding.weight.data.copy_(pretrained_embedding)
model.constant_embedding.weight.data.copy_(pretrained_embedding)

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
    for (batch_x, batch_y) in train_data_loader:
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
    test_acc,test_f1 = evaluate(
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








