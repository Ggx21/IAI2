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

input_path = "input/"

global_word_embedding_size = 50
global_sentence_length = 96


CNN_simple_params = {
    "kernel_num":100,
    "embed_size":global_word_embedding_size,
    "sentence_length":global_sentence_length,
    "dropout":0.5
}

CNN_complex_params = {
    "embed_size":global_word_embedding_size,
    "kernel_sizes":[3, 4, 5],
    "num_channels": [100, 100, 100]
}

LSTM_params = {
    "embed_size": global_word_embedding_size,
    "num_hiddens": 256,
    "num_layers": 4,
    "dropout": 0.5
}

global_params={
    "CNN_simple":CNN_simple_params,
    "CNN_complex":CNN_complex_params,
    "LSTM":LSTM_params,
}


class GetModule():
    def __init__(self, module_name):
        self.module = None
        self.optimizer = None
        self.criterion = None
        self.module_name = module_name
        self.pretrained_embedding = None
        self.vocab_len = 0
        self.get_pretrained_embedding()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_module()
        self.available_modules = ["CNN_simple", "CNN_complex","RNN","DenseNet", "ResNet", "LSTM", "BiLSTM", "GoogLeNet", "MLP","MLP_with_Dropout"]

    def init_module(self):
        if self.module_name == "CNN_simple":
            params = global_params["CNN_simple"]
            self.module = CNN_simple(**params)
        elif self.module_name == "CNN_complex":
            params = global_params["CNN_complex"]
            self.module = CNN_complex(vocab_len=self.vocab_len, **params)
            self.module.embedding.weight.data.copy_(self.pretrained_embedding)
            self.module.constant_embedding.weight.data.copy_(self.pretrained_embedding)
        elif self.module_name == "RNN":
            params = global_params["LSTM"]
            params["num_layers"] = 1
            self.module=LSTM(**params)
        elif self.module_name == "DenseNet":
            self.module=setup_DenseNet()
        elif self.module_name == "ResNet":
            self.module=setup_ResNet()
        elif self.module_name == "LSTM":
            params = global_params["LSTM"]
            self.module=LSTM(**params)
        elif self.module_name == "BiLSTM":
            params = global_params["LSTM"]
            self.module=BiLSTM(**params)
        elif self.module_name == "GoogLeNet":
            self.module=setup_GoogLeNet()
        elif self.module_name == "MLP":
            self.module=MLP()
        elif self.module_name == "MLP_with_Dropout":
            self.module=MLP_with_Dropout()
        else:
            raise Exception("module name error!","module name should be in {}".format(self.available_modules))
        self.module.to(self.device)
        self.optimizer = optim.Adam(self.module.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def get_pretrained_embedding(self):
        import pickle
        # 读取预训练的词向量
        with open(os.path.join(input_path,"word2vec.pkl"), "rb") as fin:
            word2vec = pickle.load(fin)
            pretrained_embedding  = np.array([word2vec[index] for index in range(len(word2vec.keys()))], dtype="float32")
            self.pretrained_embedding = torch.from_numpy(pretrained_embedding)
            print(pretrained_embedding.shape)
            self.vocab_len = int(pretrained_embedding.shape[0])


# ---------------------------CNN_complex----------------------------
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
        # x shape: (batch_size, channel, seq_len)
        # return shape: (batch_size, channel, 1)
        return F.max_pool1d(x, kernel_size=x.shape[2])

class CNN_complex(nn.Module):
    def __init__(self, vocab_len, embed_size, kernel_sizes, num_channels):
        super(CNN_complex, self).__init__()
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

# -----------------------------------------------------------------cnn-simple---------------------------------------------------------------
class CNN_simple(nn.Module):
    def __init__(self, kernel_num, embed_size, sentence_length, dropout):
        super(CNN_simple, self).__init__()
        out_channel = kernel_num
        self.conv = nn.Sequential(
                    nn.Conv2d(1, out_channel, (2, embed_size)),#卷积核大小为2*Embedding_size,默认步长为1
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.MaxPool2d((sentence_length-1,1)),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_channel, 2)

    def forward(self, X):
        batch_size = X.shape[0]
        embedding_X = X.unsqueeze(1)
        conved = self.conv(embedding_X)
        conved = self.dropout(conved)
        flatten = conved.view(batch_size, -1)
        output = self.fc(flatten)
        return F.softmax(output, dim=1)

# --------------------------------------------------------------DenseNet-------------------------------------------------------------------------
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
    return blk

class MyInit(nn.Module):
        def __init__(self,net):
            super(MyInit, self).__init__()
            self.origin_net=net
            self.dropout=nn.Dropout(0.5)
            self.fc=nn.Linear(10,2)
            self.softmax=nn.Softmax(dim=1)
        def forward(self, x):
            x.unsqueeze_(1)
            x=self.origin_net(x)
            x=self.dropout(x)
            x=self.fc(x)
            x=self.softmax(x)
            return x

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
def setup_DenseNet():
    net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        DB = DenseBlock(num_convs, num_channels, growth_rate)
        net.add_module("DenseBlosk_%d" % i, DB)
        # 上一个稠密块的输出通道数
        num_channels = DB.out_channels
        # 在稠密块之间加入通道数减半的过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    net.add_module("BN", nn.BatchNorm2d(num_channels))
    net.add_module("relu", nn.ReLU())
    net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 10)))



    model=MyInit(net)
    return model

#--------------------------------------------------------------BiLSTM-------------------------------------------------------------------------
class BiLSTM(nn.Module):
    """_summary_

    Args:
        embed_size (int): 词向量的维度
        num_hiddens (int): 隐藏层的维度
        num_layers (int): 隐藏层的层数
    """
    def __init__(self, embed_size, num_hiddens, num_layers, dropout=0.5):
        super(BiLSTM, self).__init__()
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                                hidden_size=num_hiddens,
                                num_layers=num_layers,
                                bidirectional=True,
                                dropout=dropout)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        outputs, _ = self.encoder(inputs) # output, (h, c)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs
#--------------------------------------------------------------LSTM-------------------------------------------------------------------------
class LSTM(nn.Module):
    """_summary_

    Args:
        embed_size (int): 词向量的维度
        num_hiddens (int): 隐藏层的维度
        num_layers (int): 隐藏层的层数
    """
    def __init__(self, embed_size, num_hiddens, num_layers,dropout):
        super(LSTM, self).__init__()
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                                hidden_size=num_hiddens,
                                num_layers=num_layers,
                                bidirectional=False,
                                dropout=dropout)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(2*num_hiddens, 2)#双向2*2*num_hiddens，单向2*num_hiddens

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        outputs, _ = self.encoder(inputs) # output, (h, c)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs

#--------------------------------------------------------------GoogLeNet-------------------------------------------------------------------------

# 3. 构建模型
class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出

def setup_GoogLeNet():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                    Inception(256, 128, (128, 192), (32, 96), 64),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                    Inception(512, 160, (112, 224), (24, 64), 64),
                    Inception(512, 128, (128, 256), (24, 64), 64),
                    Inception(512, 112, (144, 288), (32, 64), 64),
                    Inception(528, 256, (160, 320), (32, 128), 128),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                    Inception(832, 384, (192, 384), (48, 128), 128),
                    GlobalAvgPool2d())
    module = nn.Sequential(nn.Upsample(size=(96, 96),mode='bicubic', align_corners=True),b1, b2, b3, b4, b5,
                    FlattenLayer(), nn.Linear(1024, 10))

    net=MyInit(module)
    return net

# --------------------------------------------------------------ResNet-------------------------------------------------------------------------
# 3. 构建模型
class Residual(nn.Module):
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
def setup_ResNet():
    net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))
    module=MyInit(net)
    return module

# --------------------------------------------------------------MLP-------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(global_word_embedding_size*global_sentence_length, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Sigmoid(),
            nn.Linear(10, 2)
        )
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

# --------------------------------------------------------------MLP-------------------------------------------------------------------------

class MLP_with_Dropout(nn.Module):
    def __init__(self):
        super(MLP_with_Dropout, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(global_word_embedding_size*global_sentence_length, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(10, 2)
        )
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))