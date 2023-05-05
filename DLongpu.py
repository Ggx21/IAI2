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
from tqdm import tqdm

from module import GetModule

# ----------------------------hyperparameters----------------------------



class MyDataset(Dataset):
    def __init__(self, data_set):
        temp_array = np.array(data_set["comment"])
        self.comment = torch.FloatTensor(temp_array)
        self.label = torch.LongTensor(data_set["label"])

    def __getitem__(self, index):
        return self.comment[index], self.label[index]

    def __len__(self):
        return len(self.label)


class PredictClass():
    def __init__(self, device,module_name,num_epochs=20, batch_size=64, learning_rate=1e-4, sequence_length=96):
        self.num_epochs = num_epochs#训练的轮数
        self.sequence_length = sequence_length#每个句子的长度
        self.learning_rate = learning_rate#学习率
        self.Batch_Size = batch_size#批处理的大小
        self.device = device
        self.module_name = module_name
        self.get_module = None
        self.module = None
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        # self.module.load_state_dict(torch.load(os.path.join(module_path, "cnn_module.pkl")))
        # self.module.eval()
        from gensim.models import keyedvectors
        self.w2v=keyedvectors.load_word2vec_format(os.path.join("Dataset","wiki_word2vec_50.bin"),binary=True)
        self.train_data_loader = None
        self.test_data_loader = None
        self.val_data_loader = None
        self.test=0
        self.module_train_acc, self.module_test_acc = [], []
        self.module_train_f1, self.module_test_f1 = [], []
        self.module_train_loss, self.module_test_loss = [], []

    def init_from_get_module(self):
        self.get_module = GetModule(self.module_name)
        self.module = self.get_module.module
        self.optimizer = optim.Adam(self.module.parameters(), lr=self.learning_rate)

    def read_data(self):
        """从文件中读取数据"""
        input_path="input"
        import pickle
        with open(os.path.join(input_path,"train_input.pkl"), "rb") as fin:
            train_data = pickle.load(fin)
        with open(os.path.join(input_path,"test_input.pkl"), "rb") as fin:
            test_data = pickle.load(fin)
        with open(os.path.join(input_path,"validation_input.pkl"), "rb") as fin:
            val_data = pickle.load(fin)
        train_dataset = MyDataset(train_data)
        test_dataset = MyDataset(test_data)
        val_dataset = MyDataset(val_data)
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.Batch_Size, shuffle=True)
        self.test_data_loader = DataLoader(test_dataset, batch_size=self.Batch_Size, shuffle=True)
        self.val_data_loader = DataLoader(val_dataset, batch_size=self.Batch_Size, shuffle=True)

    def train(self):
        from sklearn.metrics import f1_score
        avg_acc = []
        f1_score_list = []
        avg_loss = []
        self.module.train()
        for batch_x, batch_y in tqdm(self.train_data_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_size = batch_x.size(0)
            if batch_size != self.Batch_Size:
                continue
            pred = self.module(batch_x)
            loss = self.criterion(pred, batch_y)
            # # 为Loss添加L1正则化项
            # L1_reg = 0
            # for param in self.module.parameters():
            #     L1_reg += torch.sum(torch.abs(param))
            # loss += 0.001 * L1_reg  # lambda=0.001
            acc = self.binary_acc(torch.max(pred, dim=1)[1], batch_y)
            avg_acc.append(acc)
            f1=f1_score(batch_y.cpu(),torch.max(pred, dim=1)[1].cpu(),average='macro')
            f1_score_list.append(f1)
            avg_loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return  np.array(avg_acc).mean() ,np.array(f1_score_list).mean(), np.array(avg_loss).mean()

    def evaluate(self, test=True):
        from sklearn.metrics import f1_score
        f1_score_list = []
        avg_acc = []
        self.module.eval()  # 进入测试模式
        if test:
            data_loader = self.test_data_loader
        else:
            data_loader = self.val_data_loader
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = self.module(x_batch)
                # print("pred:",torch.max(pred, dim=1)[1])
                acc = self.binary_acc(torch.max(pred, dim=1)[1], y_batch)
                avg_acc.append(acc)
                f1=f1_score(torch.max(pred, dim=1)[1].cpu(), y_batch.cpu(), average='macro')
                f1_score_list.append(f1)
        return np.array(avg_acc).mean(),np.array(f1_score_list).mean()

    def binary_acc(self,pred, y):
        """
        计算模型的准确率
        :param pred: 预测值
        :param y: 实际真实值
        :return: 返回准确率
        """
        correct = torch.eq(pred, y).float()
        acc = correct.sum() / len(correct)
        return acc.item()

    def training_cycle(self):
        # Training cycle
        for epoch in range(self.num_epochs):
            train_acc,train_f1 ,train_loss= self.train()
            print("epoch = {}, 训练准确率={},训练f值={},训练loss值={},".format(epoch + 1, train_acc,train_f1,train_loss))
            self.module_train_acc.append(train_acc)
            self.module_train_f1.append(train_f1)
            self.module_train_loss.append(train_loss)
            test_acc,test_f1 = self.evaluate(test=False)#用验证集测试模型
            print("epoch = {}, 验证集准确率={},验证集f值{}".format(epoch + 1, test_acc,test_f1))
            self.module_test_acc.append(test_acc)
            self.module_test_f1.append(test_f1)

    def real_test(self):
        # -----------------------------------测试模型-----------------------------------
        import jieba
        user_input = ""
        while user_input != "exit":
            user_input = input("请输入一句话,输入exit退出：\n")
            input_1 = jieba.lcut(user_input)
            for word in input_1:
                if word not in self.w2v or word == " ":
                    input_1.remove(word)
            print("分词结果：", input_1)
            if len(input_1) == 0:
                print("输入的句子没有一个词在词向量中，请重新输入！")
                continue
            while (len(input_1) < self.sequence_length):
                # input_1 duplicate itself
                input_1.extend(input_1)
            if len(input_1) > self.sequence_length:
                input_1 = input_1[:self.sequence_length]
            input_1 = torch.tensor([self.w2v[word] for word in input_1]).unsqueeze(0).to(device)
            pred = self.module(input_1)
            pred = torch.max(pred, dim=1)[1]
            if pred == 1:
                print("这是一句负面评价！")
            else:
                print("这是一句正面评价！")

    def plot_for_single(self):
        if not os.path.exists("plot"):
            os.mkdir("plot")
        import matplotlib.pyplot as plt
        plt.plot(self.module_train_acc)
        plt.ylim(ymin=0.5, ymax=1)
        plt.plot(self.module_test_acc)
        plt.ylim(ymin=0.5, ymax=1)
        plt.legend(["train_acc", "val_acc"])

        plt.axhline(y=0.8, color='r', linestyle='--')
        title="The accuracy of "+self.module_name+" module"
        plt.title(title)
        plt.savefig("plot/"+title+".png")
        plt.close()

        plt.plot(self.module_train_f1)
        plt.ylim(ymin=0.5, ymax=1)
        plt.plot(self.module_test_f1)
        plt.ylim(ymin=0.5, ymax=1)
        # 设定x轴的坐标
        # plt.xticks(range(0, , 5))
        plt.legend(["train_acc", "val_acc"])

        plt.axhline(y=0.8, color='r', linestyle='--')
        title = "The f of " + self.module_name + " module"
        plt.title(title)
        plt.savefig("plot/"+title+".png")
        plt.close()

        plt.plot(self.module_train_loss)
        title = "The loss of " + self.module_name + " module"
        plt.title(title)
        plt.savefig("plot/"+title+".png")
        plt.close()

    def run_from_zero(self):
        self.init_from_get_module()
        self.read_data()
        self.training_cycle()
        test_acc,test_f1 = self.evaluate()
        print("测试模式, 测试准确率={},测试f值{}".format(test_acc,test_f1))
        self.save_module()
        self.plot_for_single()

    def run_from_load_module(self):
        self.load_module()
        self.read_data()
        test_acc,test_f1 = self.evaluate()
        print("测试模式, 测试准确率={},测试f值{}".format(test_acc,test_f1))
        self.module_test_acc.append(test_acc)
        self.module_test_f1.append(test_f1)
        return test_acc,test_f1

    def save_module(self):
        if not os.path.exists("module"):
            os.mkdir("module")
        name=self.module_name
        with open("module/"+name+".pkl", "wb") as f:
            torch.save(self.module, f)

    def load_module(self):
        name=self.module_name
        with open("module/"+name+".pkl", "rb") as f:
            self.module = torch.load(f)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        self.module.to(device)

    def get_performance(self):
        return {"train_acc":self.module_train_acc,"train_f1":self.module_train_f1,"train_loss":self.module_train_loss,"test_acc":self.module_test_acc,"test_f1":self.module_test_f1}


def print_basic_info():
    # print basic information of the GPU
    print("name of the GPU:", torch.cuda.get_device_name(0))
    print("device:", device)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    avaliable_models = ["CNN_simple", "CNN_complex","RNN","DenseNet", "ResNet", "LSTM", "BiLSTM", "GoogLeNet", "MLP","MLP_with_Dropout"]
    learning_rate_dict={
        "CNN_simple":1e-2,
        "CNN_complex":1e-3,
        "RNN":1e-3,#try 1e-2
        "DenseNet":1e-3,#try 1e-3
        "ResNet":1e-4,#try 1e-4
        "LSTM":1e-3,#little more
        "BiLSTM":1e-3,
        "GoogLeNet":1e-4,
        "MLP":1e-4,
        "MLP_with_Dropout":1e-4,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(20020214)
    print_basic_info()
    Mode = "test" # "train" or "test"



    if Mode=="train":
        performance={}
        for name in avaliable_models:
            print("当前模型：",name)
            predict=PredictClass(device=device, module_name=name, num_epochs=16, batch_size=64, learning_rate=learning_rate_dict[name],)
            predict.run_from_zero()
            performance[name]=predict.get_performance()
        import matplotlib.pyplot as plt
        # 折线图，所有的模型的训练准确率
        for name in avaliable_models:
            plt.plot(performance[name]["train_acc"])
        plt.legend(avaliable_models)

        plt.axhline(y=0.8, color='r', linestyle='--')
        plt.savefig("plot/train_acc.png")
        plt.close()
        for name in avaliable_models:
            plt.plot(performance[name]["train_f1"])
        plt.legend(avaliable_models)

        plt.axhline(y=0.8, color='r', linestyle='--')
        plt.savefig("plot/train_f1.png")
        plt.close()
        for name in avaliable_models:
            plt.plot(performance[name]["train_loss"])
        plt.legend(avaliable_models)
        plt.savefig("plot/train_loss.png")
        plt.close()
        for name in avaliable_models:
            plt.plot(performance[name]["test_acc"])
        plt.legend(avaliable_models)

        plt.axhline(y=0.8, color='r', linestyle='--')
        plt.savefig("plot/val_acc.png")
        plt.close()
        for name in avaliable_models:
            plt.plot(performance[name]["test_f1"])
        plt.legend(avaliable_models)

        plt.axhline(y=0.8, color='r', linestyle='--')
        plt.savefig("plot/val_f1.png")
        plt.close()


    elif Mode=="test":
        acc_list=[]
        F1_list=[]
        for name in avaliable_models:
            print("当前模型：",name)
            predict=PredictClass(device=device, module_name=name, num_epochs=40, batch_size=64, learning_rate=1e-4,)
            test_acc,test_f1=predict.run_from_load_module()
            acc_list.append(test_acc)
            F1_list.append(test_f1)

        import matplotlib.pyplot as plt
        # 散点图
        plt.scatter(range(0, len(avaliable_models)), acc_list)
        plt.scatter(range(0, len(avaliable_models)), F1_list)
        plt.legend(["acc", "F1"])
        plt.xticks(range(0, len(avaliable_models)), avaliable_models,rotation=75)
        plt.axhline(y=0.8, color='r', linestyle='--')
        plt.savefig("plot/all_acc_F1.png")