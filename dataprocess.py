import numpy as np
import pandas as pd
import os
import pickle


data_path="Dataset/"
input_path = "input/"
train_path=data_path+"train.txt"
test_path=data_path+"test.txt"
validation_path=data_path+"validation.txt"
freq_treshold=10


class DataProcesser():
    """
    用于处理数据的类
    使用方法：
    1.初始化DataProcesser类
    2.run()方法
    """
    def __init__(self,train_path,test_path,validation_path):
        self.sequence_length = 96
        self.train_path=train_path
        self.test_path=test_path
        self.validation_path=validation_path
        self.pad_char="把"
        self.pad_id=0
        self.word2idx={}
        self.vocab=None#词表,是一个所有可能词（出现次数大于阈值）set
        self.train_data=None
        self.test_data=None
        self.validation_data=None

    def set_vocab(self):
        with open(os.path.join(data_path,"word_freq.txt"), encoding='utf-8') as fin:
            vocab = [i.split("\t")[0] for i in fin]
            vocab=set(vocab)#去重
            self.vocab=vocab
        self.word2idx={i:index for index, i in enumerate(vocab)}#构建词表
        self.pad_id=self.word2idx[self.pad_char]#设置pad_id，即pad_char在词表中的索引

    def tokenizer(self,data):
        inputs = []
        sentence_list = [i.split() for i in data["comment"]]
        # 将输入文本进行padding
        for sentence in sentence_list:
            temp=[self.word2idx.get(word,self.pad_id) for word in sentence]#如果word2idx中有这个词，那么就返回这个词的索引，如果没有，那么就返回pad_id
            if len(temp)==0:
                temp=[self.pad_id]
            while len(temp)<self.sequence_length:#如果句子长度小于sequence_length，那么复制一份句子，直到句子长度大于等于sequence_length
                temp.extend(temp)
            temp=temp[:self.sequence_length]#截断句子，使得句子长度等于sequence_length
            inputs.append(temp)
        return {"comment":inputs,"label":data["label"]}

    def init_raw_data(self):
        train_data = pd.read_csv(self.train_path,names=["label","comment"],sep="\t")
        test_data = pd.read_csv(self.test_path,names=["label","comment"],sep="\t")
        validation_data = pd.read_csv(self.validation_path,names=["label","comment"],sep="\t")
        self.train_data = self.tokenizer(train_data)
        self.test_data = self.tokenizer(test_data)
        self.validation_data = self.tokenizer(validation_data)

    def init_word_vec(self):
        from gensim.models import keyedvectors
        w2v=keyedvectors.load_word2vec_format(os.path.join(data_path,"wiki_word2vec_50.bin"),binary=True)
        vocab_l={}
        for word in self.vocab:
            word_idx=self.word2idx[word]
            try:
                vocab_l[word_idx]=w2v[word]
            except KeyError:
                vocab_l[word_idx]=np.random.uniform(-0.25,0.25,50)
        self.word2vec=vocab_l

    def pkl_dump_word2vec(self,data, path):
        data_input={}
        data_input["label"]=[i for i in data["label"]]
        data_input["comment"]=[]
        for i in data["comment"]:
            temp=[]
            for j in i:
                temp.append(self.word2vec[j])
            data_input["comment"].append(temp)
        with open(os.path.join(input_path,path), "wb") as fout:
            pickle.dump(data_input, fout)

    def gen_data_input(self):
        with open(os.path.join(input_path,"word2vec.pkl"), "wb") as fout:
            pickle.dump(self.word2vec, fout)
        self.pkl_dump_word2vec(self.train_data,"train_input.pkl")
        self.pkl_dump_word2vec(self.test_data,"test_input.pkl")
        self.pkl_dump_word2vec(self.validation_data,"validation_input.pkl")

    def run(self):
        self.set_vocab()
        self.init_raw_data()
        self.init_word_vec()
        self.gen_data_input()

data_processer=DataProcesser(train_path,test_path,validation_path)
data_processer.run()
