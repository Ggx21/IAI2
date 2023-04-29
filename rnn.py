import collections
import pickle
import os
import random
import tarfile
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "data"

print(torch.__version__, device)
fname = os.path.join(DATA_ROOT, "aclImdb_v1.tar.gz")

if not os.path.exists(os.path.join(DATA_ROOT, "aclImdb")):
    print("从压缩包解压...")
    with tarfile.open(fname, 'r') as f:
        f.extractall(DATA_ROOT)

from tqdm import tqdm
def read_imdb(folder='train', data_root="/data/aclImdb"):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

data_root = os.path.join(DATA_ROOT, "aclImdb")
train_data, test_data = read_imdb('train', data_root), read_imdb('test', data_root)
def get_tokenized_imdb(data):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    """
    data: list of [string, label]
    """
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]
def get_vocab_imdb(data):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # 去掉出现次数小于5的词
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
    UNK_TOKEN = '<unk>'
    counter[UNK_TOKEN] = 0
    PAD_TOKEN = '<pad>'
    counter[PAD_TOKEN] = 1
    return Vocab.Vocab(counter)

vocab = get_vocab_imdb(train_data)
# print vocab
print('# words in vocab:', len(vocab))
print('index of the word "the":', vocab['the'])
print('index of the word "movie":', vocab['movie'])


def preprocess_imdb(data, vocab):  # 本函数已保存在d2lzh_torch包中方便以后使用
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    w2i_tensor = []
    for words in tokenized_data:
        w2i_list = []
        for word in words:
            try:
                index=vocab[word]
                w2i_list.append(index)
            except KeyError:
                word= 'a'
                index=vocab[word]
                w2i_list.append(index)
        w2i_list = pad(w2i_list)
        w2i_tensor.append(w2i_list)
    features = torch.tensor(w2i_tensor)
    labels = torch.tensor([score for _, score in data])
    return features, labels

batch_size = 64
train_set = Data.TensorDataset(*preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*preprocess_imdb(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)

for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'#batches:', len(train_iter)

class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)

        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                                hidden_size=num_hiddens,
                                num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4*num_hiddens, 2) # 初始时间步和最终时间步的隐藏状态作为全连接层输入

    def forward(self, inputs):
        embeddings = self.embedding(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs

embed_size, num_hiddens, num_layers = 50, 100, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)

glove_vocab = Vocab.GloVe(name='6B', dim=50, cache=os.path.join(DATA_ROOT, "glove"))

def load_pretrained_embedding(words, pretrained_vocab):
    """从预训练好的vocab中提取出words对应的词向量"""
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为0
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab[word]
            embed[i, :] = idx
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    return embed

vocab_keys = vocab.vocab.keys()
vocab_keys = list(vocab_keys)
net.embedding.weight.data.copy_(load_pretrained_embedding(vocab_keys, glove_vocab))
# 输出词向量的前5个词向量
for i in range(5):
    print(vocab_keys[i], net.embedding.weight.data[i])
net.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它

lr, num_epochs = 0.01, 5
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    import time
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


def predict_sentiment(net, vocab, sentence):
    """sentence是词语的列表"""
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'

print(predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great']))

print(predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad']))




