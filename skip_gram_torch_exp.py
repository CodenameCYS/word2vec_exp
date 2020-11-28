import os
import json
import jieba
import numpy
import torch
import pickle
from tqdm import tqdm

vocabfile = "data/vocab.txt"
w2id = {line.strip():idx for idx, line in enumerate(open(vocabfile))}
id2w = {idx:line.strip() for idx, line in enumerate(open(vocabfile))}
vocab_size = len(w2id)

dim = 2
embedding = torch.nn.Embedding(vocab_size, dim)
torch.nn.init.uniform_(embedding.weight, -0.05, 0.05)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda_gpu = torch.cuda.is_available()
# cuda_gpu = False

class Model(torch.nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.embedding = embedding
        self.linear_layer = torch.nn.Linear(dim, vocab_size)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight)
        self.window_size = window_size

    def forward(self, x):
        emb = self.embedding(x)
        # print(emb.size())
        emb = torch.unsqueeze(emb, 1).repeat(1, self.window_size, 1)
        # print(emb.size())
        output = self.linear_layer(emb)
        # print(output.size())
        return output

class Word2Vec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = embedding
    
    def forward(self, x):
        output = self.embedding(x)
        return output

def cross_entropy(y_pred, y_true):
    y_pred = torch.nn.functional.softmax(y_pred, dim=-1)
    num_classes = y_pred.size()[-1]
    y_true = torch.nn.functional.one_hot(y_true, num_classes=num_classes)
    loss = - y_true * torch.log(y_pred)
    return torch.mean(torch.sum(loss, dim=-1))

def train(model, x, y, batch_size=512, epochs=10):
    n = (len(x)-1) // batch_size + 1

    optimizer = torch.optim.Adam(model.parameters())
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = cross_entropy

    for k in range(epochs):
        with tqdm(range(n), ncols=100) as t:
            for i in t:
                optimizer.zero_grad()
                if cuda_gpu:
                    inputs = torch.tensor(x[i*batch_size:(i+1)*batch_size]).cuda()
                    y_true = torch.LongTensor(y[i*batch_size:(i+1)*batch_size]).cuda()
                else:
                    inputs = torch.tensor(x[i*batch_size:(i+1)*batch_size])
                    y_true = torch.LongTensor(y[i*batch_size:(i+1)*batch_size])
                y_pred = model(inputs)
                loss = loss_fn(y_pred, y_true)
                loss.backward()
                optimizer.step()
        print("{} epochs: loss = {}".format(k+1, loss.item()))
    
    return

def test_word2vec(model, word_list):
    x = torch.tensor(numpy.array([w2id.get(w, 0) for w in word_list]))
    if cuda_gpu:
        x = x.cuda()
        y = model(x).cpu().detach().numpy()
    else:
        y = model(x).detach().numpy()
    return {w:emb for w, emb in zip(word_list, y)}


if __name__ == "__main__":
    x = pickle.load(open("data/skip_gram/src.pkl", "rb"))
    y = pickle.load(open("data/skip_gram/tgt.pkl", "rb"))
    window_size = y.shape[1]

    color = list("红赤黄绿青蓝紫黑白灰")
    place = "大理 天竺 梁山 景阳冈 洛阳 东京 江南 沧州 凉州 邯郸".split()
    jobs = "知府 教头 县令 账房 丞相 太尉 皇帝 国师 员外 知州".split()

    if cuda_gpu:
        model = Model(window_size).cuda()
        word2vec = Word2Vec().cuda()
    else:
        model = Model(window_size)
        word2vec = Word2Vec()
    print("=== model ===")
    print(model)
    print("=== word2vec ===")
    print(word2vec)

    res = test_word2vec(word2vec, color + place + jobs)
    pickle.dump(res, open("data/skip_gram_before_train_torch.pkl", "wb+"))

    train(model, x, y, batch_size=128, epochs=50)

    res = test_word2vec(word2vec, color + place + jobs)
    pickle.dump(res, open("data/skip_gram_after_train_torch.pkl", "wb+"))