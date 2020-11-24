import os
import re
import jieba
import random
import pickle
import numpy
from tqdm import tqdm
from collections import defaultdict

PUNCTUATION = re.compile(r"[,，。\?？!！…]")

def create_vocab_file(datapath, foutpath):
    vocab = defaultdict(int)
    for datafile in os.listdir(datapath):
        datafile = os.path.join(datapath, datafile)
        print(datafile)
        for line in open(datafile):
            tokens = jieba.lcut(line.strip())
            for w in tokens:
                vocab[w] += 1
    vocab = sorted(vocab.items(), key=lambda x:x[1], reverse=True)
    with open(foutpath, "w+") as fout:
        fout.write("<unk>\n")
        fout.write("<blank>\n")
        with tqdm(vocab, ncols=100) as t:
            for w, _ in t:
                fout.write("{}\n".format(w))
    return

def create_cbow_dataset(datapath, vocabfile, foutpath, window_size=3):
    os.makedirs(foutpath, exist_ok=True)
    vocab = {line.strip():idx for idx, line in enumerate(open(vocabfile))}

    def preprocess(tokens):
        n = len(tokens)
        idx = random.choice(range(window_size, n - window_size))
        src = tokens[idx-window_size:idx] + tokens[idx+1: idx+window_size+1]
        tgt = tokens[idx]
        src = [vocab.get(w, 0) for w in src]
        tgt = vocab.get(tgt, 0)
        return src, tgt

    src = []
    tgt = []
    for datafile in os.listdir(datapath):
        datafile = os.path.join(datapath, datafile)
        with tqdm(open(datafile)) as t:
            for line in t:
                for sentence in PUNCTUATION.split(line.strip()):
                    tokens = jieba.lcut(sentence)
                    if len(tokens) <= window_size*2+1:
                        continue
                    src, tgt = preprocess(tokens)
                    src.append(tokens)
                    tgt.append(tokens)
    pickle.dump(numpy.array(src), open(os.path.join(foutpath, "src.pkl"), "wb"))
    pickle.dump(numpy.array(tgt), open(os.path.join(foutpath, "tgt.pkl"), "wb"))
    return

def create_skip_gram_dataset(datapath, vocabfile, foutpath, window_size=3):
    os.makedirs(foutpath, exist_ok=True)
    vocab = {line.strip():idx for idx, line in enumerate(open(vocabfile))}

    def preprocess(tokens):
        n = len(tokens)
        idx = random.choice(range(window_size, n - window_size))
        src = tokens[idx-window_size:idx] + tokens[idx+1: idx+window_size+1]
        tgt = tokens[idx]
        src = [vocab.get(w, 0) for w in src]
        tgt = vocab.get(tgt, 0)
        return tgt, src

    src = []
    tgt = []
    for datafile in os.listdir(datapath):
        datafile = os.path.join(datapath, datafile)
        with tqdm(open(datafile)) as t:
            for line in t:
                for sentence in PUNCTUATION.split(line.strip()):
                    tokens = jieba.lcut(sentence)
                    if len(tokens) <= window_size*2+1:
                        continue
                    src, tgt = preprocess(tokens)
                    src.append(tokens)
                    tgt.append(tokens)
    pickle.dump(numpy.array(src), open(os.path.join(foutpath, "src.pkl"), "wb"))
    pickle.dump(numpy.array(tgt), open(os.path.join(foutpath, "tgt.pkl"), "wb"))
    return

def create_trival_dataset(datapath, vocabfile, foutpath, maxlen=100):
    os.makedirs(foutpath, exist_ok=True)
    vocab = {line.strip():idx for idx, line in enumerate(open(vocabfile))}

    def padding(tokens):
        if len(tokens) >= maxlen:
            tokens = tokens[:maxlen]
        else:
            tokens = tokens + ["<blank>"] * (maxlen - len(tokens))
        return [vocab.get(w, 0) for w in tokens]

    src = []
    tgt = []
    for datafile in os.listdir(datapath):
        datafile = os.path.join(datapath, datafile)
        with tqdm(open(datafile)) as t:
            for line in t:
                for sentence in PUNCTUATION.split(line.strip()):
                    tokens = jieba.lcut(sentence)
                    tokens = padding(tokens)
                    src.append(tokens)
                    tgt.append(tokens)
    pickle.dump(numpy.array(src), open(os.path.join(foutpath, "src.pkl"), "wb"))
    pickle.dump(numpy.array(tgt), open(os.path.join(foutpath, "tgt.pkl"), "wb"))
    return

def create_gensim_dataset(datapath, foutpath):
    os.makedirs(foutpath, exist_ok=True)

    with open(os.path.join(foutpath, "train.txt"), "w+") as fout:
        for datafile in os.listdir(datapath):
            datafile = os.path.join(datapath, datafile)
            with tqdm(open(datafile)) as t:
                for line in t:
                    if line.strip() == "":
                        continue
                    s = " ".join(jieba.lcut(line.strip()))
                    fout.write("{}\n".format(s))
    return

if __name__ == "__main__":
    # create_vocab_file("data/corpus", "data/vocab.txt")

    # create_cbow_dataset("data/corpus", "data/vocab.txt", "data/cbow/", window_size=3)
    # create_skip_gram_dataset("data/corpus", "data/vocab.txt", "data/skip_gram/", window_size=3)
    # create_trival_dataset("data/corpus", "data/vocab.txt", "data/trival/", maxlen=100)
    create_gensim_dataset("data/corpus", "data/gensim")