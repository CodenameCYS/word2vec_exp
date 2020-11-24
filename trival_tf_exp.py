import os
import json
import jieba
import numpy
import pickle
import tensorflow as tf

vocabfile = "data/vocab.txt"
w2id = {line.strip():idx for idx, line in enumerate(open(vocabfile))}
id2w = {idx:line.strip() for idx, line in enumerate(open(vocabfile))}
vocab_size = len(w2id)

class CrossEntropy(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        y_true = tf.one_hot(y_true, vocab_size)
        # print(y_true.shape, y_pred.shape)
        return self.loss_fn(y_true, y_pred)

class Word2Vec(object):
    def __init__(self, dim, window_size):
        self.embedding = tf.keras.layers.Embedding(vocab_size, dim)
        self.build(dim, window_size)

    def build(self, dim, window_size):
        inputs = tf.keras.Input(shape = (window_size, ))
        embedding = self.embedding(inputs)
        output = tf.keras.layers.Dense(vocab_size, activation=tf.nn.softmax)(embedding)
        self.model = tf.keras.Model(inputs, output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=CrossEntropy())
        print("=== model ===")
        self.model.summary()

        embedding_input = tf.keras.Input(shape=())
        output = self.embedding(embedding_input)
        self.embedding_model = tf.keras.Model(embedding_input, output)
        self.embedding_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=CrossEntropy())
        print("=== embedding ===")
        self.embedding_model.summary()

    def train(self, x, y, batch_size, epochs=20):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs)

    def test(self, word_list):
        x = numpy.array([w2id.get(w, 0) for w in word_list])
        y = self.embedding_model.predict(x)
        return {w:emb for w, emb in zip(word_list, y)}

    def save(self, foutpath):
        os.makedirs(os.path.split(foutpath)[0], exist_ok=True)
        self.embedding_model.save(foutpath)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    x = pickle.load(open("data/trival/src.pkl", "rb"))
    y = pickle.load(open("data/trival/tgt.pkl", "rb"))
    window_size = x.shape[1]

    color = list("红赤黄绿青蓝紫黑白灰")
    place = "大理 天竺 梁山 景阳冈 洛阳 东京 江南 沧州 凉州 邯郸".split()
    jobs = "知府 教头 县令 账房 丞相 太尉 皇帝 国师 员外 知州".split()

    model = Word2Vec(2, window_size)

    res = model.test(color + place + jobs)
    pickle.dump(res, open("data/trival_before_train.pkl", "wb+"))

    model.train(x, y, batch_size=48, epochs=10)
    model.save("model/trival_embedding.h5")
    
    res = model.test(color + place + jobs)
    pickle.dump(res, open("data/trival_after_train.pkl", "wb+"))