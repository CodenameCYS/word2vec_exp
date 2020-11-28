from gensim.models import word2vec

if __name__ == "__main__":
    sentences = word2vec.LineSentence("data/gensim/train.txt")
    model = word2vec.Word2Vec(sentences, size=100)

    #保存模型，供日後使用
    model.save("model/gensim.model")