# word2vec_exp

这里用来测试一下word2vec实验

## 1. 数据文件

这里的数据我们使用网上下载的四大名著的文本作为我们的训练数据。

数据位于[data/corpus](https://github.com/CodenameCYS/word2vec_exp/tree/main/data/corpus)目录下。

## 2. 数据处理脚本

数据处理脚本为：
- [preprocess_data.py](https://github.com/CodenameCYS/word2vec_exp/blob/main/preprocess_data.py)

输出结果除了gensim实验对应的数据文件为.txt文件外，其余实验对应的数据文件均为对应目录下的.pkl文件，为pickle数据包。

## 3. gensim测试实验

gensim实验脚本如下：
- [gensim_exp.py](https://github.com/CodenameCYS/word2vec_exp/blob/main/gensim_exp.py)

## 4. tensorflow测试实验

tensorflow的实验脚本如下：
1. cbow实验：[cbow_tf_exp.py](https://github.com/CodenameCYS/word2vec_exp/blob/main/cbow_tf_exp.py)
2. skip gram实验：[skip_gram_tf_exp.py](https://github.com/CodenameCYS/word2vec_exp/blob/main/skip_gram_tf_exp.py)
3. 直接生成实验：[trival_tf_exp.py](https://github.com/CodenameCYS/word2vec_exp/blob/main/trival_tf_exp.py)

## 5. pytorch测试实验

pytorch的实验脚本如下：
1. cbow实验：[cbow_torch_exp.py](https://github.com/CodenameCYS/word2vec_exp/blob/main/cbow_torch_exp.py)
2. skip gram实验：
    1. 使用内置cross entropy函数：[skip_gram_torch_exp_v2.py](https://github.com/CodenameCYS/word2vec_exp/blob/main/skip_gram_torch_exp_v2.py)
    2. 使用自定义cross entropy函数：[skip_gram_torch_exp.py](https://github.com/CodenameCYS/word2vec_exp/blob/main/skip_gram_torch_exp.py)

## 6. 结果评测 & 测试

评测结果可以参考两个notebook文件：
1. word2vec测试：[模型效果评测.ipynb](https://github.com/CodenameCYS/word2vec_exp/blob/main/%E6%A8%A1%E5%9E%8B%E6%95%88%E6%9E%9C%E8%AF%84%E6%B5%8B.ipynb)
2. cross entropy测试：[CrossEntropy测试.ipynb](https://github.com/CodenameCYS/word2vec_exp/blob/main/CrossEntropy%E6%B5%8B%E8%AF%95.ipynb)

