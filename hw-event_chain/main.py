#! /usr/bin/env python
#coding=utf-8
from chain import *
from pmi import pmi_prediction
from evaluate import eval
from bigram import bigram_prediction
from classify import *
import nn
from tensorflow.keras.preprocessing import sequence

# 读取原始训练集与测试集，数据由预处理脚本生成
train=read_question_corpus()
test=read_c_and_j_corpus()

# 跑 PMI 基线，用于得到一个直接的选择结果
results=pmi_prediction(train,test)
eval(test,results)

# results=bigram_prediction(train,test)
# eval(test,results)
test0=test
# 将问答样本转换为链式特征表示，方便后续向量化与模型训练
train,test=get_question_chains(train),get_question_chains(test0)
V=get_vocabulary(train,k=100)

# pair wise
# 基于 pair-wise 训练：使用前缀事件与候选事件构造二分类样本
train_x,train_y=get_pair_wise_chains_for_train(train,V)
test_x,test_y=get_pair_wise_chains(test,V)

v_len=len(V)*2
# 启动 LSTM/Embedding 分类器训练，返回已编译并拟合的模型
model=nn.lstm_train(train_x,train_y,v_len)

# 预测前需对稀疏索引序列进行统一 padding，长度由 nn.MAX_LEN 控制
test_x = sequence.pad_sequences(test_x, maxlen=nn.MAX_LEN)
X_pred = model.predict(test_x)

index=0
results=[]
for chains in test:
    choice_results=[]
    for j,choice_chains in enumerate(chains):
        score=0
        for label,choice_features,context_features in choice_chains:
            # 将 LSTM 对 pair 的得分逐一累加，得到该候选答案的总分
            score+=X_pred[index][0]
            index+=1
        choice_results.append((score,j))
    choice_results.sort()
    results.append(choice_results[-1][1])

eval(test0,results)
