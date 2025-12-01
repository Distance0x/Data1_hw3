#! /usr/bin/env python
#coding=utf-8
import numpy as np

def get_vocabulary(question_chains,k=1000):
    """基于 DF 统计挑选最常见的 k 个事件/角色特征。"""
    # DF
    df={}
    for chains in question_chains:
        for choice_chains in chains:
            for label,choice_features,context_features in choice_chains:    
                for w in choice_features:
                    if w not in df:
                        df[w]=0
                    df[w]+=1
                for w in context_features:
                    if w not in df:
                        df[w]=0
                    df[w]+=1
            
    df=sorted([(df[w],w) for w in df])
    df.reverse()
    df=[w for count,w in df[:k]]
        
    #print df[:100]

    # feature selection
    V={}
    other_count=0
    for i,w in enumerate(df):
        V[w]=len(V)

    print('length of V:',len(V))

    return V

def _get_features(event):
    """将事件转成稀疏字典：activity、role 被包装成可区分的 key。输出格式如：
    {'ACTIVITY_eat': 1, 'ROLE_John': 1}"""
    words={}
    words['ACTIVITY_%s' %event.activity]=1
    for token in event.role:
        words['ROLE_%s' %token]=1
    return words

def get_question_chains(questions):
    """把 Question 对象拆成多条 (label, choice, context) 组合链。"""
    question_chains=[]
    for q in questions:
        chains=[]
        '''
        这段代码在遍历 q.choices（使用 enumerate 获取索引 i 和候选事件 choice_event）。
        对每个候选事件调用 get_features(choice_event)，把事件转为稀疏特征字典：
        - "ACTIVITY_<activity>" = 1
        - "ROLE_<token>" = 1
        如果 i == q.answer 则把该候选标为正例 label=1，否则 label=0。
        然后对 q.context 中的每个 context_event，构造三元组 (label, choice_features, get_features(context_event))，把这些三元组组成一个列表 choice_chains，再把它追加到 chains。
        结果：chains 是一个列表，长度等于候选数；每个元素是与该候选对应的、长度为 len(q.context) 的三元组列表（即形状约为 candidates × contexts）。
        '''
        for i,choice_event in enumerate(q.choices):
            choice_features=_get_features(choice_event)
            if i==q.answer:
                label=1
            else:
                label=0
            choice_chains=[(label,choice_features,_get_features(context_event)) for context_event in q.context]
            chains.append(choice_chains)
        question_chains.append(chains)
    return question_chains

def _formatK_pair(choice_features,context_features,V):
    """把候选与前缀特征映射成整数索引，便于输入嵌入层。"""
    x=[]
    for w in choice_features:
        if w in V:
            if choice_features[w]>=1:
                x.append(V[w])
    for w in context_features:
        if w in V:
            if context_features[w]>=1:
                x.append(len(V)+V[w])
    return x

def _formatK(words,V):
    """仅对单一特征字典做索引化，用于序列模型。"""
    x=[]
    for w in words:
        if w in V:
            if words[w]>=1:
                x.append(V[w])
    return x

def get_pair_wise_chains(question_chains,V):
    """
    构造全量 pair-wise 样本，正负例全部保留。
    question_chains 是一个列表，长度等于问题数，一个元素是原始数据中一行；
    chains 是一个列表，长度等于候选数，因为chains中一个元素就是每个候选作为最后一个组成的流程；
    也就是说，每个元素是与该问题对应的、长度为候选数的三元组列表

    返回 X 和 Y：
    - X 是一个列表，每个元素是一个列表，表示该样本的特征。
    - Y 是一个列表，每个元素是一个整数，表示该样本的标签（1 或 0）。
    """
    vec_size=len(V)
    X=[]
    Y=[]
    for chains in question_chains:
        for choice_chains in chains:
            for label,choice_features,context_features in choice_chains:
                X.append(_formatK_pair(choice_features,context_features,V))
                Y.append(label)
    return X,Y

def get_pair_wise_chains_for_train(question_chains,V):
    """训练阶段做简单负采样：限制每题只取 1 个负例链。"""
    vec_size=len(V)
    X=[]
    Y=[]
    for chains in question_chains:
        neg_count=0
        for choice_chains in chains:
            cLabel=choice_chains[0][0]
            if cLabel==1 or neg_count<1:
                for label,choice_features,context_features in choice_chains:
                    X.append(_formatK_pair(choice_features,context_features,V))
                    Y.append(label)
                    break
            
            if cLabel==0:
                neg_count+=1
                
    return X,Y

def get_sequence_chains(question_chains,V):
    """把前缀按时间顺序展开，供序列模型（RNN/CNN）训练。"""
    X_list=[[] for i in range(len(question_chains[0][0])+1)]
    Y=[]
    for chains in question_chains:
        for choice_chains in chains:
            Y.append(choice_chains[0][0]) # label
            for i,(label,choice_features,context_features) in enumerate(choice_chains):
                X_list[i].append(_formatK(context_features,V))
            X_list[-1].append(_formatK(choice_chains[0][1],V))
    return X_list,Y

def get_sequence_chains_for_train(question_chains,V):
    """序列训练样本的负采样版本，逻辑与 pair-wise 保持一致。"""
    X_list=[[] for i in range(len(question_chains[0][0])+1)]
    Y=[]
    for chains in question_chains:
        neg_count=0
        for choice_chains in chains:
            label=choice_chains[0][0]
            if label==1:
                Y.append(label) 
                for i,(label,choice_features,context_features) in enumerate(choice_chains):
                    X_list[i].append(_formatK(context_features,V))
                X_list[-1].append(_formatK(choice_chains[0][1],V))
            else:
                if neg_count<1:
                    Y.append(label) 
                    for i,(label,choice_features,context_features) in enumerate(choice_chains):
                        X_list[i].append(_formatK(context_features,V))
                    X_list[-1].append(_formatK(choice_chains[0][1],V))
                neg_count+=1
    return X_list,Y
