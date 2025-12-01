# 流程预测任务

## 任务简介

实现一个基于深度学习的流程预测算法。项目已给出LSTM模型和两种事件序列的嵌入方法。请你将给定的训练数据和测试数据转换成项目可以接收的数据格式，并通过对LSTM模型进行参数调整实现模型的调优。算法基于Python实现。

### 环境设置

首先需要安装Conda。

接下来，使用以下命令创建Conda环境：

```bash
conda create -n event_chain -c conda-forge python=3.10 tensorflow=2.10 keras=2.10 numpy=1.23 pyyaml=6.0 "h5py>=3.8,<3.9"
```

激活环境：

```bash
conda activate event_chain
```

### 数据说明

- **训练数据**：位于 `data/train`，每行格式为 `答案索引<@>前缀事件1<@>...前缀事件8<@>候选事件1<@>...候选事件5`。
- **测试数据**：位于 `data/test/` 目录下，每个文件包含一个事件链，每行格式为 `id activity role`。
- 事件格式：`activity1<|>role1...`。

### 代码结构

- `main.py`：主入口，运行整个流程。
- `chain.py`：数据读取和解析，包含Event和Question类。
- `classify.py`：特征提取和预处理。
- `pmi.py`：基于PMI的事件序列嵌入方法。
- `bigram.py`：基于Bigram的事件序列嵌入方法。
- `nn.py`：神经网络模型。
- `evaluate.py`：评估函数。

### 运行步骤

1. 激活环境：`conda activate event_chain`
2. 运行主程序：`python main.py`

## 作业任务

1. 补全 `chain.py` 中的函数（注释 `TODO start` 和 `TODO end` 之间）：
   1. 实现 `read_question_corpus()`：读取训练文件，解析每行，构造Question对象列表。
   2. 实现 `read_c_and_j_corpus()`：读取测试集，构造Question对象列表。生成候选事件时可在正确的事件外通过随机生成的方式加入4个候选事件，以匹配任务的统一格式。

2. 在 `nn.py` 中设置合适的训练参数（注释 `TODO start` 和 `TODO end` 之间），以达到较好的准确率。


### 作业完成

事实证明，调参在正确率提高的影响微乎其微
我们观察候选数据train： <@>接诊<|><@>发药<|>药房<@>开单<|><@>报告生成<|><@>收费<|>


答案是带角色的，模型学习只会学到我要选带角色的输出

在chain.py的165行中加入这个即可
```python

# 清空干扰项的角色
            for i in range(1, len(choices)):
                choices[i].role = []

```

在train.py和nn.py目前都固定了随机种子