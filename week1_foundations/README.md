# Week 1：数学基础 + 输入处理

## 内容概览

| Day | 主题 | 文件 | 核心内容 |
|-----|------|------|----------|
| Day 1-3 | 数学基础 | — | 线性代数、链式法则、交叉熵（推导为主，无代码） |
| Day 4 | 反向传播 | `day04_backprop.py` | 两层网络 numpy 实现 + 梯度验证 |
| Day 5 | BPE 分词 | `day05_bpe.py` | 字节对编码完整实现 + 编码新词 |
| Day 6 | 词嵌入 | `day06_embedding.py` | Embedding 查找表 + Word2Vec 训练 |

## 运行方法

```bash
python day04_backprop.py   # 两层网络，XOR 任务，梯度验证
python day05_bpe.py        # BPE 分词，可视化每步合并
python day06_embedding.py  # 训练词向量，cat/dog 余弦相似度
```

## 关键公式

```
反向传播（五步链式法则）：
  dz2 = (ŷ - y) / batch          # sigmoid + 交叉熵联合求导
  dW2 = dz2 · a1ᵀ               # 外积
  da1 = W2ᵀ · dz2               # 转置往回传
  dz1 = da1 ⊙ ReLU'(z1)        # ReLU 开关
  dW1 = dz1 · xᵀ               # 同 dW2 模式

交叉熵：L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
BPE：反复合并语料中频率最高的相邻字符对
Embedding：E[token_id] 取矩阵对应行
```
