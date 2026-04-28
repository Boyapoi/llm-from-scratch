# Day 6：词嵌入 Embedding

## 为什么不能直接用 token id

BPE 给每个 token 分配一个整数，比如 `'cat'→8, 'dog'→9`。

但这个数字本身没有意义：
- `9 > 8` 不代表 dog 比 cat 重要
- `9 - 8 = 1` 不代表它们只差一点点

神经网络会把这些大小关系当真，结果一定是错的。

**解决方案**：把每个 id 映射到一个高维向量，让向量的方向和距离承载语义。

---

## Embedding 层的本质：查找表

```
词表大小 vocab_size = 10
向量维度 d_model    = 4

Embedding 矩阵 E，形状 (10, 4)：

       dim0   dim1   dim2   dim3
id 0 [ 0.21  -0.34   0.12   0.87]   ← 'n' 的向量
id 1 [-0.45   0.67  -0.23   0.11]   ← 'e' 的向量
...
id 8 [ 0.33   0.91  -0.12  -0.55]   ← 'cat' 的向量
id 9 [-0.12  -0.45   0.78   0.33]   ← 'dog' 的向量

token id=8 → 取第 8 行 → [0.33, 0.91, -0.12, -0.55]
```

就是**按 id 取矩阵的某一行**，没有任何复杂运算。

```python
# numpy 实现
def forward(self, token_ids):
    return self.W[token_ids]   # 花式索引 = 查表
```

---

## 向量为什么能表达语义

初始时这些向量是随机的，没有意义。  
训练过程中，反向传播会调整每一行的数值，让语义相近的词向量方向接近——不是人为规定的，是从大量文本里**自动学出来的**。

最著名的例子：
```
vec("king") - vec("man") + vec("woman") ≈ vec("queen")
```

语义关系变成了向量空间里的方向，加减法有了意义。

---

## 余弦相似度：度量语义距离

```
cos(a, b) = a·b / (|a| × |b|)
```

- 范围 [-1, 1]
- 越接近 1 → 方向越相近 → 语义越接近
- 越接近 0 → 方向垂直 → 语义无关
- 越接近 -1 → 方向相反 → 语义相反

训练后验证：
```
cos(cat, dog)  > cos(cat, the)    # 动物 > 功能词
```

---

## 训练：Word2Vec CBOW 简化版

**任务**：给定上下文词，预测中心词

```
"the cat sat on mat"，窗口=2
上下文 ['the','cat','on','mat'] → 预测 'sat'
```

**前向传播：**
```
h = mean(embedding[context_ids])   # 上下文词向量平均
scores = W_out @ h                  # 每个词的得分
probs = softmax(scores)             # 概率分布
L = -log(probs[target_id])          # 交叉熵损失
```

**反向传播（关键）：**
```
d_scores = probs - one_hot(target)
d_h = W_out.T @ d_scores

# 平均池化的反向：梯度平均分配给每个上下文词
d_context = tile(d_h / len(context_ids), (len(context_ids), 1))
```

为什么除以 `len(context_ids)`？  
前向时做了平均 `h = sum/n`，反向时链式法则要除以 n，每个上下文词只获得 `1/n` 份梯度。

---

## Embedding 的稀疏梯度更新

```python
def backward(self, grad, lr):
    for i, token_id in enumerate(self.last_ids):
        self.W[token_id] -= lr * grad[i]   # 只更新用到的行
```

和全连接层不同：Embedding 每次反向传播只更新这批数据用到的 token 对应的行，其他行梯度为零（**稀疏更新**）。

词表越大这个优势越明显，GPT-2 词表 5 万个 token，每次只需更新实际出现的几十个。

---

## 完整的输入 Pipeline

```
原始文本: "the cat sat"
    ↓ 分词 + BPE（Day 5）
token id: [18, 6, 17]
    ↓ Embedding 查表（Day 6）
向量矩阵: shape (3, d_model)
    ↓
送进 Attention 层（Day 8）
```

---

## 运行

```bash
python day06_embedding.py
```

输出包含训练前后余弦相似度对比，以及 cat/dog 最相似词排名。
