# LLM From Scratch 🧠

> 从零实现大语言模型内部原理 — 纯 numpy，无框架依赖

四周强化学习计划，每天一个主题，从数学基础到现代 LLM 优化技术，
所有核心模块都用 numpy 手写实现，配合数学推导和梯度验证。

---

## 项目结构

```
llm-from-scratch/
├── week1_foundations/          # 第一周：数学基础 + 输入处理
│   ├── day04_backprop.py       # 两层网络反向传播（含梯度验证）
│   ├── day05_bpe.py            # BPE 字节对编码分词算法
│   ├── day06_embedding.py      # 词嵌入查找表 + Word2Vec 训练
│   └── README.md
│
├── week2_attention/            # 第二周：注意力机制
│   ├── day08_attention.py      # Scaled Dot-Product Attention
│   ├── day09_attention_manual.py  # 4个token完整手算推导
│   ├── day10_causal_mask.py    # 因果掩码实现与验证
│   ├── day11_mha.py            # Multi-Head Attention
│   └── README.md
│
├── week3_transformer/          # 第三周：Transformer 完整架构（进行中）
│   └── README.md
│
├── week4_modern/               # 第四周：现代 LLM 优化技术（进行中）
│   └── README.md
│
└── README.md                   # 本文件
```

---

## 学习路线

```
Week 1  数学基础           线性代数 → 链式法则 → 交叉熵 → 反向传播
        输入处理           BPE 分词 → Embedding
          ↓
Week 2  注意力机制         Q·K·V → 缩放 → softmax → 因果掩码 → MHA
          ↓
Week 3  Transformer        FFN → 残差 → LayerNorm → 位置编码 → 完整 GPT
          ↓
Week 4  现代优化           RoPE → KV Cache → Flash Attention → LoRA → MoE
```

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/Boyapoi/llm-from-scratch.git
cd llm-from-scratch

# 只需要 numpy（Week 3 之后需要 PyTorch）
pip install numpy

# 运行任意模块
python week1_foundations/day04_backprop.py
python week2_attention/day11_mha.py
```

---

## 各模块核心内容

### Week 1

**Day 4 — 反向传播**
```python
# 五步链式法则
dz2 = (y_hat - y) / batch      # sigmoid + 交叉熵联合求导 → ŷ - y
dW2 = dz2 @ a1.T               # ∂L/∂W = 上游梯度 · 输入ᵀ
da1 = W2.T @ dz2               # 梯度沿权重转置往回传
dz1 = da1 * relu_grad(z1)      # ReLU 开关：负数区域梯度归零
dW1 = dz1 @ x.T
```

**Day 5 — BPE 分词**
```python
# 反复合并语料中频率最高的相邻字符对
# 最终：unknown → ['un','known','</w>']（零 OOV 问题）
```

**Day 6 — Embedding**
```python
# 本质：按 id 取矩阵的行（查找表）
vector = E[token_id]            # (d_model,)
# 训练后语义相近的词向量方向相近
# cos(vec("cat"), vec("dog")) > cos(vec("cat"), vec("the"))
```

### Week 2

**Day 8-9 — Attention**
```python
# Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ) · V
Q, K, V = X @ Wq, X @ Wk, X @ Wv
scores  = Q @ K.T / np.sqrt(d_k)
weights = softmax(scores)
output  = weights @ V
```

**Day 10 — 因果掩码**
```python
# 在 softmax 之前加 -∞，让未来位置权重变为 0
mask   = np.triu(np.ones((seq_len, seq_len)), k=1)
scores = scores + mask * (-1e9)     # exp(-1e9) ≈ 0，全程可微
```

**Day 11 — Multi-Head Attention**
```python
# 切分子空间 → 各头独立 Attention → 拼接 → 输出投影
Q_heads = Q.reshape(seq, n_heads, d_k).transpose(1,0,2)  # (h,seq,d_k)
# 各头学不同关系，参数量 = 4×d_model²（与头数无关）
```

---

## 依赖

| 周次 | 依赖 |
|------|------|
| Week 1-2 | `numpy` |
| Week 3 | `numpy`, `torch` |
| Week 4 | `numpy`, `torch` |

```bash
pip install numpy torch
```

---

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy 的极简 GPT 实现
- [llm-internals](https://github.com/Boyapoi/llm-internals) — 本项目的学习参考资料

---

*持续更新中，每天一个模块 🚀*
