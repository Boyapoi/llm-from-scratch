# Day 8：Attention 机制

## 完整公式

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √dₖ) · V
```

四个操作，逐个拆解。

---

## 为什么需要 Q、K、V 三个矩阵

每个 token 在 Attention 中同时扮演三个角色：

- **Q（Query）**：我想要什么信息？
- **K（Key）**：我能提供什么信息？
- **V（Value）**：如果你选了我，给你这个内容

三个矩阵让同一个输入从三个不同角度被投影：
```
Q = X · Wq
K = X · Wk
V = X · Wv
```

Wq、Wk、Wv 是训练中学出来的可训练参数。

---

## Step 1：Q·Kᵀ — 计算相似度

点积是天然的"方向相似度测量仪"：

```
a · b = |a| × |b| × cos(θ)
```

- 方向完全相同 → cos=1 → 点积最大 → 高度关注
- 方向垂直 → cos=0 → 完全不关注
- 方向相反 → cos=-1 → 被抑制

`Q·Kᵀ` 一次性算出所有 token 对之间的相似度：

```
         k_the  k_cat  k_sat  k_mat
q_the  [  1.0    0.2    0.5    0.1 ]  ← the 对各 token 的原始分数
q_cat  [  0.2    1.0    0.3    0.7 ]
q_sat  [  0.5    0.3    1.0    0.4 ]
q_mat  [  0.1    0.7    0.4    1.0 ]
```

---

## Step 2：/ √dₖ — 防止 softmax 饱和

假设 Q、K 的每个分量是均值 0、方差 1 的随机变量，那么两个 dₖ 维向量的点积**方差是 dₖ**，标准差是 √dₖ。

不缩放时 dₖ=64，点积数值可能高达几十，丢进 softmax：
```
softmax([40, 41, 42]) = [0.09, 0.24, 0.67]   # 极度尖锐
softmax([0.4, 0.5, 0.6]) = [0.30, 0.33, 0.37] # 均匀合理
```

softmax 越尖锐，梯度越小（趋近饱和区），训练越难。除以 √dₖ 把方差压回 1，保持合理范围。

---

## Step 3：softmax — 变成概率分布

```
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```

- 把任意实数变成 0~1 之间
- 所有位置加起来 = 1
- 数越大，占的概率越高

每行就是这个 query token 把注意力分配给各个位置的比例。

**数值稳定技巧**：先减去最大值再取 exp，防止 exp(大数) 溢出：
```python
x = x - x.max(axis=-1, keepdims=True)
e = np.exp(x)
return e / e.sum(axis=-1, keepdims=True)
```

---

## Step 4：× V — 加权求和

用 softmax 权重对所有 V 做加权求和：

```
output[i] = Σⱼ attention_weight[i,j] × V[j]
```

关注度高的 token 贡献多，关注度低的贡献少。  
输出向量已经不是任何一个 token 的原始 embedding，而是**融合了上下文信息的新表示**。

---

## 因果掩码（Causal Mask）

自回归生成时，token i 不能看到 i 之后的 token（否则相当于考试看答案）。

```python
# 上三角为 1（遮住），下三角+对角线为 0（可见）
mask = np.triu(np.ones((seq_len, seq_len)), k=1)

# 在 softmax 之前加 -∞
scores = scores + mask * (-1e9)   # exp(-1e9) ≈ 0
```

为什么用 `-1e9` 而不是 softmax 之后直接置 0？
- 必须在 softmax **之前**操作
- `exp(-1e9) ≈ 0` 让未来位置自动被排除在归一化之外
- 全程可微，梯度可以正常反向传播
- softmax 之后强行置 0 会破坏"每行之和=1"的性质

---

## 代码对照

```python
d_k = Q.shape[-1]
scores  = Q @ K.T / np.sqrt(d_k)     # Step 1+2: Q·Kᵀ / √dₖ
scores  = scores + mask * (-1e9)      # Step 3: 因果掩码（可选）
weights = softmax(scores, axis=-1)    # Step 4: softmax
output  = weights @ V                 # Step 5: × V
```

---

## 运行

```bash
python day08_attention.py
```
