# Day 11：Multi-Head Attention

## 为什么单头不够

单头 Attention 的根本限制：**每行 softmax 权重之和 = 1**，一次只能"聚焦一种关系"。

例子：句子 `"the animal didn't cross the street because it was too tired"`

`it` 这个词同时需要：
- 理解 `it` 指代 `animal`（语义/指代关系）
- 理解 `it` 和 `tired` 的修饰关系（句法关系）

单头无法同时给这两件事都分配高权重。

**多头的解决方案**：与其用一套权重同时解决所有问题，不如分工——每个头在自己的子空间里专注学一种关系。

---

## 完整结构

```
输入 X: (seq_len, d_model)
    ↓ 三个投影矩阵
Q = X·Wq,  K = X·Wk,  V = X·Wv    各 (seq_len, d_model)
    ↓ 切分成 n_heads 份，每份 d_k = d_model / n_heads
头0: Q₀,K₀,V₀ (seq_len, d_k) → Attention → (seq_len, d_k)
头1: Q₁,K₁,V₁ (seq_len, d_k) → Attention → (seq_len, d_k)
    ↓ 拼接
Concat: (seq_len, d_model)
    ↓ 输出投影 Wo
输出: (seq_len, d_model)    ← 和输入 X 形状完全相同
```

---

## 五步实现

### Step 1：整体投影

```python
Q = X @ self.Wq    # (seq, d_model)
K = X @ self.Wk
V = X @ self.Wv
```

### Step 2：切分成 n_heads 份

```python
# (seq, d_model) → (n_heads, seq, d_k)
Q_h = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
K_h = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
V_h = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
```

**为什么 reshape + transpose？**  
原来 Q 是 `(seq=4, d_model=8)`，把最后一维 8 切成 `(2头×4维)`，  
再把"头"这个维度挪到最前面，这样 `Q_h[h]` 直接取出第 h 个头的完整序列，  
后面 for 循环逐头计算更方便。

### Step 3：每头独立做 Attention

```python
for h in range(n_heads):
    out_h, w_h = single_head_attention(Q_h[h], K_h[h], V_h[h], mask)
    head_outputs.append(out_h)
```

### Step 4：拼接所有头

```python
# (n_heads, seq, d_k) → (seq, d_model)
concat = np.stack(head_outputs, axis=0)   # (h, seq, d_k)
         .transpose(1, 0, 2)              # (seq, h, d_k)
         .reshape(seq_len, d_model)       # (seq, d_model)
```

### Step 5：输出投影

```python
output = concat @ self.Wo
```

**为什么需要 Wo？**  
拼接后只是把各头的结果并排放在一起，头之间没有"沟通"。  
Wo 让不同头的信息可以互相混合，产生更丰富的表示。

---

## 最重要的洞察：参数量与头数无关

```
n_heads=1（单头，d_k=8）：Wq(8×8) + Wk(8×8) + Wv(8×8) + Wo(8×8) = 256
n_heads=2（双头，d_k=4）：Wq(8×8) + Wk(8×8) + Wv(8×8) + Wo(8×8) = 256
n_heads=8（八头，d_k=1）：Wq(8×8) + Wk(8×8) + Wv(8×8) + Wo(8×8) = 256
```

**参数量完全相同！** 多头不增加参数，只是把同样的计算切分到不同子空间。

---

## 不同头自然学到不同关系

这不是人为设定的，是训练过程中自然涌现的：

- 有的头学句法依赖关系
- 有的头学指代关系（"it" 指代谁）
- 有的头学局部相邻关系

GPT-3 有 96 个头，同时捕捉 96 种不同的语言关系。

---

## GPT-3 的规模

```
d_model = 12288
n_heads = 96
d_k     = 12288 / 96 = 128   # 每个头的维度
```

---

## PyTorch 中的用法

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
output, weights = mha(Q, K, V)
```

---

## 运行

```bash
python day11_mha.py
```
