# Day 10：因果掩码

## 为什么需要因果掩码

Attention 默认让每个 token 看到序列里所有其他 token。  
但语言模型的训练目标是：**根据前面的词预测下一个词**。

如果 token i 能看到 token i+1，相当于考试时提前看到答案，模型什么都学不到。

**因果掩码强制**：token i 只能看到位置 ≤ i 的 token。

```
位置:   0     1     2     3
       the   cat   sat   mat

token 0 (the): 只看 the
token 1 (cat): 看 the, cat
token 2 (sat): 看 the, cat, sat
token 3 (mat): 看 the, cat, sat, mat
```

---

## 实现原理

用上三角矩阵标记"不能看的位置"，在 softmax 之前把这些位置设为 -∞：

```
掩码矩阵（1=遮住）:        加到分数后:
[[0, 1, 1, 1],             [s00,  -∞,  -∞,  -∞]
 [0, 0, 1, 1],    →        [s10, s11,  -∞,  -∞]
 [0, 0, 0, 1],             [s20, s21, s22,  -∞]
 [0, 0, 0, 0]]             [s30, s31, s32, s33]

softmax 后：-∞ 位置变成 0，可见位置正常归一化
```

```python
mask = np.triu(np.ones((seq_len, seq_len)), k=1)
scores = scores + mask * (-1e9)   # -1e9 近似 -∞
```

---

## 为什么是 -1e9 而不是直接置 0

**错误方法**：softmax 之后强行置 0
```python
weights = softmax(scores)
weights[mask == 1] = 0   # 错！破坏了"每行之和=1"，且不可微
```

**正确方法**：softmax 之前加 -∞
```python
scores = scores + mask * (-1e9)   # exp(-1e9) ≈ 0
weights = softmax(scores)         # 未来位置自动被排除，每行之和仍=1
```

正确方法的优点：
- 全程可微，梯度可以正常反向传播
- softmax 内部自然归一化，每行和始终为 1
- 不需要额外的后处理步骤

---

## 可见性验证

加掩码后的注意力权重（`the` 的行）：
```
         the    cat    sat    mat
'the': [1.000,  0.000, 0.000, 0.000]   ← 只看自己，权重=1.0
'cat': [0.449,  0.551, 0.000, 0.000]   ← 看 the 和 cat
'sat': [0.211,  0.211, 0.578, 0.000]   ← 看前三个
'mat': [0.048,  0.239, 0.239, 0.474]   ← 看所有
```

`the` 只能看自己，softmax 里只有一个位置，权重必然是 1.0。

---

## 训练 vs 生成时的区别

**训练时**：整个序列一次性输入，掩码强制模拟"逐步生成"的条件
```
输入: [the, cat, sat, mat]（全部已知）
掩码: 上三角遮住
目标: 每个位置预测下一个 token
```

**生成时**：逐个 token 生成，天然看不到未来
```
步骤1: 输入 [the]         → 预测 cat
步骤2: 输入 [the, cat]    → 预测 sat
步骤3: 输入 [the,cat,sat] → 预测 mat
```

生成时每步都要重新跑一次 Attention，很慢——这就是 Day 24 KV Cache 要解决的问题。

---

## PyTorch 中的用法

```python
# 方式1：手动生成掩码
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

# 方式2：最简单，一个参数搞定（PyTorch 2.0+）
attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

---

## 运行

```bash
python day10_causal_mask.py
```
