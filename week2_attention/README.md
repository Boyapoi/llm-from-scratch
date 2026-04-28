# Week 2：注意力机制

## 内容概览

| Day | 主题 | 文件 | 核心内容 |
|-----|------|------|----------|
| Day 8 | Attention 公式 | `day08_attention.py` | Q·K·V 四步 + 因果掩码演示 |
| Day 9 | 手算注意力矩阵 | `day09_attention_manual.py` | 从 X·Wq/Wk/Wv 开始逐格手算 |
| Day 10 | 因果掩码 | `day10_causal_mask.py` | -∞ 技巧 + 无掩码 vs 有掩码对比 |
| Day 11 | Multi-Head Attention | `day11_mha.py` | 切分·并行·拼接·输出投影 |
| Day 12-14 | MHA 完整实现 | *(coming soon)* | numpy 实现 + PyTorch 验证 |

## 运行方法

```bash
python day08_attention.py        # Attention 四步，手算验证
python day09_attention_manual.py # 4个token完整推导
python day10_causal_mask.py      # 因果掩码，逐行验证
python day11_mha.py              # Multi-Head Attention
```

## 关键公式

```
Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ) · V

完整流程：
  Q = X·Wq,  K = X·Wk,  V = X·Wv   # 投影
  scores  = Q @ K.T / √d_k           # 相似度 + 缩放
  scores += mask * (-1e9)             # 因果掩码（可选）
  weights = softmax(scores)           # 概率分布
  output  = weights @ V               # 加权求和

Multi-Head：
  d_k = d_model // n_heads            # 切分子空间
  每头独立做 Attention，结果拼接后乘 Wo
  参数量 = 4 × d_model²（与头数无关）
```
