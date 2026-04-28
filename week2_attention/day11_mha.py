"""
Multi-Head Attention — 纯 numpy 实现
Day 11：从单头到多头，理解切分·并行·拼接·投影
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def single_head_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores += mask * (-1e9)
    weights = softmax(scores)
    return weights @ V, weights


class MultiHeadAttention:
    """
    d_model = n_heads × d_k
    切分 → 各头独立 Attention → 拼接 → 输出投影
    参数量 = 4 × d_model²（与头数无关！）
    """
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = np.sqrt(2.0 / d_model)
        self.Wq = np.random.randn(d_model, d_model) * scale
        self.Wk = np.random.randn(d_model, d_model) * scale
        self.Wv = np.random.randn(d_model, d_model) * scale
        self.Wo = np.random.randn(d_model, d_model) * scale

    def forward(self, X, mask=None, verbose=False):
        seq_len = X.shape[0]

        # Step 1: 整体投影
        Q = X @ self.Wq    # (seq, d_model)
        K = X @ self.Wk
        V = X @ self.Wv

        # Step 2: 切分成 n_heads 份
        # (seq, d_model) → (n_heads, seq, d_k)
        Q_h = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1,0,2)
        K_h = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1,0,2)
        V_h = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1,0,2)
        if verbose: print(f"  切分后每头形状: {Q_h.shape} (n_heads, seq, d_k)")

        # Step 3: 每头独立做 Attention
        head_outputs, all_weights = [], []
        for h in range(self.n_heads):
            out_h, w_h = single_head_attention(Q_h[h], K_h[h], V_h[h], mask)
            head_outputs.append(out_h)
            all_weights.append(w_h)

        # Step 4: 拼接
        # (n_heads, seq, d_k) → (seq, d_model)
        concat = np.stack(head_outputs, axis=0).transpose(1,0,2).reshape(seq_len, self.d_model)
        if verbose: print(f"  拼接后形状: {concat.shape}")

        # Step 5: 输出投影（让各头信息互相交流）
        output = concat @ self.Wo
        return output, all_weights


print("=" * 60)
print("Day 11：Multi-Head Attention 完整实现")
print("=" * 60)

tokens = ["the", "cat", "sat", "mat"]
seq_len, d_model, n_heads = 4, 8, 2
d_k = d_model // n_heads
X = np.random.randn(seq_len, d_model)

print(f"\n配置: seq={seq_len}, d_model={d_model}, n_heads={n_heads}, d_k={d_k}")

mha = MultiHeadAttention(d_model, n_heads)
output, all_weights = mha.forward(X, verbose=True)
print(f"输出形状: {output.shape}  ← 和输入 X 一致")

print(f"\n两个头的关注模式（不同头学到不同关系）:")
for h in range(n_heads):
    w = all_weights[h]
    tops = [tokens[w[i].argmax()] for i in range(seq_len)]
    print(f"  头{h}: {' | '.join(f'{tokens[i]}'+'→'+'\''+tops[i]+'\'' for i in range(seq_len))}")

print(f"\n参数量分析:")
print(f"  Wq+Wk+Wv+Wo = 4 × {d_model}×{d_model} = {4*d_model*d_model} 个参数")
print(f"  ← 与头数无关！n_heads=1 和 n_heads=8 参数量相同")
print(f"  多头的代价只是把计算切分到不同子空间，不增加参数")

print(f"\n公式对照:")
print(f"  投影   Q=X·Wq       → X @ self.Wq")
print(f"  切分   (seq,d)→(h,seq,d_k) → reshape + transpose")
print(f"  各头   headᵢ=Att(Qᵢ,Kᵢ,Vᵢ) → for h in range(n_heads)")
print(f"  拼接   Concat(head₀..headₕ) → np.stack + reshape")
print(f"  输出   MHA = Concat·Wo → concat @ self.Wo")
