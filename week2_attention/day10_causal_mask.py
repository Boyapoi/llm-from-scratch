"""
Day 10：因果掩码完整实现
强制 token i 只能看到位置 <= i 的 token
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)

def make_causal_mask(seq_len):
    """上三角为1（遮住），对角线+下三角为0（可见）"""
    return np.triu(np.ones((seq_len, seq_len)), k=1)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask * (-1e9)   # -∞ → exp(-∞)≈0 → softmax后归零
    return softmax(scores), softmax(scores) @ V

tokens = ["the", "cat", "sat", "mat"]
seq_len = 4
X = np.array([[1.,0.],[0.,1.],[1.,1.],[0.,2.]])
Q = K = V = X

print("=" * 60)
print("Day 10：因果掩码完整实现与验证")
print("=" * 60)

mask = make_causal_mask(seq_len)
print(f"\n因果掩码（1=遮住，0=可见）:")
print(mask)

w_full, _ = attention(Q, K, V)
w_causal, _ = attention(Q, K, V, mask)

print(f"\n【无掩码】注意力权重:")
for i, ti in enumerate(tokens):
    print(f"  '{ti}': {w_full[i].round(3)}")

print(f"\n【有掩码】注意力权重（token i 只看 ≤ i 的位置）:")
for i, ti in enumerate(tokens):
    row = w_causal[i].round(4)
    print(f"  '{ti}'（可见前{i+1}个）: {row}  和={row[:i+1].sum():.4f}")

print(f"\n验证：所有未来位置权重为0:")
all_ok = True
for i in range(seq_len):
    for j in range(i+1, seq_len):
        ok = abs(w_causal[i,j]) < 1e-6
        if not ok: all_ok = False
        print(f"  w[{i},{j}] ('{tokens[i]}'看'{tokens[j]}') = {w_causal[i,j]:.2e}  {'✓' if ok else '✗'}")
print(f"结论: {'掩码正确，所有未来位置权重为0' if all_ok else '存在错误'}")

print(f"""
为什么用 -1e9 而不是直接置0：
  方法A（错）：softmax之后置0 → 破坏"每行之和=1"
  方法B（对）：softmax之前加-∞ → exp(-1e9)≈0，自动排除，全程可微 ✓
""")
