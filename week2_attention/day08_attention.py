"""
Attention 机制 — 纯 numpy 实现
Day 8：从公式到代码，手算验证每一步
Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ) · V
"""

import numpy as np
np.random.seed(42)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ) · V
    输入: Q,K (seq,d_k)  V (seq,d_v)  mask (seq,seq) 可选
    输出: output (seq,d_v)  weights (seq,seq)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # Step 1+2: Q·Kᵀ / √dₖ
    if mask is not None:
        scores = scores + mask * (-1e9)       # Step 3: 因果掩码
    weights = softmax(scores, axis=-1)        # Step 4: softmax
    return weights @ V, weights              # Step 5: × V


print("=" * 60)
print("Attention 机制 — 手算验证（Day 8）")
print("=" * 60)

tokens = ["cat", "sat", "mat"]
seq_len, d_k = 3, 2

Q = np.array([[1.,0.],[0.,1.],[1.,1.]])
K = np.array([[1.,0.],[0.,1.],[1.,1.]])
V = np.array([[1.,0.],[0.,1.],[1.,1.]])

scores_raw = Q @ K.T
print(f"\nStep 1: Q·Kᵀ\n{scores_raw}")

scores_scaled = scores_raw / np.sqrt(d_k)
print(f"\nStep 2: 除以 √{d_k} ≈ {np.sqrt(d_k):.3f}\n{scores_scaled.round(3)}")

weights = softmax(scores_scaled, axis=-1)
print(f"\nStep 3: softmax\n{weights.round(4)}")
print(f"每行之和: {weights.sum(axis=1).round(6)}")

output, _ = scaled_dot_product_attention(Q, K, V)
print(f"\nStep 4: output = weights × V\n{output.round(4)}")

# 手算验证 output[0]
print(f"\n验证 output[0] (cat的输出):")
manual = sum(weights[0,j] * V[j] for j in range(seq_len))
print(f"  手算 = {manual.round(4)}")
print(f"  代码 = {output[0].round(4)}")
print(f"  匹配: {np.allclose(manual, output[0])}")

# 因果掩码演示
mask = np.triu(np.ones((seq_len, seq_len)), k=1)
output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask)
print(f"\n【因果掩码】注意力权重:")
print(weights_masked.round(4))
print("上三角验证（应全为0）:")
for i in range(seq_len):
    for j in range(i+1, seq_len):
        print(f"  w[{i},{j}] = {weights_masked[i,j]:.2e}  ✓")

print(f"\n公式对照:")
print(f"  Q·Kᵀ       → Q @ K.T")
print(f"  / √dₖ      → / np.sqrt(d_k)")
print(f"  掩码        → scores + mask * (-1e9)")
print(f"  softmax     → softmax(scores, axis=-1)")
print(f"  × V        → weights @ V")
