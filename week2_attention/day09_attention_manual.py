"""
Day 9：手算完整注意力权重矩阵
4个token，从X·Wq/Wk/Wv开始，逐步算出每一个数字
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)

tokens = ["the", "cat", "sat", "mat"]
X = np.array([[1.,0.],[0.,1.],[1.,1.],[0.,2.]])  # 每行是一个token的embedding
Wq = Wk = Wv = np.eye(2)  # 简化：投影矩阵=单位矩阵，实际模型中三者不同

Q, K, V = X @ Wq, X @ Wk, X @ Wv

print("=" * 60)
print("Day 9：手算完整注意力权重矩阵")
print("=" * 60)
print("\n输入 X（每行是一个token的embedding）:")
for i, t in enumerate(tokens): print(f"  '{t}': {X[i]}")

# Step 1: Q·Kᵀ
scores_raw = Q @ K.T
print(f"\nStep 1: Q·Kᵀ（行=query，列=key）:")
print(f"{'':8s}", end="")
for t in tokens: print(f"  k_{t:3s}", end="")
print()
for i, ti in enumerate(tokens):
    print(f"q_{ti:3s}  ", end="")
    for j in range(len(tokens)): print(f"  {np.dot(Q[i],K[j]):5.1f}", end="")
    print()

# Step 2: 除以 √dₖ
d_k = Q.shape[-1]
scores_scaled = scores_raw / np.sqrt(d_k)
print(f"\nStep 2: 除以 √{d_k} = {np.sqrt(d_k):.4f}")
print(scores_scaled)

# Step 3: softmax
def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

weights = softmax(scores_scaled)
print(f"\nStep 3: softmax（验证第0行）:")
row0 = scores_scaled[0]
exp_row0 = np.exp(row0 - row0.max())
print(f"  exp后: {exp_row0}  ÷ {exp_row0.sum():.4f} = {exp_row0/exp_row0.sum()}")
print(f"  代码:  {weights[0]}  匹配: {np.allclose(exp_row0/exp_row0.sum(), weights[0])}")
print(f"\n完整权重矩阵（每行之和={weights.sum(axis=1).mean():.1f}）:")
for i, ti in enumerate(tokens):
    print(f"  '{ti}': {weights[i]}")

# Step 4: × V
output = weights @ V
print(f"\nStep 4: output = weights × V")
print(f"'cat'的输出（第1行）:")
for j, tj in enumerate(tokens):
    print(f"  {weights[1,j]:.4f} × {V[j]} = {weights[1,j]*V[j]}")
print(f"  加总 = {output[1]}")

print(f"\n每个token最关注谁:")
for i, ti in enumerate(tokens):
    top_j = weights[i].argmax()
    print(f"  '{ti}' → '{tokens[top_j]}'（{weights[i,top_j]:.4f}）原因: q·k点积最大")
