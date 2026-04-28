"""
两层神经网络 — 纯 numpy 实现
对应 Day 1-3 推导的所有公式，含梯度验证

网络结构:
  x → [W1,b1] → z1 → ReLU → a1 → [W2,b2] → z2 → sigmoid → ŷ → L
"""

import numpy as np

np.random.seed(0)

def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
def cross_entropy(y, y_hat, eps=1e-8):
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

def forward(x, W1, b1, W2, b2):
    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    y_hat = sigmoid(z2)
    return y_hat, (x, z1, a1)

def backward(y, y_hat, cache, W2):
    x, z1, a1 = cache
    batch = x.shape[1]
    dz2 = (y_hat - y) / batch          # Step 1: ∂L/∂z2 = ŷ - y
    dW2 = dz2 @ a1.T                   # Step 2: ∂L/∂W2
    db2 = np.sum(dz2, axis=1, keepdims=True)
    da1 = W2.T @ dz2                   # Step 3: ∂L/∂a1 = W2ᵀ·dz2
    dz1 = da1 * relu_grad(z1)          # Step 4: 穿过 ReLU
    dW1 = dz1 @ x.T                   # Step 5: ∂L/∂W1
    db1 = np.sum(dz1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    return W1-lr*dW1, b1-lr*db1, W2-lr*dW2, b2-lr*db2

def numerical_grad(x, y, W1, b1, W2, b2, param, idx, eps=1e-5):
    orig = param.flat[idx]
    param.flat[idx] = orig + eps
    yh_p, _ = forward(x, W1, b1, W2, b2)
    loss_p = cross_entropy(y, yh_p)
    param.flat[idx] = orig - eps
    yh_m, _ = forward(x, W1, b1, W2, b2)
    loss_m = cross_entropy(y, yh_m)
    param.flat[idx] = orig
    return (loss_p - loss_m) / (2 * eps)

def train(X, y, n_hid=8, lr=0.5, epochs=600, print_every=100):
    n_in, n_out = X.shape[0], y.shape[0]
    W1 = np.random.randn(n_hid, n_in) * 0.1
    b1 = np.zeros((n_hid, 1))
    W2 = np.random.randn(n_out, n_hid) * 0.1
    b2 = np.zeros((n_out, 1))
    for epoch in range(epochs):
        y_hat, cache = forward(X, W1, b1, W2, b2)
        loss = cross_entropy(y, y_hat)
        dW1, db1, dW2, db2 = backward(y, y_hat, cache, W2)
        W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)
        if epoch % print_every == 0 or epoch == epochs - 1:
            acc = np.mean((y_hat > 0.5) == y)
            print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {acc:.1%}")
    return W1, b1, W2, b2

print("=" * 56)
print("两层神经网络 — 纯 numpy 实现（Day 4）")
print("=" * 56)

X = np.array([[0,0,1,1],[0,1,0,1]], dtype=float)
y = np.array([[0,1,1,0]], dtype=float)
print("\n任务：XOR（线性不可分）\n")
W1, b1, W2, b2 = train(X, y)

y_hat, cache = forward(X, W1, b1, W2, b2)
dW1, db1, dW2, db2 = backward(y, y_hat, cache, W2)
print(f"\n最终准确率: {np.mean((y_hat > 0.5) == y):.0%}")

print("\n梯度验证（差值 < 1e-5 则正确）:")
checks = [("W2[0,0]",W2,dW2,0),("W1[0,0]",W1,dW1,0),("b1[0,0]",b1,db1,0)]
for name, param, grad, idx in checks:
    a = grad.flat[idx]
    n = numerical_grad(X, y, W1, b1, W2, b2, param, idx)
    print(f"  {name}: analytic={a:+.6f}  numeric={n:+.6f}  diff={abs(a-n):.2e}  {'✓' if abs(a-n)<1e-5 else '✗'}")
