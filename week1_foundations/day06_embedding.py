"""
词嵌入 Embedding — 纯 numpy 实现
Day 6：从查找表到语义向量，理解 Embedding 层的完整工作原理
"""

import numpy as np
np.random.seed(42)


def build_token_dict(texts):
    tokens = set()
    for text in texts:
        tokens.update(text.lower().split())
    token_to_id = {tok: i for i, tok in enumerate(sorted(tokens))}
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    return token_to_id, id_to_token


class Embedding:
    """本质：(vocab_size, d_model) 的矩阵，按 id 取行（查表）"""
    def __init__(self, vocab_size, d_model):
        self.W = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.last_ids = None

    def forward(self, token_ids):
        self.last_ids = token_ids
        return self.W[token_ids]           # numpy 花式索引 = 查表

    def backward(self, grad, lr):
        # 稀疏更新：只更新用到的那些行
        for i, token_id in enumerate(self.last_ids):
            self.W[token_id] -= lr * grad[i]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def most_similar(query_vec, embedding, id_to_token, top_k=5):
    sims = [(id_to_token[i], cosine_similarity(query_vec, embedding.W[i]))
            for i in range(embedding.vocab_size)]
    return sorted(sims, key=lambda x: -x[1])[:top_k]

def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def train_embeddings(corpus, token_to_id, id_to_token,
                     d_model=8, epochs=300, lr=0.05, window=2):
    """简化版 Word2Vec CBOW：用上下文词预测中心词"""
    vocab_size = len(token_to_id)
    emb = Embedding(vocab_size, d_model)
    W_out = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)

    samples = []
    for text in corpus:
        words = text.lower().split()
        ids = [token_to_id[w] for w in words if w in token_to_id]
        for center in range(len(ids)):
            context = [ids[center+o] for o in range(-window, window+1)
                       if o != 0 and 0 <= center+o < len(ids)]
            if context:
                samples.append((context, ids[center]))

    print(f"训练样本数: {len(samples)}")
    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(samples)
        for context_ids, target_id in samples:
            # 前向
            h = emb.forward(context_ids).mean(axis=0)
            probs = softmax(W_out @ h)
            total_loss += -np.log(probs[target_id] + 1e-8)
            # 反向
            d_scores = probs.copy(); d_scores[target_id] -= 1
            W_out -= lr * np.outer(d_scores, h)
            d_h = W_out.T @ d_scores
            emb.backward(np.tile(d_h / len(context_ids), (len(context_ids), 1)), lr)
        if epoch % 60 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d} | Loss: {total_loss/len(samples):.4f}")
    return emb, W_out


print("=" * 56)
print("词嵌入 Embedding — 纯 numpy 实现（Day 6）")
print("=" * 56)

corpus = [
    "the cat sat on the mat", "the cat ate the fish",
    "the dog ran in the park", "the dog chased the cat",
    "the animal sat and ate", "cat and dog are animals",
    "fish and cat sat together", "the dog ate the fish",
    "animals ran in the park", "the cat ran fast",
    "the dog sat on mat", "cat is an animal", "dog is an animal",
]

token_to_id, id_to_token = build_token_dict(corpus)
print(f"\n词表大小: {len(token_to_id)}")

cat_id, dog_id, the_id = token_to_id['cat'], token_to_id['dog'], token_to_id['the']

print("\n训练过程:")
emb, W_out = train_embeddings(corpus, token_to_id, id_to_token)

print(f"\n训练后（语义编码进向量）:")
print(f"  cos(cat, dog) = {cosine_similarity(emb.W[cat_id], emb.W[dog_id]):+.4f}  ← 都是动物")
print(f"  cos(cat, the) = {cosine_similarity(emb.W[cat_id], emb.W[the_id]):+.4f}   ← 功能词，应该远")

print(f"\n和 'cat' 最相似的词:")
for token, sim in most_similar(emb.W[cat_id], emb, id_to_token, top_k=5):
    print(f"  {token:10s} {sim:+.4f}  {'█' * int((sim+1)*15)}")

# Embedding 本质演示
seq = ['the', 'cat', 'sat']
ids = [token_to_id[w] for w in seq]
vecs = emb.forward(ids)
print(f"\nEmbedding 本质：按 id 取矩阵的行")
print(f"输入 {seq} → id {ids} → 形状 {vecs.shape}")
