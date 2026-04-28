"""
BPE (Byte Pair Encoding) — 纯 Python 实现
Day 5：从零实现分词算法，可视化每一步合并过程
"""

from collections import defaultdict, Counter


def build_vocab(corpus):
    """把每个词拆成字符元组，词尾加 </w> 标记边界"""
    vocab = defaultdict(int)
    for word in corpus:
        vocab[tuple(list(word) + ['</w>'])] += 1
    return dict(vocab)


def get_pair_freqs(vocab):
    """统计所有相邻 token 对的频率"""
    pairs = defaultdict(int)
    for tokens, freq in vocab.items():
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i+1])] += freq
    return dict(pairs)


def merge_pair(pair, vocab):
    """把词表里所有出现 pair 的地方合并成一个新 token"""
    new_vocab = {}
    merged = ''.join(pair)
    for tokens, freq in vocab.items():
        new_tokens, i = [], 0
        while i < len(tokens):
            if i < len(tokens)-1 and tokens[i]==pair[0] and tokens[i+1]==pair[1]:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_vocab[tuple(new_tokens)] = freq
    return new_vocab


def train_bpe(corpus, num_merges, verbose=True):
    """执行 num_merges 次合并，返回有序合并规则列表"""
    vocab = build_vocab(corpus)
    if verbose:
        print("初始词表:")
        for tokens, freq in sorted(vocab.items(), key=lambda x: -x[1]):
            print(f"  {' '.join(tokens):30s} 频率: {freq}")
        print()
    merges = []
    for step in range(num_merges):
        pair_freqs = get_pair_freqs(vocab)
        if not pair_freqs:
            break
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
        vocab = merge_pair(best_pair, vocab)
        merges.append(best_pair)
        if verbose:
            print(f"Step {step+1:2d}: 合并 {str(best_pair):20s} → '{''.join(best_pair)}'  (频率 {pair_freqs[best_pair]})")
    return merges, vocab


def encode(word, merges):
    """用训练好的合并规则对新词编码（必须按顺序应用）"""
    tokens = list(word) + ['</w>']
    for pair in merges:
        merged, i, new_tokens = ''.join(pair), 0, []
        while i < len(tokens):
            if i < len(tokens)-1 and tokens[i]==pair[0] and tokens[i+1]==pair[1]:
                new_tokens.append(merged); i += 2
            else:
                new_tokens.append(tokens[i]); i += 1
        tokens = new_tokens
    return tokens


print("=" * 56)
print("BPE 算法完整实现（Day 5）")
print("=" * 56)

corpus = ['low']*5 + ['lower']*2 + ['newest']*6 + ['widest']*3 + ['new']*8 + ['wide']*4

print(f"\n语料 ({len(corpus)} 个词):")
for word, cnt in Counter(corpus).most_common():
    print(f"  '{word}': {cnt} 次")
print()

merges, final_vocab = train_bpe(corpus, num_merges=10)

print("\n" + "=" * 56)
print("对新词编码（应用训练好的合并规则）")
print("=" * 56)
for word in ['low', 'lower', 'newest', 'newlow', 'unknown']:
    print(f"  '{word}' → {encode(word, merges)}")

all_tokens = set()
for tokens in final_vocab:
    all_tokens.update(tokens)
subwords = sorted([t for t in all_tokens if len(t) > 1])
print(f"\n合并出的 subword: {subwords}")
print(f"词表大小: {len(all_tokens)} tokens")

print("\n合并规则（GPT-2 有 5 万条）:")
for i, (a, b) in enumerate(merges, 1):
    print(f"  Rule {i:2d}: '{a}' + '{b}' → '{a+b}'")
