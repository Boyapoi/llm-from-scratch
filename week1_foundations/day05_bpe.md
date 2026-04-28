# Day 5：BPE 字节对编码

## 为什么不用字符或单词

| 方案 | 问题 |
|------|------|
| 单字符 | 序列太长，字符间语义关联弱，模型难学 |
| 完整单词 | 词表巨大（英语几十万词），新词（ChatGPT）直接变成 `[UNK]` |
| **BPE subword** | 两全其美：常见词保留完整，罕见词拆成更小片段 |

---

## BPE 算法三步

```
初始化：把每个词拆成字符序列 + 词尾标记
         "low" → ('l', 'o', 'w', '</w>')

统计：找出所有相邻 token 对里频率最高的一对
         ('n', 'e') 出现 14 次 → 合并为 'ne'

重复：合并后重新统计，再找最高频对
         直到达到目标词表大小
```

---

## 为什么词尾要加 `</w>`

没有词尾标记，`'ab'` 在 `'ab'` 里和在 `'abc'` 里无法区分，会把不同词的字符错误拼接。

```
'ab'  → ('a','b','</w>')     ← 独立的词
'abc' → ('a','b','c','</w>') ← 不同的词，'ab' 只是前缀
```

---

## 逐步推导（语料示例）

语料：`['low'×5, 'lower'×2, 'newest'×6, 'new'×8]`

**初始词表：**
```
('l','o','w','</w>')           : 5
('l','o','w','e','r','</w>')   : 2
('n','e','w','e','s','t','</w>'): 6
('n','e','w','</w>')           : 8
```

**Step 1：统计相邻对频率**
```
('n','e') : 6+8 = 14   ← 最高频
('e','w') : 6+8 = 14   ← 并列（按字母序选）
('l','o') : 5+2 = 7
...
```

**Step 1 执行合并 `('n','e')` → `'ne'`：**
```
('ne','w','</w>')              : 8
('ne','w','e','s','t','</w>')  : 6
...
```

---

## 合并顺序为什么重要

**方案A**：先合并 `('a','b')`，再合并 `('ab','c')`
```
'abc' → ['a','b','c','</w>'] → ['ab','c','</w>'] → ['abc','</w>']
```

**方案B**：先合并 `('ab','c')`，再合并 `('a','b')`
```
'abc' → ['a','b','c','</w>'] → 找不到'ab' → 不变 → ['ab','c','</w>']
```

结果完全不同。`merges` 列表必须按顺序应用，这是 BPE 的核心。

---

## 对新词编码的能力

训练时没见过 `'unknown'`，BPE 不崩溃，而是拆成已知片段：

```python
encode('unknown', merges)
# → ['u', 'n', 'k', 'n', 'ow', 'n', '</w>']
# 'ow' 被合并过所以保留，其他退化成字符
```

**零 OOV（Out-of-Vocabulary）问题**是 BPE 最大的优势。

---

## 真实 LLM 的规模

| 模型 | 合并规则数 | 词表大小 |
|------|-----------|---------|
| GPT-2 | 50,000 | ~50,257 |
| GPT-4 | ~100,000 | ~100,256 |
| LLaMA 3 | 128,000 | 128,000 |

我们的实现只有 10 条合并规则，但算法逻辑完全一样。

---

## 现成库

```python
# HuggingFace tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
tokenizer = Tokenizer(BPE())
tokenizer.train_from_iterator(corpus, trainer)
output = tokenizer.encode("low newer")

# tiktoken（OpenAI 用的）
import tiktoken
enc = tiktoken.get_encoding("gpt2")
enc.encode("hello world")  # [31373, 995]
```

手写的目的是搞清楚黑盒里发生了什么。理解了原理，读这些库的文档就不会懵。

---

## 运行

```bash
python day05_bpe.py
```
