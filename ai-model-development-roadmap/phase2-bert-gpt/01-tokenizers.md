# Week 3: Tokenizers - Text를 Token으로!

## 🎯 목표

**토크나이저를 직접 구현하여 완벽히 이해하기**

```python
# 우리가 구현할 것
"Hello, world!" → ["Hello", ",", "world", "!"] → [101, 263, 145, 102]
                 Tokenization              Encoding

# BPE 알고리즘을 밑바닥부터!
```

---

## 📚 1. Why Tokenization?

### 문제: 단어 vs 글자

**단어 단위 (Word-level)**
```python
vocab = {"the": 1, "cat": 2, "sat": 3, ...}
# 문제: Vocabulary 너무 큼 (수백만 단어)
# "supercalifragilisticexpialidocious" → [UNK]  # OOV!
```

**글자 단위 (Character-level)**
```python
vocab = {"a": 1, "b": 2, ..., "z": 26}
# 문제: 시퀀스 너무 길어짐
# "transformer" → 11 tokens (비효율!)
```

### 해결: Subword Tokenization

```python
"transformer" → ["trans", "former"]  # 2 tokens!
"supercalifragilistic" → ["super", "cal", "ifrag", "ilistic"]
# → 적절한 크기의 vocab (30K-50K)
# → 짧은 시퀀스
# → OOV 문제 해결!
```

---

## 🔧 2. BPE (Byte Pair Encoding)

### 핵심 아이디어

**가장 빈번한 글자 쌍을 계속 합치기!**

```
초기: ["l", "o", "w", "e", "r"]
     ["l", "o", "w"]
     ["n", "e", "w", "e", "s", "t"]

Step 1: "e" + "r" 가장 빈번 → "er" 추가
       ["l", "o", "w", "er"]
       ["n", "e", "w", "e", "s", "t"]

Step 2: "er" + "_" (공백) → "er_"
       ["l", "o", "w", "er_"]

... 계속 반복 ...
```

### 알고리즘 구현

```python
import re
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}  # (pair) -> new_token
        self.vocab = {}   # token -> id

    def train(self, texts):
        """BPE 훈련: Merge rules 학습"""

        # 1. 초기 vocabulary (글자 단위)
        word_freqs = self._get_word_frequencies(texts)

        # 단어를 글자로 split (끝에 </w> 추가)
        splits = {
            word: [c for c in word[:-1]] + [word[-1] + '</w>']
            for word in word_freqs.keys()
        }

        # 2. Merge 반복
        while len(self.vocab) < self.vocab_size:
            # 가장 빈번한 pair 찾기
            pairs = self._get_pair_frequencies(splits, word_freqs)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)

            # Merge 수행
            splits = self._merge_pair(best_pair, splits)
            self.merges[best_pair] = best_pair[0] + best_pair[1]

        # 3. Vocabulary 구축
        self._build_vocab(splits)

    def _get_word_frequencies(self, texts):
        """단어 빈도 계산"""
        word_freqs = Counter()
        for text in texts:
            words = text.lower().split()
            word_freqs.update(words)
        return word_freqs

    def _get_pair_frequencies(self, splits, word_freqs):
        """인접한 글자 쌍의 빈도 계산"""
        pair_freqs = defaultdict(int)

        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue

            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq

        return pair_freqs

    def _merge_pair(self, pair, splits):
        """특정 pair를 merge"""
        new_splits = {}

        for word, split in splits.items():
            new_split = []
            i = 0

            while i < len(split):
                # Pair 발견하면 merge
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    new_split.append(split[i] + split[i + 1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1

            new_splits[word] = new_split

        return new_splits

    def _build_vocab(self, splits):
        """최종 vocabulary 구축"""
        vocab = set()
        for split in splits.values():
            vocab.update(split)

        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        self.vocab['<PAD>'] = len(self.vocab)
        self.vocab['<UNK>'] = len(self.vocab)

    def encode(self, text):
        """Text → Token IDs"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

    def tokenize(self, text):
        """Text → Tokens (apply merges)"""
        words = text.lower().split()

        tokens = []
        for word in words:
            # 글자로 split
            word_tokens = [c for c in word[:-1]] + [word[-1] + '</w>']

            # Merge rules 적용
            for pair, merged in self.merges.items():
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == pair:
                        word_tokens = (
                            word_tokens[:i] +
                            [merged] +
                            word_tokens[i + 2:]
                        )
                    else:
                        i += 1

            tokens.extend(word_tokens)

        return tokens

    def decode(self, ids):
        """Token IDs → Text"""
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [reverse_vocab.get(id, '<UNK>') for id in ids]
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text

# 사용 예시
texts = [
    "the quick brown fox jumps over the lazy dog",
    "the fox is quick and the dog is lazy",
    "a quick brown dog"
]

tokenizer = BPETokenizer(vocab_size=50)
tokenizer.train(texts)

# Encoding
text = "the quick fox"
ids = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Token IDs: {ids}")
print(f"Tokens: {tokenizer.tokenize(text)}")
```

**출력**:
```
Text: the quick fox
Token IDs: [42, 38, 29, 45]
Tokens: ['the</w>', 'qu', 'ick</w>', 'fox</w>']
```

---

## 🎭 3. WordPiece (BERT)

### BPE와의 차이

**BPE**: 빈도 기반
```python
# 가장 빈번한 pair 선택
best_pair = max(pairs, key=lambda p: count[p])
```

**WordPiece**: Likelihood 기반
```python
# Likelihood를 최대로 증가시키는 pair 선택
score = freq(pair) / (freq(first) * freq(second))
best_pair = max(pairs, key=lambda p: score[p])
```

### 구현

```python
import math

class WordPieceTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}

    def train(self, texts):
        """WordPiece 훈련"""
        word_freqs = Counter()
        for text in texts:
            word_freqs.update(text.lower().split())

        # 초기 vocab (글자)
        vocab = set()
        for word in word_freqs:
            vocab.update(word)

        # Merge 반복
        while len(vocab) < self.vocab_size:
            # Score 계산
            pairs = self._get_pairs(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=lambda p: self._score(p, word_freqs))

            # Merge
            new_token = best_pair[0] + best_pair[1]
            vocab.add(new_token)

        # Vocabulary 구축
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        self.vocab['[PAD]'] = len(self.vocab)
        self.vocab['[UNK]'] = len(self.vocab)
        self.vocab['[CLS]'] = len(self.vocab)
        self.vocab['[SEP]'] = len(self.vocab)
        self.vocab['[MASK]'] = len(self.vocab)

    def _score(self, pair, word_freqs):
        """Likelihood score"""
        first, second = pair
        pair_freq = sum(
            freq for word, freq in word_freqs.items()
            if first + second in word
        )
        first_freq = sum(freq for word, freq in word_freqs.items() if first in word)
        second_freq = sum(freq for word, freq in word_freqs.items() if second in word)

        if first_freq == 0 or second_freq == 0:
            return 0

        return pair_freq / (first_freq * second_freq)

    def tokenize(self, text):
        """Longest match first (greedy)"""
        tokens = []
        for word in text.lower().split():
            # Subword로 분할
            start = 0
            sub_tokens = []

            while start < len(word):
                end = len(word)
                found = False

                # Longest match 찾기
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = '##' + substr  # Continuation marker

                    if substr in self.vocab:
                        sub_tokens.append(substr)
                        found = True
                        break

                    end -= 1

                if not found:
                    sub_tokens.append('[UNK]')
                    start += 1
                else:
                    start = end

            tokens.extend(sub_tokens)

        return tokens

# 사용 예시
wp_tokenizer = WordPieceTokenizer(vocab_size=100)
wp_tokenizer.train(texts)

text = "unbelievable"
tokens = wp_tokenizer.tokenize(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
# Output: ['un', '##believ', '##able']
```

---

## 🚀 4. HuggingFace Tokenizers (Fast!)

### Rust 기반 고속 구현

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. Tokenizer 초기화
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()

# 2. 훈련
trainer = BpeTrainer(
    special_tokens=["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"],
    vocab_size=1000
)

files = ["train.txt"]
tokenizer.train(files, trainer)

# 3. 사용
output = tokenizer.encode("Hello, world!")
print(f"Tokens: {output.tokens}")
print(f"IDs: {output.ids}")

# 4. 저장
tokenizer.save("my_tokenizer.json")
```

### Pre-trained 사용

```python
from transformers import AutoTokenizer

# GPT-2 (BPE)
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(gpt2_tokenizer.tokenize("Hello, world!"))
# ['Hello', ',', 'Ġworld', '!']  # Ġ = space

# BERT (WordPiece)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(bert_tokenizer.tokenize("unbelievable"))
# ['un', '##believable']

# Special tokens
encoded = bert_tokenizer(
    "Hello, world!",
    padding="max_length",
    max_length=10,
    truncation=True,
    return_tensors="pt"
)

print(encoded)
# {
#   'input_ids': tensor([[101, 7592, 1010, 2088, 999, 102, 0, 0, 0, 0]]),
#   'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
# }
```

---

## 🎯 5. SentencePiece

### 언어 독립적 Tokenizer

**특징**:
- Raw text 처리 (공백도 token으로!)
- 언어 독립적
- Reversible (완벽한 복원)

```python
import sentencepiece as spm

# 훈련
spm.SentencePieceTrainer.train(
    input='train.txt',
    model_prefix='m',
    vocab_size=1000,
    model_type='bpe'  # or 'unigram'
)

# 로드
sp = spm.SentencePieceProcessor()
sp.load('m.model')

# Encoding
tokens = sp.encode_as_pieces('This is a test')
print(tokens)
# ['▁This', '▁is', '▁a', '▁test']  # ▁ = space

ids = sp.encode_as_ids('This is a test')
print(ids)
# [215, 32, 8, 156]

# Decoding (완벽한 복원!)
text = sp.decode_pieces(tokens)
print(text)
# 'This is a test'
```

---

## 🔍 6. Tokenizer 비교

| Algorithm | Used By | 장점 | 단점 |
|-----------|---------|------|------|
| BPE | GPT-2, RoBERTa | 간단, 효율적 | 확률적 근거 약함 |
| WordPiece | BERT | Likelihood 기반 | 느림 |
| Unigram | T5, XLNet | 확률적으로 최적 | 복잡함 |
| SentencePiece | LLaMA, GPT-4 | 언어 독립적 | - |

### Vocabulary 크기 선택

```python
# Vocabulary size의 영향

# 작은 vocab (10K):
"unbelievable" → ['un', 'be', 'lie', 'v', 'able']
# 장점: 적은 메모리
# 단점: 긴 시퀀스

# 큰 vocab (100K):
"unbelievable" → ['unbelievable']
# 장점: 짧은 시퀀스
# 단점: 많은 메모리, 희소 임베딩

# 일반적인 선택:
# - BERT: 30K (WordPiece)
# - GPT-2: 50K (BPE)
# - GPT-3: 50K (BPE)
# - LLaMA: 32K (SentencePiece)
```

---

## 🎓 학습 목표

- [ ] BPE 알고리즘 직접 구현
- [ ] WordPiece와 BPE의 차이 이해
- [ ] HuggingFace Tokenizers 사용
- [ ] SentencePiece 이해
- [ ] Vocabulary 크기 trade-off 이해

---

## 💡 실전 팁

### 1. Special Tokens

```python
# BERT
special_tokens = {
    '[PAD]': 0,   # Padding
    '[UNK]': 1,   # Unknown
    '[CLS]': 2,   # Classification
    '[SEP]': 3,   # Separator
    '[MASK]': 4   # Masked token
}

# GPT-2
special_tokens = {
    '<|endoftext|>': 50256  # End of document
}
```

### 2. Fast Tokenizer

```python
# Slow (Python)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Fast (Rust)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# 100배 빠름!
```

### 3. Custom Vocabulary

```python
from transformers import BertTokenizer

# Vocabulary 확장
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

new_tokens = ['<EMOJI>', '<URL>', '<CODE>']
tokenizer.add_tokens(new_tokens)

# Model embedding 확장 필요
model.resize_token_embeddings(len(tokenizer))
```

---

## ⏭️ 다음

👉 [Day 8-9: BERT Implementation](./02-bert-implementation.md)

**이제 토크나이저를 만들었으니, BERT를 훈련시켜봅시다!** 🚀
