# Day 10-11: GPT Implementation

## 🎯 목표

**GPT를 구현하고 Text Generation 전략 마스터!**

```python
# GPT = Generative Pre-trained Transformer
# 핵심: 다음 단어 예측 (Autoregressive)!

"The cat sat on the" → "mat" ✅
```

---

## 📖 1. GPT 아키텍처

### Decoder-only Transformer

```
GPT vs BERT:

BERT (Encoder-only):
- Bidirectional (양방향)
- Masked attention
- 용도: Understanding (분류, NER, QA)

GPT (Decoder-only):
- Unidirectional (단방향)
- Causal attention (미래 못 봄)
- 용도: Generation (텍스트 생성)
```

### 구현

```python
import torch
import torch.nn as nn
import math

class GPTBlock(nn.Module):
    """GPT Decoder Block = Masked Self-Attention + FFN"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # 1. Masked Self-Attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # 2. Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT uses GELU instead of ReLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # 3. Layer Norm (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN: Norm → Attention → Residual
        normed = self.norm1(x)
        attn_out = self.attention(normed, normed, normed, mask)
        x = x + self.dropout(attn_out)

        # Pre-LN: Norm → FFN → Residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x

class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_len=1024,
        dropout=0.1
    ):
        super().__init__()

        # Token + Position Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Decoder blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection (language model head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights (embedding과 output projection 공유)
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        # Position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        x = self.dropout(token_emb + position_emb)

        # Causal mask (prevent looking ahead)
        mask = self.create_causal_mask(seq_len).to(input_ids.device)

        # Decoder blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final norm
        x = self.ln_f(x)

        # Language model head
        logits = self.lm_head(x)

        return logits  # (batch, seq_len, vocab_size)

    def create_causal_mask(self, seq_len):
        """
        Causal mask: 미래를 볼 수 없게
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
        """
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        return mask
```

---

## 🎯 2. Pre-training: Causal Language Modeling

### 목표

**다음 토큰 예측!**

```python
Input:  "The cat sat on"
Target: "cat sat on the"

# Shift by 1!
```

### 훈련

```python
def train_gpt(model, dataloader, optimizer, device):
    model.train()

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        # input_ids: "The cat sat on the mat"

        # Shift for target
        # Input:  "The cat sat on the"
        # Target: "cat sat on the mat"
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        # Forward
        logits = model(inputs)  # (batch, seq_len-1, vocab_size)

        # Loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return loss.item()

# 훈련 루프
model = GPTModel(vocab_size=50257).to(device)  # GPT-2 vocab size
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

for epoch in range(num_epochs):
    loss = train_gpt(model, train_loader, optimizer, device)
    print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

---

## 🚀 3. Generation Strategies

### 3.1 Greedy Decoding

**항상 가장 확률 높은 토큰 선택**

```python
@torch.no_grad()
def greedy_generate(model, prompt_ids, max_length=50, eos_token_id=50256):
    model.eval()

    input_ids = prompt_ids.clone()

    for _ in range(max_length):
        # Forward
        logits = model(input_ids)  # (1, seq_len, vocab_size)

        # Next token: argmax
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

        # Append
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        # Stop if EOS
        if next_token_id.item() == eos_token_id:
            break

    return input_ids

# 사용
prompt = "Once upon a time"
prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
output_ids = greedy_generate(model, prompt_ids)
output_text = tokenizer.decode(output_ids[0])

print(output_text)
# "Once upon a time there was a king. The king was very powerful..."
```

**문제**: 반복적이고 지루함!

---

### 3.2 Beam Search

**여러 후보를 동시에 탐색**

```python
@torch.no_grad()
def beam_search(model, prompt_ids, beam_size=5, max_length=50):
    model.eval()

    # 초기 beam
    sequences = [(prompt_ids, 0.0)]  # (sequence, score)

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            # Forward
            logits = model(seq)
            next_token_logits = logits[:, -1, :]

            # Log probabilities
            log_probs = torch.log_softmax(next_token_logits, dim=-1)

            # Top-k candidates
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                candidate_seq = torch.cat([seq, topk_ids[:, i:i+1]], dim=1)
                candidate_score = score + topk_log_probs[0, i].item()
                all_candidates.append((candidate_seq, candidate_score))

        # Keep top beam_size sequences
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = sequences[:beam_size]

        # Check if all beams ended
        if all(seq[0, -1].item() == eos_token_id for seq, _ in sequences):
            break

    # Return best sequence
    best_seq, _ = sequences[0]
    return best_seq

# 사용
output_ids = beam_search(model, prompt_ids, beam_size=5)
output_text = tokenizer.decode(output_ids[0])
```

**장점**: Greedy보다 더 좋은 결과
**단점**: 여전히 반복적, 느림

---

### 3.3 Sampling (Top-k & Top-p)

#### Top-k Sampling

**상위 k개 토큰 중 랜덤 샘플링**

```python
@torch.no_grad()
def top_k_sampling(model, prompt_ids, k=50, max_length=50, temperature=1.0):
    model.eval()

    input_ids = prompt_ids.clone()

    for _ in range(max_length):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature  # Temperature scaling

        # Top-k filtering
        topk_logits, topk_indices = torch.topk(next_token_logits, k)

        # Sample from top-k
        probs = torch.softmax(topk_logits, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1)
        next_token_id = topk_indices.gather(-1, next_token_idx)

        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        if next_token_id.item() == eos_token_id:
            break

    return input_ids

# 사용
output_ids = top_k_sampling(model, prompt_ids, k=50, temperature=0.8)
```

#### Top-p (Nucleus) Sampling

**누적 확률이 p가 될 때까지의 토큰들에서 샘플링**

```python
@torch.no_grad()
def top_p_sampling(model, prompt_ids, p=0.9, max_length=50, temperature=1.0):
    model.eval()

    input_ids = prompt_ids.clone()

    for _ in range(max_length):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature

        # Sort by probability
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Remove tokens with cumulative probability > p
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Set logits to -inf
        sorted_logits[sorted_indices_to_remove] = float('-inf')

        # Sample
        probs = torch.softmax(sorted_logits, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1)
        next_token_id = sorted_indices.gather(-1, next_token_idx)

        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        if next_token_id.item() == eos_token_id:
            break

    return input_ids

# 사용 (GPT-3 기본값)
output_ids = top_p_sampling(model, prompt_ids, p=0.95, temperature=0.7)
```

### Temperature의 영향

```python
# Temperature = 0.1 (Conservative)
"Once upon a time there was a king."

# Temperature = 1.0 (Balanced)
"Once upon a time there lived a curious fox."

# Temperature = 2.0 (Creative but risky)
"Once upon a time quantum bananas danced merrily."
```

---

## 📊 4. Generation 전략 비교

| Strategy | 장점 | 단점 | 사용처 |
|----------|------|------|--------|
| Greedy | 빠름, 결정적 | 반복적, 지루함 | 짧은 생성 |
| Beam Search | 높은 품질 | 반복적, 느림 | 번역, 요약 |
| Top-k | 다양성 ↑ | k 선택 어려움 | Creative writing |
| Top-p | 동적, 일관성 | 때때로 불안정 | ChatGPT (기본) |

### 실전 설정

```python
# ChatGPT 스타일
generation_config = {
    'top_p': 0.95,
    'temperature': 0.7,
    'max_length': 2048,
    'repetition_penalty': 1.2  # 반복 억제
}

# 번역/요약 (정확성 중요)
generation_config = {
    'beam_size': 4,
    'temperature': 0.3,
    'max_length': 512
}

# Creative writing (다양성 중요)
generation_config = {
    'top_p': 0.9,
    'temperature': 1.0,
    'max_length': 1024
}
```

---

## 🎯 5. Fine-tuning GPT

### Instruction Following

```python
# Format
instruction = "Summarize the following text:"
input_text = "Long article..."
output = "Summary..."

# Prompt
prompt = f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}
"""

# Fine-tune on this format!
```

### Few-shot Prompting (GPT-3)

```python
# Zero-shot
prompt = "Translate to French: Hello"

# Few-shot (더 좋음!)
prompt = """
Translate to French:
English: Hello → French: Bonjour
English: Goodbye → French: Au revoir
English: Thank you → French: Merci
English: How are you? → French:
"""

# GPT-3: 문맥에서 학습!
```

---

## 💡 6. GPT Variants

### GPT-2 (OpenAI, 2019)

- 1.5B parameters
- 40GB text
- Zero-shot learning 가능!

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2-large')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

input_ids = tokenizer.encode("Hello, my name is", return_tensors='pt')
output = model.generate(
    input_ids,
    max_length=50,
    top_p=0.95,
    temperature=0.7
)

print(tokenizer.decode(output[0]))
```

### GPT-3 (OpenAI, 2020)

- 175B parameters
- 570GB text
- Few-shot learning 극강!

**핵심**: Pre-training만으로 충분!

### GPT-4 (OpenAI, 2023)

- Multimodal (text + image)
- 훨씬 강력한 reasoning
- (구조는 비공개)

---

## 🎓 학습 목표

- [ ] GPT 아키텍처 구현 (Decoder-only)
- [ ] Causal language modeling 이해
- [ ] Greedy, Beam Search, Top-k, Top-p 구현
- [ ] Temperature의 역할 이해
- [ ] Fine-tuning vs Few-shot prompting

---

## 🔧 실전: HuggingFace Transformers

### 간단한 사용

```python
from transformers import pipeline

# Text generation
generator = pipeline('text-generation', model='gpt2')
output = generator(
    "Once upon a time",
    max_length=50,
    top_p=0.95,
    temperature=0.7
)

print(output[0]['generated_text'])
```

### Fine-tuning

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# 모델 & 토크나이저
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Training arguments
training_args = TrainingArguments(
    output_dir='./gpt2-finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# 훈련
trainer.train()
```

### Generation with Control

```python
# Repetition penalty
output = model.generate(
    input_ids,
    max_length=100,
    repetition_penalty=1.2,  # > 1: 반복 억제
    no_repeat_ngram_size=3   # 3-gram 반복 금지
)

# Length penalty (beam search)
output = model.generate(
    input_ids,
    num_beams=5,
    length_penalty=2.0,  # > 1: 긴 문장 선호
    early_stopping=True
)

# Diverse beam search
output = model.generate(
    input_ids,
    num_beams=5,
    num_beam_groups=5,  # 각 group이 다른 방향 탐색
    diversity_penalty=1.0
)
```

---

## 📚 더 읽기

### Papers
- **GPT**: Improving Language Understanding by Generative Pre-Training (OpenAI, 2018)
- **GPT-2**: Language Models are Unsupervised Multitask Learners (OpenAI, 2019)
- **GPT-3**: Language Models are Few-Shot Learners (OpenAI, 2020)

### 리소스
- [HuggingFace Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

---

## ⏭️ 다음 단계

Phase 2 완료! 🎉

이제 **BERT (양방향)** 와 **GPT (단방향)** 를 모두 이해했습니다!

👉 [Phase 3: Diffusion Models](../phase3-diffusion/)로 진행하여 이미지 생성을 배워봅시다!

**"Text를 넘어 Image로!"** 🎨
