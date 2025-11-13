# Day 8-9: BERT Implementation

## 🎯 목표

**BERT를 밑바닥부터 구현하여 Pre-training과 Fine-tuning 완벽 이해**

```python
# BERT = Bidirectional Encoder Representations from Transformers
# 핵심: 양방향 Context 학습!

"The cat [MASK] on the mat"
→ BERT: "sat" (왼쪽 + 오른쪽 context 모두 활용!)
```

---

## 📖 1. BERT 아키텍처

### Encoder-only Transformer

```
Input: "Hello [MASK] world"
  ↓
Token Embedding + Position Embedding + Segment Embedding
  ↓
[Encoder Block] × 12  # BERT-base
  ↓
[CLS] representation → Classification
[MASK] representation → MLM prediction
```

### 구현

```python
import torch
import torch.nn as nn
import math

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=768, max_len=512):
        super().__init__()

        # 1. Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Position Embedding (learned)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # 3. Segment Embedding
        self.segment_embedding = nn.Embedding(2, d_model)  # 0 or 1

        # LayerNorm + Dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids):
        # input_ids: (batch, seq_len)
        # segment_ids: (batch, seq_len)

        seq_len = input_ids.size(1)

        # Position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        segment_emb = self.segment_embedding(segment_ids)

        # Sum
        embeddings = token_emb + position_emb + segment_emb

        # Normalize & Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BERTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_len=512,
        dropout=0.1
    ):
        super().__init__()

        # Embedding
        self.embedding = BERTEmbedding(vocab_size, d_model, max_len)

        # Encoder blocks (from Phase 1!)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.d_model = d_model

    def forward(self, input_ids, segment_ids, attention_mask=None):
        # Embedding
        x = self.embedding(input_ids, segment_ids)

        # Attention mask 변환
        if attention_mask is not None:
            # (batch, seq_len) → (batch, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Encoder blocks
        for encoder in self.encoder_blocks:
            x = encoder(x, attention_mask)

        return x  # (batch, seq_len, d_model)
```

---

## 🎭 2. Pre-training: MLM (Masked Language Model)

### 핵심 아이디어

**무작위로 15% 토큰을 가리고 예측!**

```
Original: "The cat sat on the mat"

Masked:   "The cat [MASK] on the mat"
          "The [MASK] sat on the [MASK]"

Task: 가려진 토큰 예측
```

### Masking 전략

```python
def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """
    15% 토큰 masking:
    - 80%: [MASK]로 대체
    - 10%: 랜덤 토큰으로 대체
    - 10%: 그대로 유지
    """
    labels = inputs.clone()

    # 15% 확률로 mask
    probability_matrix = torch.full(labels.shape, mlm_probability)

    # Special tokens 제외
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool),
        value=0.0
    )

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Loss 계산에서 제외

    # 80%: [MASK] token
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10%: Random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # 10%: 그대로 유지 (already done)

    return inputs, labels

# 예시
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "The cat sat on the mat"
input_ids = tokenizer.encode(text, return_tensors='pt')

masked_input, labels = mask_tokens(input_ids, tokenizer)

print(f"Original: {tokenizer.decode(input_ids[0])}")
print(f"Masked:   {tokenizer.decode(masked_input[0])}")
print(f"Labels:   {labels}")
```

### MLM Head

```python
class BERTForMLM(nn.Module):
    def __init__(self, bert_model, vocab_size):
        super().__init__()
        self.bert = bert_model

        # MLM prediction head
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_model.d_model, bert_model.d_model),
            nn.GELU(),
            nn.LayerNorm(bert_model.d_model),
            nn.Linear(bert_model.d_model, vocab_size)
        )

    def forward(self, input_ids, segment_ids, attention_mask=None, labels=None):
        # BERT encoding
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)

        # Predict masked tokens
        prediction_scores = self.mlm_head(hidden_states)

        # Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # ignore_index=-100
            loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1)
            )

        return loss, prediction_scores

# 훈련
model = BERTForMLM(bert_model, vocab_size=30522)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    # Masking
    masked_input, labels = mask_tokens(input_ids, tokenizer)

    # Forward
    loss, _ = model(
        masked_input,
        segment_ids=torch.zeros_like(masked_input),
        attention_mask=attention_mask,
        labels=labels
    )

    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 📚 3. Pre-training: NSP (Next Sentence Prediction)

### 핵심 아이디어

**두 문장이 연속인지 판단!**

```
Sentence A: "The cat sat on the mat."
Sentence B: "It was sleeping."
Label: IsNext (1)

Sentence A: "The cat sat on the mat."
Sentence B: "The economy is growing."
Label: NotNext (0)
```

### 데이터 준비

```python
def create_nsp_data(documents):
    """NSP 데이터 생성"""
    examples = []

    for doc in documents:
        sentences = doc.split('.')  # 문장 분리

        for i in range(len(sentences) - 1):
            sentence_a = sentences[i].strip()

            # 50% IsNext
            if random.random() > 0.5:
                sentence_b = sentences[i + 1].strip()
                label = 1  # IsNext
            # 50% NotNext
            else:
                random_doc = random.choice(documents)
                random_sentences = random_doc.split('.')
                sentence_b = random.choice(random_sentences).strip()
                label = 0  # NotNext

            examples.append({
                'sentence_a': sentence_a,
                'sentence_b': sentence_b,
                'label': label
            })

    return examples

# Encoding
def encode_sentence_pair(sentence_a, sentence_b, tokenizer, max_len=128):
    """
    [CLS] sentence_a [SEP] sentence_b [SEP]
    """
    encoded = tokenizer(
        sentence_a,
        sentence_b,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )

    # encoded['input_ids']: [CLS] ... [SEP] ... [SEP]
    # encoded['token_type_ids']: [0, 0, ..., 1, 1, ...]  # Segment IDs!

    return encoded
```

### NSP Head

```python
class BERTForPreTraining(nn.Module):
    def __init__(self, bert_model, vocab_size):
        super().__init__()
        self.bert = bert_model

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_model.d_model, bert_model.d_model),
            nn.GELU(),
            nn.LayerNorm(bert_model.d_model),
            nn.Linear(bert_model.d_model, vocab_size)
        )

        # NSP head
        self.nsp_head = nn.Sequential(
            nn.Linear(bert_model.d_model, bert_model.d_model),
            nn.Tanh(),
            nn.Linear(bert_model.d_model, 2)  # IsNext or NotNext
        )

    def forward(
        self,
        input_ids,
        segment_ids,
        attention_mask=None,
        mlm_labels=None,
        nsp_labels=None
    ):
        # BERT encoding
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)

        # MLM
        mlm_scores = self.mlm_head(hidden_states)

        # NSP: [CLS] token representation
        cls_representation = hidden_states[:, 0, :]  # (batch, d_model)
        nsp_scores = self.nsp_head(cls_representation)

        # Loss
        total_loss = 0

        if mlm_labels is not None:
            mlm_loss_fct = nn.CrossEntropyLoss()
            mlm_loss = mlm_loss_fct(
                mlm_scores.view(-1, mlm_scores.size(-1)),
                mlm_labels.view(-1)
            )
            total_loss += mlm_loss

        if nsp_labels is not None:
            nsp_loss_fct = nn.CrossEntropyLoss()
            nsp_loss = nsp_loss_fct(nsp_scores, nsp_labels)
            total_loss += nsp_loss

        return total_loss, mlm_scores, nsp_scores
```

### 훈련

```python
# 모델
bert_model = BERTModel(vocab_size=30522)
model = BERTForPreTraining(bert_model, vocab_size=30522)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 훈련 루프
for epoch in range(num_epochs):
    for batch in dataloader:
        # Masking
        masked_input, mlm_labels = mask_tokens(batch['input_ids'], tokenizer)

        # Forward
        loss, mlm_scores, nsp_scores = model(
            input_ids=masked_input,
            segment_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask'],
            mlm_labels=mlm_labels,
            nsp_labels=batch['nsp_labels']
        )

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

---

## 🎯 4. Fine-tuning

### [CLS] Token의 역할

```
[CLS] I love this movie [SEP]
  ↓
BERT Encoder
  ↓
[CLS] representation → Classification head
```

**[CLS] = 전체 시퀀스의 representation!**

### Text Classification

```python
class BERTForSequenceClassification(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_model.d_model, num_labels)
        )

    def forward(self, input_ids, segment_ids, attention_mask=None, labels=None):
        # BERT encoding
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)

        # [CLS] representation
        cls_output = hidden_states[:, 0, :]

        # Classification
        logits = self.classifier(cls_output)

        # Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return loss, logits

# Fine-tuning
bert_model = BERTModel.from_pretrained('bert-base-uncased')
model = BERTForSequenceClassification(bert_model, num_labels=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):  # Fine-tuning: 적은 epoch!
    for batch in train_loader:
        loss, logits = model(
            input_ids=batch['input_ids'],
            segment_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Named Entity Recognition (NER)

```python
class BERTForTokenClassification(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model

        # Token-level classifier
        self.classifier = nn.Linear(bert_model.d_model, num_labels)

    def forward(self, input_ids, segment_ids, attention_mask=None, labels=None):
        # BERT encoding
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)

        # Token classification
        logits = self.classifier(hidden_states)

        # Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        return loss, logits

# 예시: NER
# Labels: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG
# "John lives in New York" → [B-PER, O, O, B-LOC, I-LOC]
```

### Question Answering

```python
class BERTForQuestionAnswering(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model

        # Start/End position prediction
        self.qa_outputs = nn.Linear(bert_model.d_model, 2)

    def forward(
        self,
        input_ids,
        segment_ids,
        attention_mask=None,
        start_positions=None,
        end_positions=None
    ):
        # BERT encoding
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)

        # Predict start/end
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Loss
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss, start_logits, end_logits

# 사용
# Input: [CLS] What is the capital? [SEP] The capital is Paris. [SEP]
# Output: start=7, end=7 (Paris)
```

---

## 🎓 학습 목표

- [ ] BERT 아키텍처 구현 (Embedding + Encoder)
- [ ] MLM 이해 및 구현 (masking 전략)
- [ ] NSP 이해 및 구현
- [ ] Fine-tuning for classification, NER, QA
- [ ] [CLS] token의 역할 이해

---

## 💡 실전 팁

### 1. Pre-training 데이터

```python
# 큰 데이터 필요!
# - Wikipedia
# - BookCorpus
# - Common Crawl

# BERT-base: 16GB text
# BERT-large: 더 큼
```

### 2. Fine-tuning 하이퍼파라미터

```python
# Learning rate: 작게!
lr = 2e-5  # 5e-5, 3e-5도 시도

# Epochs: 적게!
epochs = 3  # 2-4 epoch

# Warmup
warmup_steps = len(train_loader) // 10
```

### 3. HuggingFace 사용

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# 모델 로드
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 훈련
trainer.train()
```

---

## 📊 BERT Variants

### RoBERTa (Facebook)

**개선점**:
1. NSP 제거 (별로 도움 안됨)
2. Dynamic masking (매번 다르게 mask)
3. 더 큰 batch, 더 긴 훈련
4. Byte-level BPE

**결과**: BERT보다 성능 ↑

### ALBERT (Google)

**개선점**:
1. Factorized embedding (메모리 절약)
2. Cross-layer parameter sharing
3. SOP (Sentence Order Prediction) instead of NSP

**결과**: 파라미터 18배 감소, 성능 유지!

### DistilBERT (HuggingFace)

**Knowledge Distillation**:
```python
# Teacher: BERT-base (110M params)
# Student: DistilBERT (66M params, 6 layers)

loss = distillation_loss(student_logits, teacher_logits)
```

**결과**: 40% 작고, 60% 빠르고, 성능 97% 유지!

---

## ⏭️ 다음

👉 [Day 10-11: GPT Implementation](./03-gpt-implementation.md)

**이제 GPT로 text generation을 배워봅시다!** 🚀
