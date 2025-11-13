# Day 7: Training Pipeline

## 🎯 목표

**Transformer를 실제로 훈련시키는 전체 파이프라인 구현**

---

## 📚 1. Dataset & DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src = self.src_vocab.encode(self.src_sentences[idx])
        tgt = self.tgt_vocab.encode(self.tgt_sentences[idx])
        
        return {
            'src': torch.tensor(src[:self.max_len]),
            'tgt': torch.tensor(tgt[:self.max_len])
        }

def collate_fn(batch, pad_idx=0):
    # Pad sequences
    src = nn.utils.rnn.pad_sequence(
        [item['src'] for item in batch],
        batch_first=True,
        padding_value=pad_idx
    )
    tgt = nn.utils.rnn.pad_sequence(
        [item['tgt'] for item in batch],
        batch_first=True,
        padding_value=pad_idx
    )
    return src, tgt

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)
```

---

## 🎯 2. Training Loop

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        
        # Teacher forcing: use tgt[:-1] as input
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Create masks
        src_mask = create_padding_mask(src)
        tgt_mask = create_target_mask(tgt_input)
        
        # Forward
        optimizer.zero_grad()
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        # Loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

---

## 📈 3. Learning Rate Scheduling

```python
class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        # Formula from paper
        arg1 = self.step_num ** (-0.5)
        arg2 = self.step_num * (self.warmup_steps ** (-1.5))
        return (self.d_model ** (-0.5)) * min(arg1, arg2)

# Usage
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = TransformerLRScheduler(optimizer, d_model=512, warmup_steps=4000)

for epoch in range(num_epochs):
    for batch in dataloader:
        # ... training ...
        scheduler.step()
```

---

## 🔍 4. Inference (Greedy Decoding)

```python
@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, start_token, end_token):
    model.eval()
    
    # Encode source
    encoder_output = model.encode(src, src_mask)
    
    # Start with start token
    tgt = torch.tensor([[start_token]], device=src.device)
    
    for _ in range(max_len):
        # Decode
        tgt_mask = create_causal_mask(tgt.size(1)).to(src.device)
        output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Get next token
        logits = model.output_projection(output[:, -1, :])
        next_token = logits.argmax(dim=-1, keepdim=True)
        
        # Append
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Stop if end token
        if next_token.item() == end_token:
            break
    
    return tgt
```

---

## 🎯 5. Beam Search

```python
def beam_search(model, src, src_mask, beam_size=5, max_len=100):
    # ... more sophisticated decoding ...
    pass
```

---

## 🎓 Complete Training Script

```python
# Setup
model = Transformer(...).to(device)
optimizer = Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
scheduler = TransformerLRScheduler(optimizer, d_model=512)

# Training
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = eval_epoch(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Save checkpoint
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'best_model.pt')
```

---

## 🎓 학습 목표

- [ ] Dataset & DataLoader 구현
- [ ] Teacher forcing 이해
- [ ] LR scheduling 구현
- [ ] Greedy decoding 구현
- [ ] Complete training pipeline

---

**Phase 1 완료! 🎉**
