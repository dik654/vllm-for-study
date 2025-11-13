# Day 5-6: Transformer Blocks

## 🎯 목표

**Encoder와 Decoder 블록을 완전히 이해하고 구현**

```python
# Transformer = Stack of blocks
Encoder = [EncoderBlock] * N  # N=6 in original
Decoder = [DecoderBlock] * N

# Each block = Attention + FFN + Normalization + Residual
```

---

## 🏗️ Encoder Block

### 구조

```
Input
  ↓
Layer Norm → Multi-Head Self-Attention → Add (residual)
  ↓
Layer Norm → Feed-Forward Network → Add (residual)
  ↓
Output
```

### 구현

```python
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # 1. Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 2. Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 3. Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        
        # Self-Attention with residual
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual  
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
```

---

## 🔄 Decoder Block

### 구조

```
Input
  ↓
Layer Norm → Masked Self-Attention → Add
  ↓
Layer Norm → Cross-Attention (with Encoder) → Add
  ↓
Layer Norm → Feed-Forward Network → Add
  ↓
Output
```

### 구현

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # 1. Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 2. Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 3. Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 4. Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # x: (batch, tgt_len, d_model)
        # encoder_output: (batch, src_len, d_model)
        
        # 1. Masked Self-Attention
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. Cross-Attention
        cross_out = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_out))
        
        # 3. FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x
```

---

## 🔧 Layer Normalization

### Pre-LN vs Post-LN

```python
# Post-LN (Original Transformer)
def post_ln_block(x):
    # Attention
    attn_out = attention(x)
    x = x + attn_out
    x = layer_norm(x)  # Norm AFTER residual
    
    # FFN
    ffn_out = ffn(x)
    x = x + ffn_out
    x = layer_norm(x)
    return x

# Pre-LN (Modern, more stable)
def pre_ln_block(x):
    # Attention
    normed = layer_norm(x)  # Norm BEFORE attention
    attn_out = attention(normed)
    x = x + attn_out
    
    # FFN
    normed = layer_norm(x)
    ffn_out = ffn(normed)
    x = x + ffn_out
    return x
```

**Pre-LN 장점**:
- 훈련 안정성 ↑
- Gradient flow 개선
- GPT, LLaMA 등에서 사용

---

## 📦 Complete Transformer

```python
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, src, src_mask=None):
        # Embed + Position
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # Embed + Position
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
```

---

## 🎭 Masking

### Padding Mask

```python
def create_padding_mask(seq, pad_idx=0):
    # seq: (batch, seq_len)
    # Returns: (batch, 1, 1, seq_len)
    
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask  # True = attend, False = ignore
```

### Causal Mask (Look-ahead)

```python
def create_causal_mask(size):
    # size: seq_len
    # Returns: (1, 1, size, size)
    
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    mask = ~mask  # Invert: True = attend, False = mask
    return mask.unsqueeze(0).unsqueeze(0)

# Example
causal_mask = create_causal_mask(5)
# [[True,  False, False, False, False],
#  [True,  True,  False, False, False],
#  [True,  True,  True,  False, False],
#  [True,  True,  True,  True,  False],
#  [True,  True,  True,  True,  True]]
```

### Combined Mask

```python
def create_target_mask(tgt, pad_idx=0):
    # Padding mask
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)
    
    # Causal mask
    tgt_len = tgt.size(1)
    tgt_causal_mask = create_causal_mask(tgt_len).to(tgt.device)
    
    # Combine (both must be True to attend)
    tgt_mask = tgt_pad_mask & tgt_causal_mask
    return tgt_mask
```

---

## 🎓 학습 목표

- [ ] Encoder block 구현
- [ ] Decoder block 구현  
- [ ] Pre-LN vs Post-LN 차이 이해
- [ ] Masking 구현 (padding, causal)
- [ ] Complete Transformer 조립

---

## ⏭️ 다음

👉 [Day 7: Training Pipeline](./04-training-pipeline.md)
