# Week 11: Efficient Fine-tuning Methods

## 🎯 목표

**소형 GPU로 대규모 모델을 효율적으로 fine-tuning하기!**

```python
# 문제:
# - LLaMA-7B full fine-tuning: 140GB VRAM 필요
# - 대부분은 24GB GPU만 보유

# 해결: PEFT (Parameter-Efficient Fine-Tuning)
# - LoRA: 0.01% 파라미터만 훈련
# - QLoRA: 4-bit + LoRA
# - 결과: 7B 모델을 단일 GPU에서!
```

---

## 📖 Part 1: LoRA (Low-Rank Adaptation)

### 1.1 핵심 아이디어

**거대 모델 전체를 업데이트하지 말고, 작은 matrices만 훈련!**

```python
# Full Fine-tuning:
W_new = W_pretrained + ΔW  # ΔW는 거대!

# LoRA:
W_new = W_pretrained + B @ A

# W: (d, k)
# B: (d, r), A: (r, k)
# r << d, k  (예: r=8, d=4096)

# 훈련 파라미터: d×k → r×(d+k)  (99.9% 감소!)
```

### 1.2 수학적 근거

**Hypothesis**: Fine-tuning의 변화량 ΔW는 본질적으로 low-rank

```python
# Intrinsic dimensionality of task adaptation is low
# → SVD로 확인 가능:
ΔW = UΣV^T
# → 상위 몇 개 singular values가 대부분 설명

# LoRA는 이를 명시적으로 모델링!
ΔW ≈ B @ A  (rank-r approximation)
```

### 1.3 구현

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    LoRA adaptation layer

    y = Wx + (BA)x * (α/r)
      = Wx + BAx * scaling
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        super().__init__()

        # Frozen pretrained weight
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Scaling
        self.scaling = alpha / rank
        self.rank = rank

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original path (frozen)
        result = self.linear(x)

        # LoRA path
        lora_result = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        result = result + lora_result

        return result

# Replace linear layer
linear = nn.Linear(4096, 4096)
lora_linear = LoRALayer(4096, 4096, rank=8)

# Copy pretrained weights
lora_linear.linear.weight.data = linear.weight.data.clone()

# Parameter count
original_params = sum(p.numel() for p in linear.parameters())
lora_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)

print(f"Original: {original_params:,} parameters")
print(f"LoRA trainable: {lora_params:,} parameters")
print(f"Reduction: {original_params / lora_params:.1f}x")
```

### 1.4 HuggingFace PEFT

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')

# LoRA config
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,  # Scaling factor
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],  # Which layers
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%

# Train normally
# ...

# Save only LoRA weights (tiny!)
model.save_pretrained('./lora_weights')  # ~16MB instead of 13GB!

# Load
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
model = PeftModel.from_pretrained(base_model, './lora_weights')
```

---

## 🔧 Part 2: QLoRA (Quantized LoRA)

### 2.1 핵심 아이디어

**4-bit base model + LoRA = 단일 GPU로 65B 모델 훈련!**

```python
# QLoRA = 3가지 혁신:

# 1. 4-bit NormalFloat (NF4)
#    - 정규분포에 최적화된 quantization

# 2. Double Quantization
#    - Quantization constants도 quantize!

# 3. Paged Optimizers
#    - Optimizer states를 CPU로 offload
```

### 2.2 NF4 (4-bit NormalFloat)

**Neural network weights는 정규분포 → 최적 quantization bins**

```python
# FP16: 2^16 values (-65504 ~ 65504)
# INT4: 2^4 = 16 values

# Uniform quantization (bad):
# bins = [-8, -7, -6, ..., 7]

# NF4 (good):
# bins optimized for N(0,1) distribution
# More bins near 0, fewer bins at extremes

NF4_BINS = [
    -1.0, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911, 0.0,
    0.0911, 0.1848, 0.2844, 0.3949,
    0.5251, 0.6962, 1.0
]
```

### 2.3 구현

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',  # NormalFloat4
    bnb_4bit_use_double_quant=True,  # Double quantization
    bnb_4bit_compute_dtype=torch.bfloat16  # Computation dtype
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    quantization_config=bnb_config,
    device_map='auto'
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Add LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

model = get_peft_model(model, lora_config)

# Memory usage
import GPUtil
gpu = GPUtil.getGPUs()[0]
print(f"GPU Memory: {gpu.memoryUsed / 1024:.2f} GB")
# ~5GB for 7B model! (vs 14GB for fp16)
```

---

## 🎯 Part 3: Other PEFT Methods

### 3.1 Prefix Tuning

**Idea**: Prepend learnable "virtual tokens" to input

```python
# Input: [x1, x2, x3, ...]
# With prefix: [p1, p2, ..., pk, x1, x2, x3, ...]
#              └─ learnable ─┘

class PrefixTuning(nn.Module):
    def __init__(self, num_virtual_tokens, hidden_size):
        super().__init__()

        # Learnable prefix embeddings
        self.prefix = nn.Parameter(torch.randn(num_virtual_tokens, hidden_size))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)

        # Prepend prefix
        batch_size = x.size(0)
        prefix = self.prefix.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate
        x_with_prefix = torch.cat([prefix, x], dim=1)

        return x_with_prefix

# Usage
prefix_tuning = PrefixTuning(num_virtual_tokens=10, hidden_size=768)

# Parameters
print(f"Trainable: {10 * 768 = 7,680} parameters")
# vs Full fine-tuning: millions!
```

### 3.2 Adapter Layers

**Idea**: Insert small bottleneck modules

```python
class AdapterLayer(nn.Module):
    """
    Adapter with bottleneck architecture
    """

    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()

        # Down projection
        self.down_proj = nn.Linear(hidden_size, adapter_size)

        # Up projection
        self.up_proj = nn.Linear(adapter_size, hidden_size)

        # Activation
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)

        # Residual
        residual = x

        # Adapter
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)

        # Add residual
        return x + residual

# Insert after each transformer layer
# Only train adapter weights!
```

### 3.3 IA³ (Infused Adapter by Inhibiting and Amplifying)

**Idea**: Learnable scaling vectors

```python
class IA3Layer(nn.Module):
    """
    IA³: Element-wise scaling
    """

    def __init__(self, hidden_size):
        super().__init__()

        # Learnable scaling vectors (initialized to 1)
        self.scale_k = nn.Parameter(torch.ones(hidden_size))
        self.scale_v = nn.Parameter(torch.ones(hidden_size))
        self.scale_ff = nn.Parameter(torch.ones(hidden_size))

    def forward_attention(self, k, v):
        # Scale keys and values
        k = k * self.scale_k
        v = v * self.scale_v
        return k, v

    def forward_feedforward(self, x):
        # Scale feedforward
        return x * self.scale_ff

# Minimal parameters: 3 × hidden_size
# For 768-dim: only 2,304 parameters!
```

---

## 📊 Part 4: PEFT Comparison

### 4.1 Parameter Efficiency

| Method | Parameters | Memory (7B) | Training Speed |
|--------|-----------|-------------|----------------|
| Full Fine-tuning | 100% | 140GB | 1x |
| LoRA (r=8) | 0.01% | 24GB | 1.2x |
| QLoRA (4-bit) | 0.01% | 9GB | 0.9x |
| Prefix Tuning | 0.001% | 24GB | 1.1x |
| Adapter | 0.1% | 28GB | 1.0x |
| IA³ | 0.0001% | 24GB | 1.3x |

### 4.2 Performance Comparison

```python
import matplotlib.pyplot as plt

methods = ['Full FT', 'LoRA', 'QLoRA', 'Prefix', 'Adapter', 'IA³']
accuracy = [0.92, 0.91, 0.90, 0.88, 0.89, 0.87]  # Example scores
trainable_params = [100, 0.01, 0.01, 0.001, 0.1, 0.0001]  # Percentage

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
axes[0].bar(methods, accuracy)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Performance Comparison')
axes[0].set_ylim(0.8, 1.0)

# Parameters
axes[1].bar(methods, trainable_params)
axes[1].set_ylabel('Trainable Parameters (%)')
axes[1].set_title('Parameter Efficiency')
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('peft_comparison.png')
```

---

## 🔬 Part 5: Practical Tips

### 5.1 Rank Selection (LoRA)

```python
# Lower rank (r=4):
# + Less memory
# + Faster training
# - Lower capacity

# Higher rank (r=64):
# + More capacity
# + Better performance
# - More memory

# Recommendation:
# - Start with r=8 or r=16
# - Increase if underfitting
# - Task complexity matters:
#   * Simple tasks: r=4
#   * Complex tasks: r=32+
```

### 5.2 Target Modules

```python
# Which layers to apply LoRA?

# Option 1: Attention only (most common)
target_modules = ['q_proj', 'v_proj']

# Option 2: All attention
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

# Option 3: Attention + FFN
target_modules = ['q_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj']

# Trade-off:
# More modules = Better performance, More memory
```

### 5.3 Merging LoRA Weights

```python
# After training, merge LoRA into base model
def merge_lora_weights(model):
    """
    W_merged = W_base + BA
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            # Compute LoRA update
            lora_update = module.lora_B @ module.lora_A * module.scaling

            # Merge into base weight
            module.linear.weight.data += lora_update

            # Remove LoRA parameters
            del module.lora_A
            del module.lora_B

    return model

# Now inference is as fast as original model!
```

---

## 🎓 Complete Training Example

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# 1. Load model (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    quantization_config=bnb_config,
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare for training
model = prepare_model_for_kbit_training(model)

# 3. LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Load dataset
dataset = load_dataset('timdettmers/openassistant-guanaco')

# 5. Training arguments
training_args = TrainingArguments(
    output_dir='./qlora_llama2',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_strategy='epoch',
    optim='paged_adamw_8bit',  # QLoRA optimizer
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    group_by_length=True
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    tokenizer=tokenizer
)

# 7. Train
trainer.train()

# 8. Save LoRA weights
model.save_pretrained('./qlora_weights')

# 9. Inference
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
model = PeftModel.from_pretrained(base_model, './qlora_weights')

prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## 🎓 학습 목표

- [ ] LoRA 수학적 원리 이해
- [ ] LoRA 직접 구현
- [ ] QLoRA로 대규모 모델 훈련
- [ ] PEFT 방법론 비교
- [ ] Rank selection 전략 이해
- [ ] Production 파이프라인 구축

---

## 💡 Key Takeaways

```python
# PEFT의 핵심:
1. 99%+ 파라미터는 freeze
2. 작은 adaptation module만 훈련
3. 결과: 메모리 10x 감소, 성능 유지!

# 언제 사용?
- GPU 메모리 제한
- 여러 task에 adapt (multi-task)
- 빠른 실험 iteration
- 모델 sharing (base + adapters)
```

---

## ⏭️ 다음

👉 [Week 12: Quantization & Deployment](./02-quantization-deployment.md)

**이제 모델을 더 압축하고 배포하는 방법을 배웁니다!** 🚀
