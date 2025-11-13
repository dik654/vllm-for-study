# Efficient Fine-tuning: LoRA & QLoRA

## 🎯 목표

**적은 파라미터로 대형 모델을 효율적으로 fine-tuning**

문제: 7B 모델 fine-tuning = 7B 파라미터 업데이트 = 막대한 메모리/시간
해결: **LoRA**로 0.1% 파라미터만 업데이트 → 99.9% 성능 유지!

---

## ⚠️ Full Fine-tuning의 문제

### 메모리 요구사항

```python
# Llama 2 7B full fine-tuning 메모리 계산

Parameters: 7B
  - FP16: 7B × 2 bytes = 14 GB

Gradients: 7B
  - FP16: 7B × 2 bytes = 14 GB

Optimizer states (AdamW):
  - Momentum: 7B × 4 bytes = 28 GB
  - Variance: 7B × 4 bytes = 28 GB

Total: 14 + 14 + 28 + 28 = 84 GB!
```

**문제점**:
- A100 (80GB)도 부족
- 작은 GPU (24GB)로는 불가능
- 비용이 너무 많이 듦

### 시간 문제

```python
# Fine-tuning 시간 비교

Full fine-tuning (7B params):
  - 1 epoch = 10 hours (A100)
  - 3 epochs = 30 hours
  - Cost: ~$100

LoRA (21M params, 0.3% of 7B):
  - 1 epoch = 2 hours
  - 3 epochs = 6 hours
  - Cost: ~$20

→ 5배 빠름, 5배 저렴!
```

---

## 💡 LoRA: Low-Rank Adaptation

### 핵심 아이디어

**"Fine-tuning 중 weight 변화는 low-rank"**

```python
# Full fine-tuning
W_new = W_pretrained + ΔW
# ΔW는 (d, d) 행렬, full rank

# LoRA
W_new = W_pretrained + B·A
# B: (d, r), A: (r, d)
# r << d (예: r=8, d=4096)
# 파라미터: 2dr (vs d²)
```

**수학적 근거**:
- Pre-trained 모델의 weight는 이미 좋은 위치
- Fine-tuning은 작은 adjustment만 필요
- 이 adjustment는 low-dimensional space에 있음

### 구현

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    LoRA를 적용한 Linear layer

    W_new = W + BA
    where:
      W: frozen pretrained weight (d_out, d_in)
      B: trainable matrix (d_out, r)
      A: trainable matrix (r, d_in)
      r: LoRA rank (보통 4, 8, 16)

    핵심 아이디어: rank-decomposition으로 파라미터 대폭 감소
    """
    def __init__(
        self,
        in_features,
        out_features,
        r=8,              # LoRA rank
        lora_alpha=16,    # LoRA scaling factor
        lora_dropout=0.1
    ):
        super().__init__()

        # Original linear layer (frozen)
        # 의도: pretrained weight는 그대로 유지
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False  # Freeze!

        # LoRA matrices
        # 의도: 작은 rank로 weight adjustment 표현
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Scaling factor
        # 의도: rank에 따른 초기화 scale 조정
        self.scaling = lora_alpha / r

        # Dropout (optional)
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialize LoRA weights
        # A: Gaussian, B: zero
        # 의도: 처음에는 ΔW = BA = 0 (pretrained model과 동일하게 시작)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """
        Forward pass with LoRA

        y = (W + BA) · x
          = W·x + BA·x
          = linear(x) + lora_output

        의도: frozen model + learnable adaptation
        """
        # Original output (frozen)
        original_output = self.linear(x)

        # LoRA output
        # x: (batch, seq_len, in_features)
        # A: (r, in_features)
        # B: (out_features, r)

        # x @ A^T: (batch, seq_len, in_features) @ (in_features, r)
        #        → (batch, seq_len, r)
        lora_output = x @ self.lora_A.T

        # dropout
        lora_output = self.dropout(lora_output)

        # @ B^T: (batch, seq_len, r) @ (r, out_features)
        #      → (batch, seq_len, out_features)
        lora_output = lora_output @ self.lora_B.T

        # Scale and combine
        return original_output + lora_output * self.scaling


def apply_lora_to_model(model, target_modules=["q_proj", "v_proj"], r=8, lora_alpha=16):
    """
    Pretrained 모델에 LoRA 적용

    Args:
        model: Pretrained model (e.g., Llama)
        target_modules: LoRA를 적용할 module 이름 (일반적으로 attention의 Q, V)
        r: LoRA rank
        lora_alpha: Scaling factor

    의도: 전체 model이 아닌 특정 layer만 LoRA 적용 (메모리 최소화)
    """
    lora_params = 0
    total_params = 0

    for name, module in model.named_modules():
        # Linear layer이고, target module에 포함된 경우
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            # 기존 Linear를 LoRALinear로 교체
            in_features = module.in_features
            out_features = module.out_features

            lora_layer = LoRALinear(
                in_features,
                out_features,
                r=r,
                lora_alpha=lora_alpha
            )

            # Pretrained weight 복사
            lora_layer.linear.weight.data = module.weight.data.clone()

            # Module 교체
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            parent_module = model
            for part in parent_name.split('.'):
                if part:
                    parent_module = getattr(parent_module, part)

            setattr(parent_module, child_name, lora_layer)

            # 파라미터 계산
            lora_params += r * (in_features + out_features)
            total_params += in_features * out_features

    print(f"LoRA parameters: {lora_params:,} ({lora_params/total_params*100:.2f}%)")

    return model


# 사용 예제
def lora_finetuning_example():
    """
    LoRA fine-tuning 예제
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. Pretrained 모델 로드
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # 2. LoRA 적용
    # 의도: attention의 Q, V projection만 학습
    # (K는 변경하지 않아도 성능 유지)
    model = apply_lora_to_model(
        model,
        target_modules=["q_proj", "v_proj"],
        r=8,              # 작은 rank (메모리 절약)
        lora_alpha=16     # scaling factor
    )

    # 3. 학습 가능한 파라미터만 optimizer에 추가
    # 의도: frozen parameters는 메모리에서 제외
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # 4. Fine-tuning 루프
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 5. LoRA weights 저장
    # 의도: base model은 그대로, LoRA weights만 저장 (몇 MB)
    lora_state_dict = {
        name: param for name, param in model.named_parameters()
        if 'lora' in name
    }
    torch.save(lora_state_dict, 'lora_weights.pth')
```

---

## 🎯 LoRA 설정 가이드

### Rank 선택

```python
# Rank (r)에 따른 trade-off

r=4:
  - 파라미터: 최소
  - 성능: ~95% of full fine-tuning
  - 사용: 간단한 task, 메모리 제약 큼

r=8:  ← 일반적으로 가장 많이 사용
  - 파라미터: 균형
  - 성능: ~98% of full fine-tuning
  - 사용: 대부분의 task

r=16:
  - 파라미터: 중간
  - 성능: ~99% of full fine-tuning
  - 사용: 복잡한 task, 충분한 메모리

r=64:
  - 파라미터: 많음
  - 성능: ~99.5% of full fine-tuning
  - 사용: 거의 full fine-tuning에 가까움
```

### Target Modules 선택

```python
# 어떤 layer에 LoRA를 적용할까?

옵션 1: Q, V만 (가장 일반적)
  target_modules = ["q_proj", "v_proj"]
  - 파라미터: 최소
  - 성능: 충분히 좋음
  - 이유: K는 content matching, Q/V가 representation 학습

옵션 2: Q, K, V 모두
  target_modules = ["q_proj", "k_proj", "v_proj"]
  - 파라미터: 1.5배
  - 성능: 약간 더 좋음

옵션 3: Q, K, V, O (output projection)
  target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
  - 파라미터: 2배
  - 성능: 최고

옵션 4: Attention + FFN
  target_modules = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
  - 파라미터: 많음
  - 성능: full fine-tuning에 근접
```

### Alpha 설정

```python
# LoRA alpha: scaling factor

일반적 규칙:
  lora_alpha = 2 × r

예:
  r=4  → alpha=8
  r=8  → alpha=16
  r=16 → alpha=32

의도:
  - Alpha가 크면: LoRA의 영향력 증가
  - Alpha가 작으면: Pretrained model에 가까움
  - 2×r이 경험적으로 가장 좋은 균형
```

---

## 🚀 QLoRA: Quantized LoRA

### 문제

LoRA로도 여전히 base model은 메모리에 full precision으로 로드:
- Llama 2 7B: 14GB (FP16)
- A100 (80GB)는 괜찮지만, RTX 4090 (24GB)는 불가능

### 해결: QLoRA

**Base model을 4-bit quantization + LoRA**

```python
# QLoRA 메모리 계산

Llama 2 7B:
  - FP16: 14 GB
  - INT4: 3.5 GB  ← 4배 감소!

LoRA parameters:
  - FP16: ~40 MB (r=8)

Total: 3.5 GB + 0.04 GB = 3.54 GB

→ RTX 3090 (24GB)로도 7B 모델 fine-tuning 가능!
```

### 구현 (bitsandbytes 사용)

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def create_qlora_model(model_name, r=8):
    """
    QLoRA 모델 생성

    핵심: 4-bit quantization + LoRA
    의도: 작은 GPU로도 대형 모델 fine-tuning
    """

    # 1. 4-bit quantization 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 4-bit quantization
        bnb_4bit_quant_type="nf4",            # NormalFloat4 (최적)
        bnb_4bit_compute_dtype=torch.float16, # Computation은 FP16
        bnb_4bit_use_double_quant=True       # Double quantization (더 압축)
    )

    # 2. Quantized model 로드
    # 의도: base model은 INT4, 메모리 4배 절감
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"  # 자동으로 GPU 할당
    )

    # 3. LoRA 설정
    lora_config = LoraConfig(
        r=r,                          # LoRA rank
        lora_alpha=16,                # Scaling
        target_modules=[              # Q, V만 학습
            "q_proj",
            "v_proj"
        ],
        lora_dropout=0.05,            # Dropout
        bias="none",                  # Bias 학습 안함
        task_type="CAUSAL_LM"         # Causal LM task
    )

    # 4. LoRA 적용
    # 의도: quantized model + learnable LoRA adapters
    model = get_peft_model(model, lora_config)

    # 학습 가능한 파라미터 확인
    model.print_trainable_parameters()
    # Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062

    return model


# 사용 예제
model = create_qlora_model("meta-llama/Llama-2-7b-hf", r=8)

# 훈련 (일반 PyTorch와 동일)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 📊 성능 비교

### 메모리 사용량

```
Llama 2 7B Fine-tuning on RTX 3090 (24GB):

Method              | Memory | Possible?
--------------------|--------|----------
Full Fine-tuning    | 84 GB  | ❌ No
LoRA (r=8)          | 18 GB  | ✅ Yes
QLoRA (r=8)         | 6 GB   | ✅ Yes (4개 동시 가능!)


Llama 2 70B Fine-tuning on A100 (80GB):

Method              | Memory  | Possible?
--------------------|---------|----------
Full Fine-tuning    | 560 GB  | ❌ No (8x A100 필요)
LoRA (r=8)          | 140 GB  | ❌ No (2x A100 필요)
QLoRA (r=8)         | 48 GB   | ✅ Yes (1x A100)
```

### 성능 유지율

```
Task: Instruction following (Alpaca dataset)

Method              | Win Rate vs Full FT | Params Trained
--------------------|---------------------|---------------
Full Fine-tuning    | 100%               | 100%
LoRA (r=64)         | 99.3%              | 0.8%
LoRA (r=8)          | 97.8%              | 0.1%
QLoRA (r=64)        | 99.0%              | 0.8%
QLoRA (r=8)         | 96.9%              | 0.1%

결론: 0.1% 파라미터로 97-98% 성능 달성!
```

---

## 💻 실전 예제: Llama 2 Fine-tuning with QLoRA

```python
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. 데이터셋 로드
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 3. QLoRA 모델 생성
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Gradient checkpointing (메모리 절약)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# 4. LoRA 설정
lora_config = LoraConfig(
    r=16,                # Rank
    lora_alpha=32,       # Alpha
    target_modules=[     # Target layers
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. 데이터 전처리
def format_instruction(example):
    """Alpaca format"""
    if example["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example["instruction"]}

### Response:
{example["output"]}"""

def tokenize_function(examples):
    texts = [format_instruction(ex) for ex in examples]
    return tokenizer(texts, truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./llama2-7b-qlora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_steps=100,
    logging_steps=10,
    optim="paged_adamw_8bit",  # QLoRA optimizer
    warmup_ratio=0.03,
    lr_scheduler_type="cosine"
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# 8. 훈련 시작!
trainer.train()

# 9. LoRA weights 저장
model.save_pretrained("./llama2-7b-qlora-final")

print("QLoRA fine-tuning 완료!")
```

---

## 🎯 LoRA vs Full Fine-tuning 비교

### 언제 LoRA를 사용할까?

**LoRA가 좋은 경우**:
- ✅ 메모리/GPU 제한이 있을 때
- ✅ 여러 task에 대해 여러 adapter 필요
- ✅ 빠른 iteration 필요
- ✅ 소규모 데이터셋 (overfitting 방지)

**Full Fine-tuning이 좋은 경우**:
- ✅ 충분한 GPU 메모리 (80GB+)
- ✅ Task가 pre-training과 매우 다름
- ✅ 최고 성능이 절대적으로 필요
- ✅ 대규모 데이터셋

### Task별 권장사항

```
Instruction Following (Alpaca, Dolly):
  → LoRA r=8-16 충분

Conversational AI (ShareGPT):
  → LoRA r=16-32

Domain Adaptation (Medical, Legal):
  → LoRA r=32-64 or Full FT

Code Generation:
  → LoRA r=16 충분

Summarization, Translation:
  → LoRA r=8 충분
```

---

## 📝 Summary

### Key Takeaways

1. **LoRA = Low-Rank Adaptation**
   - Weight update를 BA로 decompose
   - 0.1% 파라미터로 97-98% 성능

2. **QLoRA = Quantized LoRA**
   - Base model 4-bit quantization
   - 메모리 4배 절감
   - RTX 3090으로 7B 모델 fine-tuning 가능

3. **Hyperparameters**:
   - Rank: 일반적으로 8-16
   - Alpha: 2 × rank
   - Target modules: Q, V 또는 Q, K, V, O

4. **장점**:
   - 메모리 효율
   - 빠른 훈련
   - Multiple adapters 관리 용이
   - Overfitting 방지

5. **실전 팁**:
   - 시작은 r=8, alpha=16
   - 성능 부족하면 r 증가
   - QLoRA로 작은 GPU 활용

---

## 📚 References

**Papers**:
1. **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
2. **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023)

**Code**:
- HuggingFace PEFT: `peft.LoraConfig`
- bitsandbytes: Quantization library

---

## ⏭️ Next Steps

LoRA를 마스터했으니:

1. **Prefix Tuning** → 다른 PEFT 방법
2. **Adapter Layers** → 구조적 adaptation
3. **Multi-task LoRA** → 여러 task 동시 학습

👉 Continue to **프로덕션 배포**

**Efficient Fine-tuning 마스터 완료!** 🚀
