# Day 6-7: DPO (Direct Preference Optimization)

## 🎯 목표

**RLHF보다 간단하고 효율적인 Alignment**

```python
# RLHF: 복잡함
Step 1: Train reward model (별도 모델)
Step 2: PPO with that reward model (RL, 불안정)

# DPO: 간단함!
Step 1: 직접 preference에서 학습 (끝!)
```

---

## 📖 핵심 아이디어

### RLHF의 문제점

**1. 복잡성**
```python
# RLHF pipeline
SFT model → Reward model training → PPO fine-tuning
  ↓           ↓                      ↓
3개월       2개월                   4개월 (불안정!)
```

**2. 불안정성**
- PPO는 RL 알고리즘 → 하이퍼파라미터에 민감
- KL penalty 조정 어려움
- Value network 학습 불안정

**3. 비효율성**
- Reward model: 별도 모델 필요 (메모리 2배)
- PPO: 여러 forward/backward pass

### DPO의 해결책

**핵심: Reward model 없이 직접 preference 학습!**

```
RLHF:  Preference → Reward Model → RL (PPO) → Policy

DPO:   Preference → Policy (직접!)
```

**수학적 통찰**

RLHF의 목표:
```
maximize: 𝔼[r(x, y)]  subject to: KL(π || π_ref) < δ
```

이 최적 정책은 **closed-form**으로 표현 가능:
```
π*(y|x) = π_ref(y|x) · exp(r(x,y) / β) / Z(x)
```

역으로 풀면:
```
r(x, y) = β log(π*(y|x) / π_ref(y|x)) + β log Z(x)
```

→ **Reward를 policy의 ratio로 표현!**

---

## 🧮 DPO 수식

### Bradley-Terry Model

Preference 확률:
```
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
```
- y_w: chosen (선호)
- y_l: rejected (비선호)
- σ: sigmoid

### DPO Loss

위 r을 policy ratio로 치환:
```
P(y_w > y_l | x) = σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))
```

**DPO Loss (negative log likelihood)**:
```
ℒ_DPO(θ) = -𝔼[(x,y_w,y_l)] log σ(
    β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x)
)
```

**직관**:
- π_θ(y_w|x) ↑ (chosen 확률 높이기)
- π_θ(y_l|x) ↓ (rejected 확률 낮추기)
- π_ref와 너무 멀어지지 않게 (implicit KL penalty!)

---

## 💻 실습: DPO 구현

### 데이터 준비

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Preference 데이터셋
# Format: {"prompt": "...", "chosen": "...", "rejected": "..."}

dataset = load_dataset("Anthropic/hh-rlhf", split="train")
# 예시:
# {
#   "prompt": "How do I make a cake?",
#   "chosen": "Here's a simple recipe: 1. Preheat oven...",
#   "rejected": "I don't know."
# }

# Reference model (frozen)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

π_ref = AutoModelForCausalLM.from_pretrained(model_name)
π_ref.eval()  # Freeze

# Policy model (trainable)
π_θ = AutoModelForCausalLM.from_pretrained(model_name)
```

### DPO Loss 함수

```python
def compute_dpo_loss(
    policy_model,
    ref_model,
    prompts,
    chosen_responses,
    rejected_responses,
    β=0.1
):
    """
    DPO loss 계산

    Args:
        policy_model: π_θ (trainable)
        ref_model: π_ref (frozen)
        prompts: 입력 프롬프트
        chosen_responses: 선호 응답
        rejected_responses: 비선호 응답
        β: temperature parameter
    """

    # 1. Tokenize
    chosen_inputs = tokenizer(
        [p + c for p, c in zip(prompts, chosen_responses)],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    rejected_inputs = tokenizer(
        [p + r for p, r in zip(prompts, rejected_responses)],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # 2. Get log probabilities
    with torch.no_grad():
        # Reference model (frozen)
        ref_chosen_logp = get_log_prob(ref_model, chosen_inputs)
        ref_rejected_logp = get_log_prob(ref_model, rejected_inputs)

    # Policy model (trainable)
    policy_chosen_logp = get_log_prob(policy_model, chosen_inputs)
    policy_rejected_logp = get_log_prob(policy_model, rejected_inputs)

    # 3. Compute ratios
    chosen_ratio = policy_chosen_logp - ref_chosen_logp
    rejected_ratio = policy_rejected_logp - ref_rejected_logp

    # 4. DPO loss
    logits = β * (chosen_ratio - rejected_ratio)
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    # 5. Metrics
    accuracy = (logits > 0).float().mean()

    return loss, accuracy

def get_log_prob(model, inputs):
    """Get log probability of the sequence"""
    outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits

    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, 1:].contiguous()

    # Compute log prob
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=shift_labels.unsqueeze(2)
    ).squeeze(2)

    # Mask padding
    mask = (shift_labels != tokenizer.pad_token_id).float()
    seq_log_prob = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1)

    return seq_log_prob
```

### 훈련 루프

```python
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Optimizer
optimizer = AdamW(π_θ.parameters(), lr=1e-6)

# DataLoader
def collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    chosen = [item["chosen"] for item in batch]
    rejected = [item["rejected"] for item in batch]
    return prompts, chosen, rejected

train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

# Training loop
π_θ.train()
π_ref.eval()

for epoch in range(3):
    total_loss = 0
    total_acc = 0

    for batch in train_loader:
        prompts, chosen, rejected = batch

        # Forward
        loss, acc = compute_dpo_loss(
            π_θ, π_ref,
            prompts, chosen, rejected,
            β=0.1
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

    print(f"Epoch {epoch+1}")
    print(f"  Loss: {total_loss / len(train_loader):.4f}")
    print(f"  Accuracy: {total_acc / len(train_loader):.4f}")

# Save
π_θ.save_pretrained("./gpt2-dpo")
```

---

## 🔬 DPO vs RLHF 비교

### 코드 복잡도

```python
# RLHF
class RLHFTrainer:
    def __init__(self):
        self.policy_model = ...
        self.value_model = ...      # 추가 모델!
        self.reward_model = ...      # 추가 모델!
        self.reference_model = ...

    def train_step(self):
        # 1. Generate responses
        responses = self.policy_model.generate(...)

        # 2. Get rewards
        rewards = self.reward_model(...)

        # 3. Compute advantages
        values = self.value_model(...)
        advantages = compute_gae(rewards, values)

        # 4. PPO update (clipping, multiple epochs, ...)
        for _ in range(ppo_epochs):
            ratio = ...
            clipped = ...
            loss = ...
        # 복잡!

# DPO
class DPOTrainer:
    def __init__(self):
        self.policy_model = ...
        self.reference_model = ...   # 이것만!

    def train_step(self):
        # 1. Compute log probs
        policy_logp = ...
        ref_logp = ...

        # 2. Loss
        loss = -logsigmoid(β * (policy_logp - ref_logp)).mean()
        # 간단!
```

### 메모리 사용량

| Method | Models in Memory | Relative Memory |
|--------|------------------|-----------------|
| RLHF   | Policy + Value + Reward + Ref | 4x |
| DPO    | Policy + Ref | 2x |

### 안정성

```python
# RLHF: 하이퍼파라미터 많음
ppo_config = {
    "clip_range": 0.2,           # 민감!
    "kl_penalty": 0.1,           # 민감!
    "value_clip_range": 0.2,
    "ppo_epochs": 4,
    "mini_batch_size": 64,
    # ...
}

# DPO: 하이퍼파라미터 적음
dpo_config = {
    "β": 0.1,                    # 이것만!
    "learning_rate": 1e-6
}
```

---

## 📊 실험 결과 (논문)

### Sentiment Task

| Method | Accuracy | Training Time |
|--------|----------|---------------|
| SFT    | 65.2%    | 1x            |
| RLHF   | 78.5%    | 10x           |
| DPO    | 79.1%    | 3x            |

**DPO가 RLHF보다 빠르고 성능도 비슷하거나 좋음!**

### Summarization

| Method | Human Win Rate vs SFT |
|--------|------------------------|
| RLHF   | 61%                    |
| DPO    | 58%                    |

**거의 동등한 성능**

---

## 🎯 β (Beta) 파라미터

### β의 역할

```python
# β가 크면
β = 1.0
→ π_θ가 π_ref에서 많이 벗어날 수 있음
→ 빠른 학습, 하지만 불안정

# β가 작으면
β = 0.01
→ π_θ가 π_ref에 가까이 유지
→ 느린 학습, 하지만 안정적
```

### 최적 β 찾기

```python
# Grid search
for β in [0.01, 0.05, 0.1, 0.5, 1.0]:
    model = train_dpo(dataset, β=β)
    score = evaluate(model)
    print(f"β={β}: {score}")

# 일반적으로 β=0.1이 잘 작동
```

---

## 🔥 고급 기법

### 1. IPO (Identity Preference Optimization)

**DPO 문제**: Sigmoid에서 overfitting 가능

**IPO 해결**:
```python
# DPO loss
loss_dpo = -logsigmoid(β * (chosen_ratio - rejected_ratio))

# IPO loss (squared)
loss_ipo = (chosen_ratio - rejected_ratio - 1/(2*β)) ** 2
```

### 2. RPO (Ranking Preference Optimization)

**여러 개의 응답을 순위로**
```python
# Pairwise (DPO)
responses = [y1, y2]  # chosen vs rejected

# Ranking (RPO)
responses = [y1, y2, y3, y4]  # 순위: 1 > 2 > 3 > 4

# Loss
loss = 0
for i in range(len(responses)):
    for j in range(i+1, len(responses)):
        # i가 j보다 선호
        loss += -logsigmoid(β * (logp[i] - logp[j]))
```

### 3. Iterative DPO

**한 번만 하지 말고 반복!**

```python
# Round 1
π_1 = DPO(π_ref, dataset_1)

# Round 2: π_1로 새 응답 생성
dataset_2 = generate_with_π_1()
π_2 = DPO(π_1, dataset_2)  # π_1이 새 reference

# Round 3
dataset_3 = generate_with_π_2()
π_3 = DPO(π_2, dataset_3)

# 계속 개선!
```

---

## 💻 실습 2: TRL 라이브러리로 DPO

**Hugging Face의 TRL (Transformer Reinforcement Learning)** 라이브러리 사용

```bash
pip install trl
```

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 모델
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Reference model (자동으로 복사됨)
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")

# 데이터
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")

# Config
config = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=1e-6,
    beta=0.1,  # DPO β
    max_length=512,
    max_prompt_length=256,
)

# Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Train!
trainer.train()

# Save
model.save_pretrained("./gpt2-dpo-final")
```

**단 20줄로 DPO 완성!**

---

## 🎓 학습 목표 체크리스트

- [ ] RLHF의 한계 이해 (복잡, 불안정, 비효율)
- [ ] DPO의 핵심 아이디어 이해 (reward → policy ratio)
- [ ] DPO loss 수식 이해 및 구현
- [ ] β 파라미터의 역할 이해
- [ ] DPO를 직접 구현 완료
- [ ] TRL 라이브러리로 DPO 훈련
- [ ] DPO vs RLHF 성능/복잡도 비교

---

## 📊 언제 DPO? 언제 RLHF?

### DPO 선택

✅ **간단한 alignment**
✅ **빠른 iteration 필요**
✅ **안정적인 학습 원함**
✅ **메모리 제약**
✅ **Preference 데이터만 있음**

### RLHF 선택

✅ **복잡한 reward 필요** (예: 코드 실행 결과)
✅ **Online learning** (계속 새 데이터 생성)
✅ **Multi-objective reward**
✅ **최고 성능 필요** (리소스 무한)

---

## 📚 참고 논문

- **Direct Preference Optimization** (Rafailov et al., 2023)
  - [Paper](https://arxiv.org/abs/2305.18290)
  - DPO 원논문

- **IPO** (Azar et al., 2023)
  - [Paper](https://arxiv.org/abs/2310.12036)
  - DPO 개선

- **ORPO** (Hong et al., 2024)
  - SFT + DPO를 동시에

---

## ⏭️ 다음 단계

DPO로 **간단하고 효율적인 alignment**를 배웠습니다!

하지만 "선호도"를 어디서 얻을까요?
- 인간 라벨링 = 비쌈 💰

👉 [Day 11-12: Constitutional AI](./05-constitutional-ai.md)에서 **AI가 스스로 preference를 만드는** 방법을 배웁니다!

**"이제 ChatGPT를 만들 수 있습니다!"** 🎉
