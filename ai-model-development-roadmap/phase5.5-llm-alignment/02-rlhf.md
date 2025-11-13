# Day 3-5: RLHF - ChatGPT의 비밀

## 🎯 학습 목표

**"GPT-3를 ChatGPT로 만드는 마법을 이해합니다."**

RLHF (Reinforcement Learning from Human Feedback)는:
- ✅ ChatGPT, Claude, Gemini의 핵심 기술
- ✅ LLM을 "helpful, harmless, honest"하게 만드는 방법
- ✅ 인간 선호도를 AI에 주입하는 기술

---

## 1. RLHF가 해결하는 문제

### 📖 문제 상황

**Base LLM의 한계:**

```python
# GPT-3 (Base model, RLHF 전)
prompt = "How do I bake a cake?"

bad_outputs = [
    "How do I bake a cake? How do I bake bread? ...",  # 질문만 반복
    "I don't know.",  # 도움 안 됨
    "[Toxic content]",  # 유해한 내용
    "To bake a cake, you need... [2000 words]",  # 너무 김
]

# ChatGPT (RLHF 후)
good_output = """
Here's a simple chocolate cake recipe:

Ingredients:
- 2 cups flour
- 1.5 cups sugar
...

Steps:
1. Preheat oven to 350°F
2. Mix dry ingredients...
"""
# → 간결하고 유용하고 안전함!
```

**핵심 질문**: *어떻게 "좋은 답변"을 학습시키나?*

---

## 2. RLHF 3단계 파이프라인

### 📊 전체 흐름

```
┌─────────────┐
│  Base LLM   │ (GPT-3)
│ (Pre-trained)│
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Step 1: SFT        │ ← 고품질 데이터로 fine-tune
│ (Supervised)        │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Step 2:            │
│  Reward Model       │ ← 인간 선호도 학습
│  Training           │   "A가 B보다 좋다"
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Step 3:            │
│  PPO Optimization   │ ← Reward를 최대화
│  (RL fine-tuning)   │
└─────────────────────┘
       │
       ▼
   ChatGPT! 🎉
```

---

## 3. Step 1: Supervised Fine-Tuning (SFT)

### 📖 개념

Base model을 **고품질 instruction-response 쌍**으로 fine-tune합니다.

### 💻 실습 1: SFT 데이터셋 구조

```python
# sft_dataset_example.py
sft_data = [
    {
        "instruction": "What is the capital of France?",
        "response": "The capital of France is Paris."
    },
    {
        "instruction": "Write a Python function to reverse a string.",
        "response": """
Here's a simple function to reverse a string:

```python
def reverse_string(s):
    return s[::-1]

# Example usage
print(reverse_string("hello"))  # Output: "olleh"
```
"""
    },
    {
        "instruction": "Explain quantum computing in simple terms.",
        "response": "Quantum computing uses quantum mechanics principles..."
    }
]

# 특징:
# - 고품질 (사람이 작성 or 검증)
# - 다양한 태스크
# - 안전하고 유용한 답변
```

### 💻 실습 2: SFT 훈련

```python
# sft_training.py
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 1. 모델과 토크나이저 로딩
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. 데이터셋 준비
def format_instruction(example):
    """Instruction을 프롬프트 포맷으로 변환"""
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    return {"text": prompt}

dataset = load_dataset("your-sft-dataset")
dataset = dataset.map(format_instruction)

# 3. 토크나이제이션
def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# 4. 훈련 설정
training_args = TrainingArguments(
    output_dir="./sft_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    logging_steps=100,
    save_strategy="epoch",
)

# 5. Trainer 생성 및 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

print("SFT 훈련 시작...")
trainer.train()
print("SFT 완료!")

# 6. 저장
model.save_pretrained("./sft_model_final")
tokenizer.save_pretrained("./sft_model_final")
```

---

## 4. Step 2: Reward Model Training

### 📖 개념

**Reward Model (RM)**: 응답의 "품질"을 점수로 매기는 모델

```
입력: (prompt, response)
출력: scalar reward (높을수록 좋음)
```

### 🧮 수학적 정의

Pairwise comparison을 통해 학습:

$$
\text{Loss} = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right]
$$

여기서:
- $x$: prompt
- $y_w$: 선호하는 응답 (winner)
- $y_l$: 선호하지 않는 응답 (loser)
- $r_\theta$: Reward model
- $\sigma$: Sigmoid function

**직관**: "좋은 답변의 reward - 나쁜 답변의 reward"를 최대화

### 💻 실습 3: 인간 선호도 데이터

```python
# preference_data.py
preference_data = [
    {
        "prompt": "Explain machine learning",
        "chosen": "Machine learning is a subset of AI that enables systems to learn from data...",
        "rejected": "I don't know."
    },
    {
        "prompt": "Write a haiku about AI",
        "chosen": "Silicon dreams rise\nAlgorithms learn and grow fast\nFuture unfolds now",
        "rejected": "AI AI AI\nAI AI AI AI AI\nAI AI AI AI"  # 의미 없음
    },
    {
        "prompt": "How do I hack into a system?",
        "chosen": "I can't help with that. Hacking into systems without authorization is illegal...",
        "rejected": "Here's how to hack: ..."  # 유해함
    }
]

# 수집 방법:
# 1. 사람이 직접 비교 (A vs B, 어느 게 더 좋나?)
# 2. 여러 labeler의 평균
# 3. 수만~수십만 개 필요
```

### 💻 실습 4: Reward Model 훈련

```python
# train_reward_model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """
    Reward Model: (prompt, response) → scalar reward
    """
    def __init__(self, base_model_name):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size

        # Reward head: hidden state → scalar
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Base model의 마지막 hidden state
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden = outputs.last_hidden_state

        # [EOS] 토큰의 hidden state 사용
        # (sequence의 마지막 정보를 담고 있음)
        eos_indices = attention_mask.sum(dim=1) - 1
        eos_hidden = last_hidden[torch.arange(last_hidden.size(0)), eos_indices]

        # Reward 계산
        reward = self.reward_head(eos_hidden)
        return reward.squeeze(-1)  # (batch_size,)

# 훈련 함수
def train_reward_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Unpack
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        # Forward pass
        reward_chosen = model(chosen_ids, chosen_mask)
        reward_rejected = model(rejected_ids, rejected_mask)

        # Pairwise ranking loss
        # Loss = -log(sigmoid(r_chosen - r_rejected))
        loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 예제 사용
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 초기화
    reward_model = RewardModel("gpt2").to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)

    # 훈련 (pseudo-code)
    for epoch in range(3):
        loss = train_reward_model(reward_model, train_dataloader, optimizer, device)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # 저장
    torch.save(reward_model.state_dict(), "reward_model.pt")
    print("Reward Model 훈련 완료!")
```

### 💻 실습 5: Reward Model 테스트

```python
# test_reward_model.py
def test_reward_model(reward_model, tokenizer, prompt, responses):
    """
    여러 응답에 대해 reward 계산
    """
    reward_model.eval()

    print(f"Prompt: {prompt}\n")

    for i, response in enumerate(responses):
        # 토크나이제이션
        text = f"{prompt}\n{response}"
        inputs = tokenizer(text, return_tensors="pt")

        # Reward 계산
        with torch.no_grad():
            reward = reward_model(
                inputs["input_ids"],
                inputs["attention_mask"]
            )

        print(f"Response {i+1}: {response}")
        print(f"  Reward: {reward.item():.4f}\n")

# 테스트
prompt = "What is the capital of France?"
responses = [
    "The capital of France is Paris.",  # 정확
    "I don't know.",  # 도움 안 됨
    "Paris is a city.",  # 애매
    "France France France",  # 의미 없음
]

test_reward_model(reward_model, tokenizer, prompt, responses)

# 예상 출력:
# Response 1: Reward: 0.85 (가장 높음)
# Response 2: Reward: 0.23
# Response 3: Reward: 0.41
# Response 4: Reward: -0.12 (가장 낮음)
```

---

## 5. Step 3: PPO (Proximal Policy Optimization)

### 📖 개념

**PPO**: Reward Model의 점수를 최대화하도록 LLM을 fine-tune하는 RL 알고리즘

**RL 용어 매핑:**
- **Policy (π)**: LLM (텍스트 생성 정책)
- **State**: Prompt + 지금까지 생성된 텍스트
- **Action**: 다음 토큰 선택
- **Reward**: Reward Model의 점수

### 🧮 PPO Objective

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

여기서:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ (probability ratio)
- $\hat{A}_t$: Advantage (이 action이 얼마나 좋은가)
- $\epsilon$: Clipping parameter (보통 0.2)

**직관**: 모델을 업데이트하되, 너무 크게 바뀌지 않게 clip!

### 💻 실습 6: PPO 핵심 구현

```python
# ppo_core.py
import torch
import torch.nn.functional as F

def compute_ppo_loss(
    logprobs_new,      # 새 정책의 log prob
    logprobs_old,      # 옛 정책의 log prob
    advantages,        # Advantage 값
    clip_epsilon=0.2
):
    """
    PPO clipped objective 계산

    Args:
        logprobs_new: (batch, seq_len)
        logprobs_old: (batch, seq_len)
        advantages: (batch, seq_len)
    """
    # Probability ratio
    ratio = torch.exp(logprobs_new - logprobs_old)

    # Clipped ratio
    ratio_clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # PPO loss (minimize negative objective)
    loss1 = ratio * advantages
    loss2 = ratio_clipped * advantages
    loss = -torch.min(loss1, loss2).mean()

    return loss

# 예제
if __name__ == "__main__":
    batch_size, seq_len = 4, 10

    # 더미 데이터
    logprobs_new = torch.randn(batch_size, seq_len)
    logprobs_old = torch.randn(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len)

    loss = compute_ppo_loss(logprobs_new, logprobs_old, advantages)
    print(f"PPO Loss: {loss.item():.4f}")
```

### 💻 실습 7: Value Function (Advantage 계산용)

```python
# value_function.py
class ValueHead(nn.Module):
    """
    Value function: state → expected return
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        values = self.value_head(hidden).squeeze(-1)  # (batch, seq_len)
        return values

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """
    GAE (Generalized Advantage Estimation) 계산

    Args:
        rewards: (batch, seq_len)
        values: (batch, seq_len)
        gamma: discount factor
        lam: GAE lambda
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    # Backward pass
    for t in reversed(range(seq_len - 1)):
        delta = rewards[:, t] + gamma * values[:, t+1] - values[:, t]
        advantages[:, t] = last_gae = delta + gamma * lam * last_gae

    return advantages
```

### 💻 실습 8: 전체 RLHF 파이프라인 (TRL 라이브러리)

```python
# rlhf_with_trl.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import respond_to_batch

# 1. 모델 준비
model_name = "gpt2"
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Reward model 로딩
reward_model = RewardModel("gpt2")
reward_model.load_state_dict(torch.load("reward_model.pt"))
reward_model.eval()

# 2. PPO 설정
ppo_config = PPOConfig(
    batch_size=16,
    learning_rate=1e-5,
    ppo_epochs=4,
    mini_batch_size=4,
    clip_range=0.2,
    kl_penalty="kl",  # KL divergence penalty
    target_kl=0.1,    # SFT 모델과 너무 멀어지지 않게
)

# 3. PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

# 4. 훈련 루프
prompts = [
    "What is the capital of France?",
    "Explain quantum computing",
    # ... more prompts
]

for epoch in range(3):
    for prompt in prompts:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate response
        response = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9
        )

        # Compute reward
        response_text = tokenizer.decode(response[0])
        with torch.no_grad():
            reward = reward_model(response, attention_mask=...)

        # PPO update
        stats = ppo_trainer.step([inputs["input_ids"]], [response], [reward])

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Mean reward = {stats['ppo/mean_scores']:.3f}")

print("RLHF 완료!")
model.save_pretrained("./chatgpt_style_model")
```

---

## 6. RLHF의 도전과제

### ⚠️ 문제점들

1. **복잡성**
   - 3단계 파이프라인
   - Reward model + Policy model 동시 관리

2. **불안정성**
   - RL 훈련은 불안정 (divergence 위험)
   - Hyperparameter에 민감

3. **비용**
   - 인간 선호도 데이터 수집 비용
   - 훈련 시간 (RL은 느림)

4. **Reward Hacking**
   - 모델이 reward를 exploit
   - "좋아 보이지만 사실 별로"인 답변 생성

5. **KL Divergence Balancing**
   - 너무 높으면: 변화 없음
   - 너무 낮으면: 원본 모델과 너무 달라짐

---

## ✅ Day 3-5 완료 체크리스트

### 이론 이해
- [ ] RLHF의 3단계 이해
- [ ] Reward Model의 원리 이해
- [ ] PPO의 clipped objective 이해
- [ ] KL penalty의 필요성 이해

### 실습 완료
- [ ] SFT 훈련 실행
- [ ] Reward Model 구현 및 훈련
- [ ] PPO loss 구현
- [ ] TRL로 mini RLHF 실행

### AI 연결
- [ ] ChatGPT vs GPT-3 차이 설명 가능
- [ ] Reward hacking 문제 이해
- [ ] RLHF의 한계점 파악

---

## 🎯 최종 챌린지

**Mini ChatGPT 만들기!**

1. GPT-2로 시작
2. Instruction dataset으로 SFT
3. Preference data로 Reward Model 훈련
4. PPO로 fine-tuning
5. 평가: 이전 vs 이후 비교

성공하면, 당신은 **ChatGPT의 원리를 완전히 마스터**했습니다! 🎉

---

## ⏭️ 다음 단계

👉 [Day 6-7: DPO (Direct Preference Optimization)](./03-dpo.md)

**"RLHF보다 더 간단한 방법이 있습니다!"** 🚀
