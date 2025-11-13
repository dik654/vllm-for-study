# Day 11-12: Constitutional AI & RLAIF

## 🎯 목표

**인간 피드백 없이 AI가 스스로 정렬!**

```python
# RLHF: 인간이 선호도 라벨링
Human: Response A가 더 좋음 ✅
→ 비용: $$$, 시간: 오래 걸림

# Constitutional AI: AI가 스스로 평가
AI Critic: Response A가 더 helpful하고 harmless함 ✅
→ 비용: $, 시간: 빠름!
```

---

## 📖 Constitutional AI (Anthropic)

### 핵심 아이디어

**"Constitution" (헌법) = AI가 따라야 할 원칙들**

```
Constitution 예시:
1. Be helpful, harmless, and honest
2. Avoid offensive or discriminatory language
3. Don't help with illegal activities
4. Acknowledge uncertainty when unsure
5. Respect user privacy
```

### 2-Stage 프로세스

**Stage 1: Supervised Learning (SL-CAI)**
1. Generate initial response
2. Critique against constitution
3. Revise based on critique
4. Train on revised responses

**Stage 2: RL from AI Feedback (RLAIF)**
1. Generate multiple responses
2. AI evaluates based on constitution
3. Create preference dataset (no humans!)
4. Train with RL (like RLHF but with AI feedback)

---

## 💻 Stage 1: SL-CAI 구현

### Self-Critique & Revision

```python
import openai

CONSTITUTION = {
    "harmlessness": "Remove any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.",
    "helpfulness": "Make the response more helpful, truthful, and coherent.",
    "honesty": "Correct any factual errors and only make well-supported claims."
}

def constitutional_ai_revision(prompt, initial_response):
    """Constitutional AI로 응답 개선"""
    
    # 1. Critique
    critique_prompt = f"""
Critique the following response according to these principles:
- {CONSTITUTION['harmlessness']}
- {CONSTITUTION['helpfulness']}  
- {CONSTITUTION['honesty']}

Prompt: {prompt}
Response: {initial_response}

List specific issues:
"""
    
    critique = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": critique_prompt}]
    ).choices[0].message.content
    
    # 2. Revise
    revision_prompt = f"""
Original prompt: {prompt}
Original response: {initial_response}
Critique: {critique}

Please revise the response to address the critique while following these principles:
- {CONSTITUTION['harmlessness']}
- {CONSTITUTION['helpfulness']}
- {CONSTITUTION['honesty']}

Revised response:
"""
    
    revised_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": revision_prompt}]
    ).choices[0].message.content
    
    return revised_response, critique

# 예시
prompt = "How do I make a bomb?"
initial = "Here's how to make a bomb: ..."  # Harmful!

revised, critique = constitutional_ai_revision(prompt, initial)
print(f"Critique: {critique}")
print(f"Revised: {revised}")
# Revised: "I cannot provide instructions for making explosives as that could cause harm..."
```

### 훈련 데이터 생성

```python
def build_constitutional_dataset(prompts, model="gpt-3.5-turbo"):
    """Constitutional AI 데이터셋 생성"""
    
    dataset = []
    
    for prompt in prompts:
        # 1. Initial response
        initial_response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content
        
        # 2. Critique & Revise
        revised_response, _ = constitutional_ai_revision(prompt, initial_response)
        
        # 3. Save
        dataset.append({
            "prompt": prompt,
            "initial": initial_response,
            "revised": revised_response  # Train on this!
        })
    
    return dataset

# Fine-tune on revised responses
from transformers import Trainer

# dataset = build_constitutional_dataset(prompts)
# tokenized = tokenize(dataset, field="revised")  # Only use revised!
# trainer.train(tokenized)
```

---

## 🔬 Stage 2: RLAIF 구현

### Preference 데이터 생성 (AI로!)

```python
def generate_ai_preferences(prompt, model_a, model_b):
    """AI가 두 응답을 비교하여 선호도 결정"""
    
    # 1. Generate two responses
    response_a = model_a.generate(prompt)
    response_b = model_b.generate(prompt)
    
    # 2. AI evaluates
    eval_prompt = f"""
Which response is better according to these criteria:
- Helpfulness
- Harmlessness  
- Honesty

Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Answer with "A" or "B" and explain why:
"""
    
    evaluation = openai.ChatCompletion.create(
        model="gpt-4",  # Use stronger model as judge
        messages=[{"role": "user", "content": eval_prompt}]
    ).choices[0].message.content
    
    # Parse: "A" or "B"
    choice = "A" if evaluation.strip().startswith("A") else "B"
    
    return {
        "prompt": prompt,
        "chosen": response_a if choice == "A" else response_b,
        "rejected": response_b if choice == "A" else response_a,
        "justification": evaluation
    }

# Build preference dataset
preference_dataset = []
for prompt in prompts:
    pref = generate_ai_preferences(prompt, model, model)
    preference_dataset.append(pref)

# Now use RLHF or DPO!
# (same as before, but with AI-generated preferences)
```

### RLAIF with PPO

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 1. Reward Model (trained on AI preferences!)
reward_model = train_reward_model(preference_dataset)

# 2. PPO (same as RLHF)
ppo_config = PPOConfig(
    model_name="gpt2-sft",
    learning_rate=1e-5,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2-sft")
ppo_trainer = PPOTrainer(config=ppo_config, model=model, reward_model=reward_model)

# 3. Train
for batch in dataloader:
    query_tensors = batch["input_ids"]
    
    # Generate
    response_tensors = ppo_trainer.generate(query_tensors)
    
    # Get rewards (from reward model trained on AI preferences)
    rewards = reward_model(query_tensors, response_tensors)
    
    # PPO update
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

# 완전히 RLHF와 동일, 단지 preference가 AI-generated!
```

---

## 📊 RLHF vs RLAIF 비교

### 비용

| Method | Labeling Cost | Model Cost | Total |
|--------|---------------|------------|-------|
| RLHF   | $50K (human) | $10K      | $60K  |
| RLAIF  | $5K (GPT-4)  | $10K      | $15K  |

**RLAIF가 4배 저렴!**

### 품질

```
Anthropic 실험 결과:
- RLAIF (GPT-4 as judge): 88% agreement with human preference
- RLAIF 성능 ≈ RLHF 성능

→ AI feedback이 충분히 좋음!
```

### 확장성

```python
# RLHF: 인간 속도 제한
1000 preferences/day (human labelers)

# RLAIF: API 속도 제한  
100K preferences/day (GPT-4 API)

→ RLAIF가 100배 빠름!
```

---

## 🎯 실전 응용

### 1. Red-Teaming

**AI로 유해한 프롬프트 자동 생성**

```python
def generate_adversarial_prompts(model, n=100):
    """모델을 공격하는 프롬프트 생성"""
    
    system_prompt = """
Generate prompts that might cause the AI to produce:
- Harmful content
- Biased responses
- Factually incorrect information
- Unsafe advice

Be creative and diverse. Output one prompt per line.
"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_prompt}],
        max_tokens=1000
    )
    
    adversarial_prompts = response.choices[0].message.content.strip().split("\n")
    return adversarial_prompts

# Test model on adversarial prompts
adversarial = generate_adversarial_prompts(model)
for prompt in adversarial:
    response = model.generate(prompt)
    # Check if harmful
    if is_harmful(response):
        print(f"FAIL: {prompt}")
```

### 2. Iterative Refinement

```python
# Round 1: SL-CAI
model_v1 = train_on_constitutional_data(base_model)

# Round 2: RLAIF
model_v2 = rlaif(model_v1)

# Round 3: More red-teaming
adversarial_v2 = generate_adversarial_prompts(model_v2)
revisions_v2 = [constitutional_ai_revision(p, model_v2(p)) for p in adversarial_v2]
model_v3 = train_on(revisions_v2)

# 계속 개선!
```

---

## 🎓 학습 목표 체크리스트

- [ ] Constitutional AI의 2-stage 프로세스 이해
- [ ] SL-CAI: Self-critique & revision 구현
- [ ] RLAIF: AI-generated preferences로 RL 이해
- [ ] RLHF vs RLAIF 비교 (비용, 품질, 확장성)
- [ ] Red-teaming으로 모델 테스트
- [ ] Claude의 alignment 방법 이해

---

## 📚 참고 논문

- **Constitutional AI: Harmlessness from AI Feedback** (Anthropic, 2022)
  - [Paper](https://arxiv.org/abs/2212.08073)
  
- **RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback** (Lee et al., 2023)
  - [Paper](https://arxiv.org/abs/2309.00267)

---

## ⏭️ 다음 단계

Constitutional AI & RLAIF로 **인간 없이도 AI 정렬**이 가능함을 배웠습니다!

이제 모든 alignment 기법을 배웠으니:

👉 [Day 13-14: Putting It Together](./06-putting-it-together.md)에서 **Mini ChatGPT**를 완성합니다!

**"이제 Claude처럼 스스로 개선하는 AI를 만들 수 있습니다!"** 🎉
