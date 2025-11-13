# Day 1-2: Instruction Tuning

## 🎯 목표

**Base Language Model → Instruction-Following Model**

```python
# Before: Base GPT
prompt = "Translate to French: Hello"
output = "Translate to French: Hello\nTranslate to Spanish: Hola\n..."  # 계속 패턴 반복

# After: Instruction-tuned
prompt = "Translate to French: Hello"
output = "Bonjour"  # 지시를 따름!
```

---

## 📖 핵심 개념

### 왜 Instruction Tuning이 필요한가?

**Base LM의 문제점**
```python
# GPT-3 (Base model) - Next token prediction 학습
prompt = "Q: What is the capital of France?\nA:"
output = "What is the capital of Germany?\nA:"  # 질문 패턴만 학습!
```

**훈련 데이터의 차이**

| Base LM | Instruction-tuned LM |
|---------|---------------------|
| 인터넷 텍스트 (무작위) | 지시-응답 쌍 (의도적) |
| "Paris is the capital..." | "Q: Capital of France?\nA: Paris" |
| 패턴 학습 | 지시 따르기 학습 |

---

## 🔧 Supervised Fine-Tuning (SFT)

### 1. 데이터셋 구조

**기본 포맷**
```json
{
  "instruction": "아래 문장을 한국어로 번역하세요.",
  "input": "The weather is nice today.",
  "output": "오늘 날씨가 좋습니다."
}
```

**Alpaca 포맷**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

**ShareGPT 포맷** (대화형)
```json
{
  "conversations": [
    {"from": "human", "value": "Python으로 정렬 알고리즘 짜줘"},
    {"from": "gpt", "value": "물론이죠! 여기 퀵정렬 구현입니다:\n\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```"}
  ]
}
```

### 2. 주요 데이터셋

**Alpaca (52K instructions)**
- Stanford에서 GPT-3.5로 생성
- Self-Instruct 방법 사용
- 다양한 task 커버

**Dolly (15K instructions)**
- Databricks에서 인간이 직접 작성
- 고품질 보장
- 상업적 사용 가능

**FLAN Collection (1800+ tasks)**
- Google의 대규모 instruction 컬렉션
- Task 다양성 극대화

**Evol-Instruct**
- WizardLM에서 사용
- GPT-4로 instruction을 점진적으로 복잡하게 진화

---

## 💻 실습 1: GPT-2 Instruction Tuning

### 준비

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

# 1. 모델 & 토크나이저 로드
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Padding token 설정 (GPT-2는 기본적으로 없음)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
```

### 데이터 준비

```python
# Alpaca 스타일 데이터 로드
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Prompt 템플릿 정의
PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

def format_instruction(example):
    """Instruction을 프롬프트로 변환"""
    prompt = PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        output=example["output"]
    )
    return {"text": prompt}

# 데이터셋 변환
formatted_dataset = dataset.map(format_instruction)

# 토크나이징
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=formatted_dataset.column_names
)
```

### Fine-tuning

```python
# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-instruct",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    fp16=True,  # Mixed precision
    report_to="tensorboard"
)

# Data collator (labels = input_ids for causal LM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM (not masked LM)
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 훈련 시작!
trainer.train()

# 모델 저장
model.save_pretrained("./gpt2-instruct-final")
tokenizer.save_pretrained("./gpt2-instruct-final")
```

### 추론 테스트

```python
def generate_response(instruction, model, tokenizer):
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response = response.split("### Response:")[-1].strip()
    return response

# 테스트
print(generate_response(
    "Python으로 피보나치 수열을 구하는 함수를 작성하세요.",
    model,
    tokenizer
))
```

---

## 🔬 고급: Self-Instruct

### 개념

**인간이 직접 데이터 만들기 = 비쌈**
→ LLM으로 데이터 자동 생성!

```python
# Self-Instruct 프로세스
1. Seed instructions (175개 정도)
2. LLM으로 새 instruction 생성
3. LLM으로 response 생성
4. 품질 필터링
5. 반복
```

### 구현 예시

```python
import openai

def generate_instructions(seed_instructions, n=100):
    """GPT-3.5/4로 새로운 instruction 생성"""

    prompt = f"""Generate {n} diverse task instructions. Here are some examples:

{chr(10).join(seed_instructions[:5])}

Generate new instructions that are:
1. Clear and specific
2. Diverse (different topics and styles)
3. Realistic user requests

Output format (one per line):
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,  # High diversity
        max_tokens=1500
    )

    new_instructions = response.choices[0].message.content.strip().split("\n")
    return [inst.strip() for inst in new_instructions if inst.strip()]

def generate_response_for_instruction(instruction):
    """Instruction에 대한 response 생성"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": instruction}],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

# 사용 예시
seed_instructions = [
    "Python으로 정렬 알고리즘을 구현하세요.",
    "프랑스어로 번역하세요: Hello, how are you?",
    "다음 문장의 감정을 분석하세요: I love this product!",
]

# 새 instruction 생성
new_instructions = generate_instructions(seed_instructions, n=50)

# 각 instruction에 대한 response 생성
dataset = []
for instruction in new_instructions:
    try:
        response = generate_response_for_instruction(instruction)
        dataset.append({
            "instruction": instruction,
            "output": response
        })
    except Exception as e:
        print(f"Error: {e}")
        continue

print(f"생성된 데이터: {len(dataset)}개")
```

---

## 🎯 Evol-Instruct (WizardLM)

### 개념

**Instruction을 점진적으로 복잡하게 진화**

```
Simple: "정렬 알고리즘을 설명하세요."
  ↓ Evol
Medium: "버블 정렬과 퀵 정렬의 시간 복잡도를 비교하세요."
  ↓ Evol
Complex: "O(n log n) 정렬 알고리즘 3가지를 구현하고, 각각의 장단점을 실제 벤치마크 결과와 함께 분석하세요."
```

### Evolution 연산자

```python
EVOLUTION_PROMPTS = {
    "add_constraints": "I want you to act as a Prompt Rewriter. Add more specific constraints to the following instruction: {instruction}",

    "deepen": "I want you to act as a Prompt Rewriter. Make the following instruction more in-depth and complex: {instruction}",

    "concretize": "I want you to act as a Prompt Rewriter. Replace general concepts with more specific concepts in the following instruction: {instruction}",

    "increase_reasoning": "I want you to act as a Prompt Rewriter. Rewrite the following instruction to require multi-step reasoning: {instruction}",
}

def evolve_instruction(instruction, operator="deepen"):
    prompt = EVOLUTION_PROMPTS[operator].format(instruction=instruction)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# 예시
simple = "Python으로 리스트를 정렬하세요."
evolved = evolve_instruction(simple, "increase_reasoning")
print(evolved)
# 출력: "Python으로 리스트를 정렬하되, 다음 조건을 만족해야 합니다:
#        1) 제자리 정렬일 것
#        2) 시간 복잡도 O(n log n)
#        3) 안정 정렬일 것
#        각 조건이 왜 중요한지 설명하고 구현하세요."
```

---

## 📊 데이터 품질의 중요성

### 품질 > 수량

```python
# 실험 결과 (Alpaca 논문)
52K 고품질 instructions > 500K 저품질 instructions
```

### 필터링 전략

```python
def filter_instruction(instruction, response):
    """저품질 데이터 제거"""

    # 1. 너무 짧은 것
    if len(response) < 20:
        return False

    # 2. 반복적인 것
    if response.count(response[:10]) > 2:
        return False

    # 3. 프롬프트가 그대로 반복되는 것
    if instruction.lower() in response.lower():
        return False

    # 4. "I cannot", "I'm sorry" 등 거부 응답
    refusal_phrases = ["cannot", "sorry", "can't help"]
    if any(phrase in response.lower() for phrase in refusal_phrases):
        return False

    return True

# 품질 점수 매기기
def score_quality(instruction, response):
    """0-10 점수"""
    score = 5.0

    # 길이 적절
    if 50 < len(response) < 500:
        score += 1

    # 구체적 (예시 포함)
    if "```" in response or "example" in response.lower():
        score += 1

    # 구조화 (번호 매김, 단계)
    if any(marker in response for marker in ["1.", "2.", "Step", "-"]):
        score += 1

    # 다양한 단어 사용
    unique_words = len(set(response.split()))
    if unique_words / len(response.split()) > 0.7:
        score += 1

    return score
```

---

## 🎓 학습 목표 체크리스트

- [ ] Instruction tuning과 pre-training의 차이 이해
- [ ] Alpaca, Dolly, FLAN 데이터셋 형식 이해
- [ ] GPT-2를 instruction-following으로 fine-tune 완료
- [ ] Self-Instruct 원리 이해 및 간단 구현
- [ ] Evol-Instruct로 복잡한 instruction 생성
- [ ] 데이터 품질 필터링 구현

---

## 🔥 실전 팁

### 1. 프롬프트 템플릿 중요

```python
# Bad: 일관성 없음
"Translate: Hello" → "Bonjour"
"번역해줘: Hello" → "Bonjour"

# Good: 일관된 템플릿
"### Instruction: Translate to French\n### Input: Hello\n### Response: Bonjour"
"### Instruction: 프랑스어로 번역\n### Input: Hello\n### Response: Bonjour"
```

### 2. Task 다양성

```python
# 다양한 task 포함
tasks = [
    "Question Answering",
    "Summarization",
    "Translation",
    "Code Generation",
    "Creative Writing",
    "Math Problem Solving",
    "Logical Reasoning",
    # ...
]
```

### 3. Learning Rate 조정

```python
# Base LM fine-tuning보다 낮은 LR
learning_rate = 1e-5  # vs 5e-5 for base training

# Warmup 사용
warmup_ratio = 0.1
```

---

## 📚 참고 자료

- **Alpaca**: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- **Self-Instruct**: [Paper](https://arxiv.org/abs/2212.10560)
- **Evol-Instruct**: [WizardLM Paper](https://arxiv.org/abs/2304.12244)
- **FLAN**: [FLAN Collection](https://github.com/google-research/FLAN)

---

## ⏭️ 다음 단계

Instruction tuning으로 모델이 지시를 따르게 되었습니다!

하지만 아직 부족합니다:
- "도움이 되는" 답변을 선호하지 않음
- 유해한 내용 생성 가능
- 인간 선호도를 모름

👉 [Day 3-5: RLHF](./02-rlhf.md)에서 **인간 선호도 학습**을 배웁니다!

**"이제 ChatGPT처럼 만들 차례입니다!"** 🚀
