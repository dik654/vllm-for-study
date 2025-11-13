# Day 5-6: Benchmarks & Datasets

## 🎯 목표

**AI 모델 평가의 표준을 이해하고 커스텀 벤치마크 구축하기**

```python
# 이 장에서 배울 것:
- 평가 지표: Perplexity, BLEU, ROUGE, FID 등
- 주요 벤치마크: MMLU, HumanEval, MT-Bench
- 데이터셋 로딩 및 전처리
- Custom 평가 파이프라인
```

---

## 📊 Part 1: 평가 지표 (Metrics)

### 1.1 Language Models

#### Perplexity (혼란도)

**정의**: "모델이 얼마나 확신하는가?"

```python
# Lower is better!
PPL = exp(-1/N * Σ log P(token_i))

# 직관:
# PPL = 10: 평균적으로 10개 단어 중 고민
# PPL = 100: 평균적으로 100개 단어 중 고민
```

**구현**:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(model, tokenizer, text):
    """
    Compute perplexity on text
    """
    # Tokenize
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids

    # Compute loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    # Perplexity = exp(loss)
    perplexity = torch.exp(loss)

    return perplexity.item()

# Usage
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

text = "The quick brown fox jumps over the lazy dog"
ppl = compute_perplexity(model, tokenizer, text)

print(f"Perplexity: {ppl:.2f}")
```

#### BLEU (Bilingual Evaluation Understudy)

**용도**: Machine Translation, Text Generation

**정의**: n-gram precision

```python
# BLEU = BP × exp(Σ w_n log p_n)

# p_n: n-gram precision
# BP: Brevity penalty (짧은 번역 페널티)
```

**구현**:

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def compute_bleu(reference, candidate):
    """
    reference: List of reference sentences
    candidate: Generated sentence
    """
    # Tokenize
    reference_tokens = [ref.split() for ref in reference]
    candidate_tokens = candidate.split()

    # BLEU-4 (unigram to 4-gram)
    score = sentence_bleu(reference_tokens, candidate_tokens)

    return score

# Example
references = [
    "the cat is on the mat",
    "there is a cat on the mat"
]
candidate = "the cat is sitting on the mat"

bleu = compute_bleu(references, candidate)
print(f"BLEU: {bleu:.4f}")
```

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**용도**: Summarization

**Types**:
- ROUGE-N: n-gram overlap
- ROUGE-L: Longest Common Subsequence
- ROUGE-S: Skip-bigram

**구현**:

```python
from rouge_score import rouge_scorer

def compute_rouge(reference, candidate):
    """
    Compute ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

# Example
reference = "the cat was found under the bed"
candidate = "the cat was under the bed"

scores = compute_rouge(reference, candidate)
print(f"ROUGE-1: {scores['rouge1']:.4f}")
print(f"ROUGE-2: {scores['rouge2']:.4f}")
print(f"ROUGE-L: {scores['rougeL']:.4f}")
```

---

### 1.2 Vision Models

#### FID (Fréchet Inception Distance)

**용도**: Generative models (GAN, Diffusion)

**정의**: 생성 이미지와 실제 이미지의 분포 거리

```python
# FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2√(Σ_real Σ_fake))

# Lower is better! (< 10 = excellent)
```

**구현**:

```python
from pytorch_fid import fid_score

def compute_fid(real_images_path, generated_images_path):
    """
    Compute FID score

    Args:
        real_images_path: Directory with real images
        generated_images_path: Directory with generated images
    """
    fid = fid_score.calculate_fid_given_paths(
        [real_images_path, generated_images_path],
        batch_size=50,
        device='cuda',
        dims=2048  # Inception features
    )

    return fid

# Usage
fid = compute_fid('data/real/', 'data/generated/')
print(f"FID: {fid:.2f}")
```

#### IS (Inception Score)

**용도**: Generative models

**정의**: 생성 이미지의 quality와 diversity

```python
# IS = exp(E[KL(p(y|x) || p(y))])

# Higher is better! (> 10 = good)
```

**구현**:

```python
from torchmetrics.image.inception import InceptionScore

def compute_inception_score(images):
    """
    Compute Inception Score

    Args:
        images: Tensor of shape (N, 3, H, W)
    """
    inception = InceptionScore(normalize=True)
    inception.update(images)

    mean, std = inception.compute()

    return mean.item(), std.item()

# Usage
import torch

# Generate fake images (for demo)
fake_images = torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8)

mean, std = compute_inception_score(fake_images)
print(f"IS: {mean:.2f} ± {std:.2f}")
```

---

## 🎯 Part 2: 주요 벤치마크

### 2.1 MMLU (Massive Multitask Language Understanding)

**측정**: 57개 과목 지식 (수학, 역사, 법률 등)

**Format**:
```python
{
    "question": "What is the capital of France?",
    "choices": ["London", "Berlin", "Paris", "Madrid"],
    "answer": 2
}
```

**구현**:

```python
from datasets import load_dataset
from transformers import pipeline

def evaluate_mmlu(model_name, subject='all', num_shots=5):
    """
    Evaluate model on MMLU

    Args:
        model_name: HuggingFace model name
        subject: Subject name or 'all'
        num_shots: Few-shot examples (0, 5, or 10)
    """
    # Load dataset
    dataset = load_dataset("cais/mmlu", subject)

    # Load model
    generator = pipeline('text-generation', model=model_name, device=0)

    correct = 0
    total = 0

    for example in dataset['test']:
        # Format prompt
        prompt = format_mmlu_prompt(example, num_shots)

        # Generate
        output = generator(prompt, max_new_tokens=1, do_sample=False)
        prediction = extract_answer(output[0]['generated_text'])

        # Check
        if prediction == example['answer']:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy

def format_mmlu_prompt(example, num_shots):
    """
    Format MMLU prompt with few-shot examples
    """
    prompt = ""

    # Few-shot examples (if num_shots > 0)
    if num_shots > 0:
        # Add few-shot examples here
        pass

    # Question
    prompt += f"Question: {example['question']}\n"

    # Choices
    for i, choice in enumerate(example['choices']):
        prompt += f"{chr(65+i)}. {choice}\n"

    prompt += "Answer:"

    return prompt

# Usage
accuracy = evaluate_mmlu('meta-llama/Llama-2-7b-hf', subject='mathematics', num_shots=5)
print(f"MMLU Accuracy: {accuracy:.2%}")
```

---

### 2.2 HumanEval (Code Generation)

**측정**: Python 코드 생성 능력

**Format**:
```python
{
    "task_id": "HumanEval/0",
    "prompt": "def has_close_elements(numbers, threshold):\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    \"\"\"",
    "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False",
    "test": "...",
    "entry_point": "has_close_elements"
}
```

**구현**:

```python
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

def evaluate_humaneval(model_name, num_samples=1):
    """
    Evaluate model on HumanEval

    Args:
        model_name: HuggingFace model name
        num_samples: Number of samples per problem
    """
    # Load problems
    problems = read_problems()

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Generate solutions
    samples = []
    for task_id, problem in problems.items():
        prompt = problem['prompt']

        # Generate
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_return_sequences=num_samples,
            temperature=0.2,
            do_sample=True
        )

        # Decode
        for output in outputs:
            completion = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            samples.append({
                'task_id': task_id,
                'completion': completion
            })

    # Save samples
    write_jsonl('samples.jsonl', samples)

    # Evaluate
    results = evaluate_functional_correctness('samples.jsonl')

    return results['pass@1']

# Usage
pass_at_1 = evaluate_humaneval('codegen-350M-mono')
print(f"Pass@1: {pass_at_1:.2%}")
```

---

### 2.3 MT-Bench (Chatbot)

**측정**: Multi-turn conversation quality

**Categories**: Writing, Roleplay, Reasoning, Math, Coding, etc.

**구현**:

```python
def evaluate_mt_bench(model_name, judge_model='gpt-4'):
    """
    Evaluate model on MT-Bench using GPT-4 as judge

    Args:
        model_name: Model to evaluate
        judge_model: Judge model (usually GPT-4)
    """
    from fastchat.llm_judge.gen_model_answer import generate_answers
    from fastchat.llm_judge.gen_judgment import generate_judgments

    # Generate answers
    generate_answers(
        model_name=model_name,
        bench_name='mt_bench',
        num_turns=2
    )

    # Judge answers
    scores = generate_judgments(
        judge_model=judge_model,
        model_name=model_name
    )

    # Average score
    avg_score = sum(scores.values()) / len(scores)

    return avg_score

# Usage
score = evaluate_mt_bench('vicuna-7b-v1.5')
print(f"MT-Bench Score: {score:.2f}/10")
```

---

## 📦 Part 3: Datasets

### 3.1 Loading Datasets

**HuggingFace Datasets**:

```python
from datasets import load_dataset

# GLUE (NLP)
glue = load_dataset('glue', 'sst2')
print(glue['train'][0])
# {'sentence': 'hide new secretions from the parental units', 'label': 0}

# ImageNet (Vision)
imagenet = load_dataset('imagenet-1k', split='train', streaming=True)

# COCO (Vision)
coco = load_dataset('detection-datasets/coco', split='train')

# C4 (Language Modeling)
c4 = load_dataset('c4', 'en', split='train', streaming=True)
```

### 3.2 Custom Dataset

```python
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': item['label']
        }

# Convert to HuggingFace Dataset
def create_hf_dataset(data_list):
    """
    Convert list of dicts to HuggingFace Dataset
    """
    dataset = HFDataset.from_list(data_list)
    return dataset

# Usage
data = [
    {'text': 'This is great!', 'label': 1},
    {'text': 'This is bad.', 'label': 0}
]

hf_dataset = create_hf_dataset(data)
print(hf_dataset)
```

### 3.3 Data Preprocessing

**Common Pipelines**:

```python
def preprocess_text(examples):
    """
    Preprocess text data
    """
    # Lowercase
    examples['text'] = examples['text'].lower()

    # Remove special characters
    examples['text'] = re.sub(r'[^a-zA-Z0-9\s]', '', examples['text'])

    # Tokenize
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

# Apply to dataset
dataset = dataset.map(preprocess_text, batched=True)

# Filter
dataset = dataset.filter(lambda x: len(x['text']) > 10)

# Shuffle
dataset = dataset.shuffle(seed=42)

# Split
train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test['train']
test_dataset = train_test['test']
```

---

## 🔧 Part 4: Custom Evaluation Pipeline

### 4.1 Complete Pipeline

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class EvaluationResult:
    """Store evaluation results"""
    metric_name: str
    score: float
    samples: List[Dict]

class EvaluationPipeline:
    """
    Custom evaluation pipeline
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.results = []

    def evaluate_perplexity(self, dataset):
        """Evaluate perplexity"""
        total_loss = 0
        total_tokens = 0

        for example in dataset:
            # Tokenize
            inputs = self.tokenizer(example['text'], return_tensors='pt')

            # Compute loss
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss

            total_loss += loss.item() * inputs['input_ids'].size(1)
            total_tokens += inputs['input_ids'].size(1)

        # Perplexity
        perplexity = np.exp(total_loss / total_tokens)

        result = EvaluationResult(
            metric_name='perplexity',
            score=perplexity,
            samples=[]
        )

        self.results.append(result)
        return perplexity

    def evaluate_accuracy(self, dataset, label_key='label'):
        """Evaluate classification accuracy"""
        correct = 0
        total = 0
        samples = []

        for example in dataset:
            # Generate
            inputs = self.tokenizer(example['text'], return_tensors='pt')
            outputs = self.model(**inputs)

            # Predict
            prediction = outputs.logits.argmax(dim=-1).item()
            label = example[label_key]

            if prediction == label:
                correct += 1

            total += 1

            samples.append({
                'text': example['text'],
                'prediction': prediction,
                'label': label,
                'correct': prediction == label
            })

        accuracy = correct / total

        result = EvaluationResult(
            metric_name='accuracy',
            score=accuracy,
            samples=samples
        )

        self.results.append(result)
        return accuracy

    def evaluate_generation(self, prompts, reference_texts):
        """Evaluate generation quality"""
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        rouge_scores = []
        samples = []

        for prompt, reference in zip(prompts, reference_texts):
            # Generate
            inputs = self.tokenizer(prompt, return_tensors='pt')
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Score
            scores = scorer.score(reference, generated)

            rouge_scores.append({
                'rouge1': scores['rouge1'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            })

            samples.append({
                'prompt': prompt,
                'reference': reference,
                'generated': generated,
                'scores': scores
            })

        # Average
        avg_rouge1 = np.mean([s['rouge1'] for s in rouge_scores])
        avg_rougeL = np.mean([s['rougeL'] for s in rouge_scores])

        result = EvaluationResult(
            metric_name='generation',
            score=avg_rouge1,
            samples=samples
        )

        self.results.append(result)

        return {'rouge1': avg_rouge1, 'rougeL': avg_rougeL}

    def generate_report(self):
        """Generate evaluation report"""
        report = "=" * 50 + "\n"
        report += "EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"

        for result in self.results:
            report += f"{result.metric_name.upper()}: {result.score:.4f}\n"

            if result.samples:
                report += f"  Sample Results:\n"
                for i, sample in enumerate(result.samples[:3]):  # Show first 3
                    report += f"    [{i+1}] {sample}\n"

            report += "\n"

        return report

# Usage
pipeline = EvaluationPipeline(model, tokenizer)

# Evaluate
perplexity = pipeline.evaluate_perplexity(test_dataset)
accuracy = pipeline.evaluate_accuracy(test_dataset)

# Report
report = pipeline.generate_report()
print(report)
```

---

## 🎓 학습 목표

- [ ] 주요 평가 지표 이해 및 구현
- [ ] MMLU, HumanEval 벤치마크 실행
- [ ] HuggingFace Datasets 로딩 및 전처리
- [ ] Custom evaluation pipeline 구축
- [ ] 벤치마크 결과 해석

---

## 💡 실전 팁

### Metric 선택

```python
# Task별 권장 metric:

# Language Generation:
- BLEU: Translation
- ROUGE: Summarization
- Perplexity: Language modeling

# Classification:
- Accuracy: Balanced datasets
- F1: Imbalanced datasets
- AUC-ROC: Binary classification

# Generation (Vision):
- FID: Distribution similarity
- IS: Quality + Diversity
- CLIP Score: Text-image alignment
```

### 벤치마크 해석

```python
# Good scores:
- Perplexity < 20
- BLEU > 30
- FID < 10
- MMLU > 60%
- HumanEval Pass@1 > 40%

# 주의: Absolute numbers보다 relative improvement가 중요!
```

---

## ⏭️ 다음

👉 [Day 7: Training Pipeline](./04-training-pipeline.md)

**이제 모델 훈련 파이프라인을 마스터합니다!** 🚀
