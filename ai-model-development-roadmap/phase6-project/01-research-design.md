# Week 13-14: Research & Design

## 🎯 목표

**독창적인 AI 프로젝트를 체계적으로 설계하기**

---

## 📋 Week 13: 문제 정의 및 연구

### Day 1-2: 문제 선정

#### 1. 문제 찾기

**좋은 문제의 조건**:
- 해결 가능 (4주 내)
- 측정 가능 (명확한 메트릭)
- 가치 있음 (실제 활용 가능)
- 흥미로움 (지속 가능한 동기)

**브레인스토밍 질문**:
```
1. 어떤 AI 기술에 가장 관심이 있는가?
   - NLP? Vision? Multi-modal? Audio?

2. 어떤 문제가 아직 잘 해결되지 않았는가?
   - Efficiency? Quality? Controllability? Fairness?

3. 기존 모델의 어떤 점이 아쉬운가?
   - 속도? 메모리? 정확도? 다양성?

4. 어떤 데이터를 사용할 수 있는가?
   - 공개 데이터셋? 수집 가능? 크기는?
```

#### 2. 문제 명세서 작성

**Template**:
```markdown
# Problem Statement

## Background
- 현재 상황과 문제점
- 왜 이 문제가 중요한가?

## Goal
- 구체적으로 무엇을 달성하려 하는가?
- 성공의 기준은?

## Constraints
- 시간: 4주
- 데이터: 무엇을 사용할 수 있는가?
- 컴퓨팅: GPU 사양, 예산

## Success Metrics
- Primary: FID < 20, BLEU > 30, etc.
- Secondary: Speed, memory, etc.
```

#### 3. 타당성 검증

**체크리스트**:
- [ ] 4주 내 완료 가능한가?
- [ ] 필요한 데이터를 구할 수 있는가?
- [ ] 계산 자원이 충분한가?
- [ ] Baseline 모델이 존재하는가?
- [ ] 개선 방향이 명확한가?

---

### Day 3-4: 문헌 조사

#### 1. 논문 검색 전략

**Where to search**:
- arXiv.org
- Google Scholar
- Papers with Code
- Semantic Scholar

**Keywords**:
```
- Your problem + "neural network"
- Your problem + "transformer"
- Your problem + "recent"
- Your problem + "survey"
```

#### 2. 논문 읽기 전략

**3-Pass Method**:

**Pass 1 (5분)**: Skim
- Title, abstract, conclusion
- Section headings
- Figures

**Pass 2 (30분)**: Read actively
- Introduction: Problem + motivation
- Related work: What exists?
- Method: How do they solve it?
- Results: Does it work?
- Skip: Math details, implementation

**Pass 3 (2시간)**: Deep dive (for 3-5 key papers)
- Every equation
- Implementation details
- Supplementary materials

#### 3. 정리 방법

**Paper Notes Template**:
```markdown
## [Paper Title] (Year)

### Main Idea
- One sentence summary

### Key Contributions
1. ...
2. ...

### Method
- Architecture diagram
- Key equations
- Novel components

### Results
- Main metrics
- Comparison with baselines

### Strengths
- What's good?

### Weaknesses
- What's missing?

### Relevance to My Project
- How can I use this?
- What can I improve?
```

---

### Day 5-7: 데이터셋 구축

#### 1. 데이터셋 선택

**Option A: 기존 데이터셋 사용**

**Popular Datasets**:
- NLP: GLUE, SQuAD, WikiText, C4
- Vision: ImageNet, COCO, CelebA, FFHQ
- Multi-modal: LAION, Conceptual Captions

**Evaluation Checklist**:
- [ ] Size: 충분한가? (최소 10K samples)
- [ ] Quality: Clean한가?
- [ ] Diversity: 다양한가?
- [ ] License: 상업적 사용 가능한가?
- [ ] Splits: Train/val/test 제공되는가?

**Option B: 새 데이터 수집**

**Steps**:
1. 데이터 소스 파악
2. 크롤링/API 스크립트 작성
3. 법적/윤리적 검토
4. 자동 수집
5. Manual verification (sample)

#### 2. 데이터 전처리

**Standard Pipeline**:
```python
# 1. Load
data = load_dataset("dataset_name")

# 2. Clean
data = remove_duplicates(data)
data = filter_low_quality(data)

# 3. Normalize
# - Text: Lowercase, remove special chars
# - Image: Resize, normalize pixels
# - Audio: Resample, normalize volume

# 4. Split
train, val, test = split_data(data, [0.8, 0.1, 0.1])

# 5. Save
save_dataset(train, "train.pkl")
save_dataset(val, "val.pkl")
save_dataset(test, "test.pkl")
```

#### 3. 데이터 분석

**EDA (Exploratory Data Analysis)**:
```python
# Distribution
plt.hist(label_counts)
plt.title("Class Distribution")

# Statistics
print(f"Mean length: {np.mean(lengths)}")
print(f"Std length: {np.std(lengths)}")

# Samples
for i in range(5):
    print(f"Sample {i}: {data[i]}")

# Imbalance check
if max(counts) / min(counts) > 10:
    print("WARNING: Class imbalance detected!")
```

---

## 🏗️ Week 14: 아키텍처 설계

### Day 1-3: 초기 설계

#### 1. Baseline 선정

**선택 기준**:
- 널리 사용되는 표준 모델
- 재현 가능 (공개 코드)
- 계산 가능 (GPU 메모리 안에서)

**Examples**:
- NLP: BERT-base, GPT-2
- Vision: ResNet-50, ViT-base
- Diffusion: DDPM, Stable Diffusion
- GAN: StyleGAN2

#### 2. 아이디어 브레인스토밍

**Improvement Vectors**:

**1. Architecture**:
- 새로운 attention mechanism?
- 효율적인 layer (MoE, LoRA)?
- Better inductive bias?

**2. Training**:
- 새로운 loss function?
- Better optimization?
- Data augmentation?

**3. Inference**:
- Faster sampling?
- Better quality-speed tradeoff?
- Controllability?

**Validation Questions**:
```
1. 왜 이 아이디어가 작동해야 하는가?
   - 이론적 근거는?

2. 기존 연구와의 차이는?
   - Novel contribution은?

3. 구현 가능한가?
   - 4주 내 완성 가능?

4. 측정 가능한가?
   - 개선을 어떻게 증명?
```

#### 3. 설계 문서 작성

**Architecture Design Doc**:
```markdown
## Model Architecture

### Overview
- High-level diagram
- Input → Output flow

### Components

#### Component 1: [Name]
- Purpose: Why this component?
- Input: Shape and type
- Output: Shape and type
- Implementation: Key details

#### Component 2: [Name]
...

### Novel Contributions
1. What's new?
2. Why it's better?

### Computational Complexity
- Parameters: XX M
- FLOPs: XX G
- Memory: XX GB
```

---

### Day 4-5: Ablation Study 계획

#### 실험 매트릭스

**Template**:
```
| Experiment | Component A | Component B | Component C | Metric |
|------------|-------------|-------------|-------------|--------|
| Baseline   | ✗           | ✗           | ✗           | ?      |
| +A         | ✓           | ✗           | ✗           | ?      |
| +B         | ✗           | ✓           | ✗           | ?      |
| +A+B       | ✓           | ✓           | ✗           | ?      |
| +A+B+C     | ✓           | ✓           | ✓           | ?      |
```

**목적**: 각 컴포넌트의 기여도 측정!

---

### Day 6-7: 구현 준비

#### 1. 코드 구조

**Recommended Structure**:
```
my-project/
├── data/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── baseline.py
│   └── my_model.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── loss.py
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
├── configs/
│   ├── baseline.yaml
│   └── my_model.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
├── tests/
│   └── test_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

#### 2. 개발 환경

**requirements.txt**:
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
wandb>=0.15.0
pytest>=7.3.0
```

#### 3. Git Setup

```bash
git init
git add .
git commit -m "Initial commit: Project structure"
git branch -M main
git remote add origin <your-repo>
git push -u origin main
```

---

## 🎓 Deliverables (Week 13-14)

완료 시 다음을 가지고 있어야 합니다:

- [ ] Problem statement (1-2 페이지)
- [ ] Literature review (10+ papers, organized notes)
- [ ] Dataset (cleaned, split, analyzed)
- [ ] Architecture design doc (with diagrams)
- [ ] Ablation study plan (experiment matrix)
- [ ] Code skeleton (folders, configs, README)
- [ ] Git repository (initialized, first commit)

---

## ⏭️ 다음

👉 [Week 15-16: Implementation & Experiments](./02-implementation-experiments.md)

**이제 설계를 코드로 구현하고 실험을 진행합니다!** 🚀
