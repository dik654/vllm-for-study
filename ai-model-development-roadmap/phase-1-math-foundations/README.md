# Phase -1: AI를 위한 수학적 기초

## 🎯 왜 이 Phase가 필요한가?

**"수학 없이 AI를 하는 것은 악보 없이 피아노를 치는 것과 같습니다."**

### 현실 체크 ✅

Phase 0부터 바로 시작하면:
- ❌ 논문의 수식을 읽을 수 없음
- ❌ "왜 이렇게 동작하는가?"를 이해할 수 없음
- ❌ 디버깅할 때 근본 원인을 찾을 수 없음
- ❌ 새로운 아이디어를 설계할 수 없음

### 이 Phase를 완료하면 ✨

- ✅ Transformer 논문의 모든 수식 이해
- ✅ Backpropagation을 손으로 계산 가능
- ✅ Loss function을 설계할 수 있음
- ✅ Optimizer의 동작 원리 설명 가능
- ✅ 논문의 증명을 따라갈 수 있음

---

## 📚 커리큘럼 (2주, 80시간)

대학 4년 수학을 **AI에 실제로 필요한 2주**로 압축했습니다.

### Week 1: 선형대수 & 미적분

#### [Day 1-2: 선형대수 기초](./01-linear-algebra-basics.md)
**AI 연결**: Embeddings, Attention, Matrix Operations
- Vectors와 Matrices
- Matrix 곱셈의 의미
- 전치(Transpose), 역행렬(Inverse)
- 내적(Dot Product)과 코사인 유사도
- 💻 **실습**: NumPy로 Attention 수학 구현

#### [Day 3-4: 선형대수 심화](./02-linear-algebra-advanced.md)
**AI 연결**: PCA, SVD, Low-rank Approximation (LoRA!)
- Eigenvalues & Eigenvectors
- Singular Value Decomposition (SVD)
- Matrix Decomposition
- Rank와 LoRA의 관계
- 💻 **실습**: LoRA를 수학적으로 이해하고 구현

#### [Day 5-6: 미적분](./03-calculus.md)
**AI 연결**: Backpropagation, Gradient Descent
- Derivatives (미분)
- Partial Derivatives (편미분)
- Chain Rule (연쇄 법칙) ⭐ **가장 중요!**
- Gradients (∇)
- Jacobian, Hessian
- 💻 **실습**: Backpropagation 손으로 계산 → 코드 구현

#### [Day 7: 최적화 기초](./04-optimization.md)
**AI 연결**: SGD, Adam, Learning Rate
- Gradient Descent의 수학
- Convex vs Non-convex
- Local Minima, Saddle Points
- Momentum, Adam의 수학
- 💻 **실습**: Optimizer를 밑바닥부터 구현

### Week 2: 확률/통계 & 정보이론

#### [Day 8-9: 확률과 통계](./05-probability-statistics.md)
**AI 연결**: Dropout, Batch Norm, Sampling
- 확률 분포 (Gaussian, Bernoulli, Categorical)
- 기댓값(Expectation), 분산(Variance)
- Bayes' Theorem
- Maximum Likelihood Estimation (MLE)
- 💻 **실습**: Gaussian 분포로 노이즈 생성 (Diffusion 준비)

#### [Day 10-11: 정보이론](./06-information-theory.md)
**AI 연결**: Cross Entropy Loss, KL Divergence, Attention
- Entropy (엔트로피)
- Cross Entropy
- KL Divergence
- Mutual Information
- 💻 **실습**: Loss function의 수학적 유도

#### [Day 12-14: 종합 실습](./07-putting-it-together.md)
**AI 모델의 수학 완전 분해**
- Softmax의 수학 (선형대수 + 확률)
- Attention의 수학 (행렬곱 + softmax)
- Backpropagation 전체 유도
- Loss function 설계
- 💻 **최종 프로젝트**: 2-layer Neural Network를 수식 → 코드 완전 구현

---

## 🎓 학습 철학

### 1. **AI 중심 (Not 순수 수학)**
- 증명보다 **직관**과 **응용**
- 대학 수학의 10%만, 하지만 AI의 90%를 커버
- "어떻게 쓰이는가?"에 집중

### 2. **코드로 배우기**
```
수식 읽기 → 의미 이해 → NumPy로 구현 → 시각화 → "아하!"
```

### 3. **Just-Enough Math**
- 정리(Theorem) 증명: ❌ Skip
- AI에서의 활용: ✅ 집중
- 예: Eigenvalue 이론 전부 ❌ → PCA에 어떻게 쓰이나 ✅

### 4. **시각화 우선**
- 모든 개념을 그림으로
- Matplotlib으로 직접 그려보기
- 3Blue1Brown 스타일

---

## 📖 각 주제별 AI 연결

### 선형대수 → AI
| 수학 개념 | AI 응용 |
|----------|---------|
| Matrix 곱셈 | Attention, Linear layers |
| Dot product | 유사도 계산, Query·Key |
| Transpose | Attention scores (Q·Kᵀ) |
| Eigenvalues | PCA, Stability 분석 |
| SVD | LoRA, 압축, 추천 시스템 |

### 미적분 → AI
| 수학 개념 | AI 응용 |
|----------|---------|
| Derivative | Gradient (어느 방향으로?) |
| Partial derivative | Parameter별 gradient |
| Chain rule | Backpropagation ⭐ |
| Gradient | ∇L = 전체 loss의 변화 방향 |

### 확률/통계 → AI
| 수학 개념 | AI 응용 |
|----------|---------|
| Gaussian | 가중치 초기화, Noise |
| Bernoulli | Dropout |
| Expectation | Loss function의 평균 |
| Bayes' theorem | 사후 확률, Inference |

### 정보이론 → AI
| 수학 개념 | AI 응용 |
|----------|---------|
| Entropy | 불확실성 측정 |
| Cross Entropy | Classification loss |
| KL Divergence | 분포 차이 (VAE, Diffusion) |

---

## 🛠️ 실습 구조

각 문서는 다음 패턴을 따릅니다:

### 1️⃣ 개념 (Why?)
왜 이 개념이 필요한가?

### 2️⃣ 수식 (What?)
핵심 수식과 의미

### 3️⃣ 직관 (Aha!)
비유와 시각화로 이해

### 4️⃣ AI 연결 (Where?)
실제 AI 모델에서 어디에 쓰이나?

### 5️⃣ 코드 (How?)
NumPy로 직접 구현

### 6️⃣ 연습 문제
손으로 계산 → 코드로 검증

---

## ✅ 완료 기준

### 최소 목표 (필수)
- [ ] Matrix 곱셈을 손으로 계산 가능
- [ ] Chain rule로 간단한 함수 미분 가능
- [ ] Gradient descent 1 step을 손으로 계산 가능
- [ ] Cross entropy loss 공식 이해
- [ ] Softmax의 수학적 의미 설명 가능

### 중간 목표 (권장)
- [ ] 2-layer Neural Network의 backprop 유도
- [ ] Attention 수식을 행렬로 표현
- [ ] KL divergence를 코드로 구현
- [ ] Adam optimizer 수식 이해

### 최종 목표 (이상적)
- [ ] Transformer 논문의 모든 수식 이해
- [ ] 새로운 loss function 설계 가능
- [ ] 논문의 수학적 증명 따라가기

---

## 📊 난이도 가이드

### 당신의 수학 레벨은?

**레벨 0: 고등학교 수학도 기억 안 남**
→ 이 Phase를 **3주**로 늘리세요. 천천히!

**레벨 1: 대학 1학년 수학 (미적분, 선형대수) 배웠음**
→ 이 Phase는 **1-2주** 충분합니다. 복습 중심!

**레벨 2: 공업수학까지 마스터**
→ **3-4일**로 빠르게 리뷰하고 Phase 0으로!

**레벨 3: 수학 전공/대학원**
→ 이 Phase는 **Skip** 가능. 하지만 "AI 연결" 섹션은 읽어보세요!

---

## 🎯 학습 전략

### Do's ✅
- ✅ **손으로 계산하기**: 컴퓨터가 아닌 종이에!
- ✅ **시각화하기**: 모든 개념을 그림으로
- ✅ **AI와 연결하기**: "이게 어디에 쓰이지?"
- ✅ **코드로 검증하기**: 손 계산 → NumPy로 확인
- ✅ **이해 안 되면 넘어가기**: 80% 이해면 OK

### Don't's ❌
- ❌ 완벽한 증명 요구하지 마세요
- ❌ 모든 정리를 외우려 하지 마세요
- ❌ 수학 교과서 처음부터 읽지 마세요
- ❌ 이해 없이 공식만 암기하지 마세요

---

## 📚 참고 자료

### 필수
- **3Blue1Brown**: Essence of Linear Algebra (YouTube)
- **3Blue1Brown**: Essence of Calculus (YouTube)
- **StatQuest**: 확률/통계 직관적 설명 (YouTube)

### 추천
- **Mathematics for Machine Learning** (책, 무료 PDF)
- **Deep Learning Book** - Chapter 2-4 (Goodfellow)
- **Khan Academy**: 수학 복습

### 온라인 도구
- **Desmos**: 함수 그래프
- **GeoGebra**: 선형대수 시각화
- **Wolfram Alpha**: 수식 계산

---

## 🚀 시작 준비됐나요?

### 빠른 자가 진단 (2분)

다음을 **지금 당장** 할 수 있나요?

1. 행렬 곱셈: $\begin{bmatrix}1&2\\3&4\end{bmatrix} \times \begin{bmatrix}5\\6\end{bmatrix} = ?$

2. 편미분: $f(x,y) = x^2 + 3xy$일 때, $\frac{\partial f}{\partial x} = ?$

3. 확률: 동전 3번 던져서 앞면 2번 나올 확률은?

**모두 30초 내 답 나옴**: Phase 0으로 바로 진행!
**1-2개 막힘**: 이 Phase를 1주로 빠르게!
**전부 막힘**: 이 Phase를 2-3주로 천천히!

---

## ⏭️ 다음 단계

Day 1부터 시작하세요:

👉 [Day 1-2: 선형대수 기초](./01-linear-algebra-basics.md)

**"수학은 AI의 언어입니다. 이 언어를 배우면, AI의 모든 것이 명확해집니다."**

Let's master the fundamentals! 💪
