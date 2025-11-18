# 논문 읽기를 위한 배경지식

## 🎯 목표

**최신 AI 논문을 이해하기 위한 필수 수학/CS 배경지식 정리**

AI 논문은 다음 분야의 지식을 요구합니다:
- 선형대수 (Linear Algebra)
- 확률/통계 (Probability & Statistics)
- 최적화 (Optimization)
- 정보이론 (Information Theory)
- 강화학습 (Reinforcement Learning)
- 그래프 이론 (Graph Theory)

---

## 📐 1. Linear Algebra (선형대수)

### 필수 개념

```python
import numpy as np

# 1. Vector (벡터)
v = np.array([1, 2, 3])

# 내적 (Dot Product)
# 의도: 벡터 간 유사도 측정
a = np.array([1, 2])
b = np.array([3, 4])
dot_product = np.dot(a, b)  # 1*3 + 2*4 = 11

# 노름 (Norm) - 벡터의 크기
# L2 norm: √(x₁² + x₂² + ... + xₙ²)
norm = np.linalg.norm(v)  # √(1 + 4 + 9) = √14

# 코사인 유사도 (Cosine Similarity)
# 의도: Attention, Embedding 비교에 사용
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 2. Matrix (행렬)
A = np.array([[1, 2], [3, 4]])

# 전치 (Transpose)
# 의도: Attention에서 Q·K^T
A_T = A.T

# 행렬 곱셈 (Matrix Multiplication)
# (m×n) × (n×p) = (m×p)
B = np.array([[5, 6], [7, 8]])
C = A @ B  # 또는 np.matmul(A, B)

# 역행렬 (Inverse)
# A · A⁻¹ = I
A_inv = np.linalg.inv(A)


# 3. Eigenvalues & Eigenvectors (고유값, 고유벡터)
# Av = λv
# 의도: PCA, Spectral methods
eigenvalues, eigenvectors = np.linalg.eig(A)


# 4. Singular Value Decomposition (SVD)
# A = U·Σ·V^T
# 의도: LoRA, 차원 축소
U, S, Vt = np.linalg.svd(A)

print(f"U shape: {U.shape}")
print(f"S (singular values): {S}")
print(f"Vt shape: {Vt.shape}")
```

### 논문에서 자주 보는 표기

```
표기법:
- x: 스칼라 (소문자)
- x: 벡터 (굵은 소문자 또는 화살표)
- X: 행렬 (대문자)
- X^T: 전치 행렬
- X⁻¹: 역행렬
- ||x||: 노름
- ⟨x,y⟩: 내적
- ⊗: 크로네커 곱
- ⊙: 요소별 곱 (Hadamard product)

차원 표기:
- x ∈ ℝ^d: x는 d차원 실수 벡터
- X ∈ ℝ^{m×n}: X는 m×n 행렬
- W ∈ ℝ^{d_model × d_ff}: Weight 행렬
```

### Attention 수식 이해

```python
"""
Self-Attention 수식:

Attention(Q, K, V) = softmax(QK^T / √d_k) V

단계별 이해:

1. QK^T ∈ ℝ^{n×n}
   - Q: (n × d_k) - Query 행렬
   - K: (n × d_k) - Key 행렬
   - K^T: (d_k × n) - Key 전치
   - QK^T: (n × n) - Attention scores

   의도: 각 query와 모든 key 간 유사도

2. / √d_k
   의도: Gradient 안정화 (scaling)
   왜? d_k가 크면 내적 값이 커짐 → softmax 기울기 소실

3. softmax(...)
   의도: 확률 분포로 변환 (합=1)
   각 token이 다른 token에 "얼마나 주목"할지

4. ... V
   의도: Value 행렬과 곱하여 가중합
   결과: Attention-weighted representation
"""

def self_attention(Q, K, V):
    """Self-Attention 구현"""
    d_k = Q.shape[-1]

    # Step 1: QK^T
    scores = Q @ K.T  # (n, d_k) @ (d_k, n) = (n, n)

    # Step 2: Scale
    scores = scores / np.sqrt(d_k)

    # Step 3: Softmax
    attention_weights = softmax(scores, axis=-1)  # (n, n)

    # Step 4: Weighted sum of V
    output = attention_weights @ V  # (n, n) @ (n, d_v) = (n, d_v)

    return output, attention_weights


def softmax(x, axis=-1):
    """Numerically stable softmax"""
    # 의도: exp(x) 오버플로우 방지
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

---

## 📊 2. Probability & Statistics (확률/통계)

### 필수 개념

```python
import scipy.stats as stats

# 1. 확률 분포 (Probability Distributions)

# 정규분포 (Normal/Gaussian Distribution)
# N(μ, σ²)
# 의도: Parameter 초기화, Noise 모델링
mu, sigma = 0, 1
x = np.random.normal(mu, sigma, 1000)

# PDF (Probability Density Function)
# 의도: 특정 값의 확률 밀도
pdf_value = stats.norm.pdf(0, mu, sigma)


# 베르누이 분포 (Bernoulli Distribution)
# 의도: 이진 분류
p = 0.7  # 확률
sample = np.random.binomial(1, p, 100)  # 0 또는 1


# 카테고리 분포 (Categorical Distribution)
# 의도: Multi-class 분류
probs = [0.2, 0.3, 0.5]  # 3개 클래스
samples = np.random.choice(3, size=100, p=probs)


# 2. 기댓값 (Expectation)
# E[X] = ∫ x·p(x) dx
# 의도: 평균, 손실 함수
def expectation(values, probabilities):
    """
    E[X] 계산

    의도: 확률적 시나리오의 평균 결과
    """
    return np.sum(values * probabilities)

# 예: 주사위 기댓값
dice_values = np.array([1, 2, 3, 4, 5, 6])
dice_probs = np.ones(6) / 6
E_dice = expectation(dice_values, dice_probs)  # 3.5


# 3. 분산 (Variance)
# Var(X) = E[(X - E[X])²]
# 의도: 불확실성 측정
def variance(values, probabilities):
    """분산 계산"""
    mean = expectation(values, probabilities)
    squared_diff = (values - mean) ** 2
    return expectation(squared_diff, probabilities)


# 4. 조건부 확률 (Conditional Probability)
# P(A|B) = P(A ∩ B) / P(B)
# 의도: Bayes' Theorem, Autoregressive models

# 베이즈 정리 (Bayes' Theorem)
# P(A|B) = P(B|A)·P(A) / P(B)
def bayes_theorem(p_b_given_a, p_a, p_b):
    """
    베이즈 정리

    의도: 사전 확률 → 사후 확률
    예: 질병 진단, Bayesian inference
    """
    return (p_b_given_a * p_a) / p_b

# 예: 질병 테스트
p_disease = 0.01  # 질병 유병률 1%
p_positive_given_disease = 0.95  # 민감도
p_positive = 0.05  # 양성 확률

p_disease_given_positive = bayes_theorem(
    p_positive_given_disease,
    p_disease,
    p_positive
)
print(f"P(질병|양성) = {p_disease_given_positive:.3f}")  # ~0.19


# 5. Maximum Likelihood Estimation (MLE)
"""
최대 우도 추정

목표: 데이터를 가장 잘 설명하는 파라미터 찾기

Likelihood: L(θ|X) = P(X|θ)
Log-Likelihood: log L(θ|X) = Σ log P(x_i|θ)

의도:
- Cross-Entropy Loss의 이론적 근거
- Neural Network 학습 = MLE!
"""

def log_likelihood_normal(data, mu, sigma):
    """정규분포의 log-likelihood"""
    n = len(data)
    ll = -n/2 * np.log(2 * np.pi * sigma**2)
    ll -= np.sum((data - mu)**2) / (2 * sigma**2)
    return ll

# MLE로 mu, sigma 추정
data = np.random.normal(5, 2, 1000)
mu_mle = np.mean(data)  # MLE of mu
sigma_mle = np.std(data)  # MLE of sigma


# 6. Kullback-Leibler (KL) Divergence
"""
KL Divergence: D_KL(P||Q)

의도: 두 확률 분포의 차이 측정

수식:
D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))
           = E_P[log P(x) - log Q(x)]

특징:
- D_KL(P||Q) ≥ 0
- D_KL(P||Q) = 0 ⟺ P = Q
- 비대칭: D_KL(P||Q) ≠ D_KL(Q||P)

사용:
- VAE loss
- Policy gradient (RL)
- Distillation
"""

def kl_divergence(p, q):
    """
    KL Divergence 계산

    의도: P를 Q로 근사할 때의 정보 손실
    """
    # Avoid log(0)
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)

    return np.sum(p * np.log(p / q))

# 예
p = np.array([0.1, 0.2, 0.7])
q = np.array([0.2, 0.3, 0.5])
print(f"D_KL(P||Q) = {kl_divergence(p, q):.4f}")
```

### Cross-Entropy와 MLE의 관계

```python
"""
Cross-Entropy Loss의 이론적 근거

분류 문제:
- True distribution: P(y|x) - 실제 정답
- Model distribution: Q(y|x; θ) - 모델 예측

Cross-Entropy:
H(P, Q) = -Σ P(y) log Q(y)

Multi-class classification:
- P(y) = one-hot encoding [0, 0, 1, 0, ...]
- Q(y) = softmax output [0.1, 0.2, 0.6, 0.1, ...]

Loss = -log Q(y_true)

이것이 바로 Maximum Likelihood Estimation!

MLE:
θ* = argmax_θ Π P(y_i|x_i; θ)
   = argmax_θ Σ log P(y_i|x_i; θ)
   = argmin_θ Σ -log P(y_i|x_i; θ)
   = argmin_θ Cross-Entropy

결론: Cross-Entropy를 최소화 = MLE
"""

# PyTorch 구현
import torch
import torch.nn.functional as F

def cross_entropy_from_scratch(logits, targets):
    """
    Cross-Entropy Loss 직접 구현

    Args:
        logits: (batch, num_classes) - raw scores
        targets: (batch,) - class indices

    의도: Softmax + Negative Log-Likelihood
    """
    # Step 1: Softmax
    # 의도: logits → probabilities
    probs = F.softmax(logits, dim=-1)

    # Step 2: Log probabilities
    log_probs = torch.log(probs + 1e-10)  # 수치 안정성

    # Step 3: Negative log-likelihood
    # 의도: 정답 클래스의 log prob만 선택
    batch_size = targets.size(0)
    nll = -log_probs[range(batch_size), targets]

    # Step 4: 평균
    loss = nll.mean()

    return loss

# 사용 예
logits = torch.randn(32, 10)  # batch=32, classes=10
targets = torch.randint(0, 10, (32,))

loss = cross_entropy_from_scratch(logits, targets)
print(f"Cross-Entropy Loss: {loss.item():.4f}")

# PyTorch 내장 함수와 비교
loss_builtin = F.cross_entropy(logits, targets)
print(f"Built-in: {loss_builtin.item():.4f}")
```

---

## 🎲 3. Information Theory (정보이론)

### 핵심 개념

```python
"""
Information Theory의 기본 개념들

1. Entropy (엔트로피)
   H(X) = -Σ P(x) log P(x)

   의도: 불확실성의 양
   - 낮은 엔트로피: 예측 가능 (예: 동전이 항상 앞면)
   - 높은 엔트로피: 예측 불가능 (예: 공정한 주사위)

2. Cross-Entropy
   H(P, Q) = -Σ P(x) log Q(x)

   의도: P를 Q로 근사할 때의 정보량

3. KL Divergence
   D_KL(P||Q) = H(P, Q) - H(P)

   의도: 추가로 필요한 bits

4. Mutual Information (상호 정보량)
   I(X;Y) = H(X) + H(Y) - H(X,Y)

   의도: X와 Y가 공유하는 정보량
"""

def entropy(probs):
    """
    Entropy 계산

    의도: 확률 분포의 불확실성
    """
    # Log base 2: bits
    # Log base e: nats
    probs = np.clip(probs, 1e-10, 1)
    return -np.sum(probs * np.log2(probs))

# 예: 공정한 동전
fair_coin = np.array([0.5, 0.5])
print(f"Fair coin entropy: {entropy(fair_coin):.2f} bits")  # 1.0

# 예: 불공정한 동전
unfair_coin = np.array([0.9, 0.1])
print(f"Unfair coin entropy: {entropy(unfair_coin):.2f} bits")  # 0.47

# 예: 공정한 주사위
fair_dice = np.ones(6) / 6
print(f"Fair dice entropy: {entropy(fair_dice):.2f} bits")  # 2.58


def mutual_information(joint_prob, marginal_x, marginal_y):
    """
    Mutual Information 계산

    I(X;Y) = Σ_x Σ_y P(x,y) log(P(x,y) / (P(x)P(y)))

    의도: X를 알면 Y에 대한 불확실성이 얼마나 줄어드는가?
    """
    mi = 0
    for i in range(len(marginal_x)):
        for j in range(len(marginal_y)):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log2(
                    joint_prob[i, j] / (marginal_x[i] * marginal_y[j])
                )
    return mi


"""
논문에서의 활용:

1. Language Modeling
   - Perplexity = 2^H(P)
   - 낮은 perplexity = 좋은 모델

2. Attention Mechanism
   - Attention weights의 entropy
   - 낮은 entropy = 집중적 attention
   - 높은 entropy = 분산된 attention

3. Variational Autoencoder (VAE)
   - Loss = Reconstruction + KL(q(z|x)||p(z))
   - KL term: latent space regularization

4. Distillation
   - Student가 Teacher의 분포를 학습
   - KL(Teacher||Student) 최소화
"""
```

---

## 🔧 4. Optimization (최적화)

### Gradient Descent 계열

```python
"""
최적화 알고리즘의 진화:

1. SGD (Stochastic Gradient Descent)
   θ_{t+1} = θ_t - η·∇L(θ_t)

2. Momentum
   v_{t+1} = β·v_t + ∇L(θ_t)
   θ_{t+1} = θ_t - η·v_t

3. Adam (Adaptive Moment Estimation)
   m_t = β1·m_{t-1} + (1-β1)·∇L
   v_t = β2·v_{t-1} + (1-β2)·(∇L)²
   θ_{t+1} = θ_t - η·m_t / (√v_t + ε)
"""

class Optimizer:
    """최적화 알고리즘 구현"""

    @staticmethod
    def sgd(params, grads, lr=0.01):
        """
        Stochastic Gradient Descent

        의도: 가장 기본적인 최적화
        문제: 느린 수렴, local minima
        """
        return params - lr * grads

    @staticmethod
    def momentum(params, grads, velocity, lr=0.01, beta=0.9):
        """
        Momentum

        의도: 관성 추가, 진동 감소
        수식:
        v = β·v + ∇L
        θ = θ - η·v
        """
        velocity = beta * velocity + grads
        params = params - lr * velocity
        return params, velocity

    @staticmethod
    def adam(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Adam Optimizer

        의도:
        - 각 파라미터마다 adaptive learning rate
        - Momentum (1st moment)
        - RMSProp (2nd moment)

        왜 효과적?
        - 학습률 자동 조정
        - Sparse gradient 처리
        - 안정적 수렴
        """
        # 1st moment (momentum)
        m = beta1 * m + (1 - beta1) * grads

        # 2nd moment (squared gradient)
        v = beta2 * v + (1 - beta2) * (grads ** 2)

        # Bias correction
        # 의도: 초기 step에서 m, v가 0에 편향되는 것 방지
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

        return params, m, v


# 시뮬레이션
def rosenbrock(x, y):
    """Rosenbrock function (최적화 테스트)"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def grad_rosenbrock(x, y):
    """Gradient of Rosenbrock"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# Adam으로 최적화
params = np.array([-1.0, 1.0])
m = np.zeros_like(params)
v = np.zeros_like(params)

for t in range(1, 1001):
    grads = grad_rosenbrock(*params)
    params, m, v = Optimizer.adam(params, grads, m, v, t)

    if t % 100 == 0:
        loss = rosenbrock(*params)
        print(f"Step {t}: params={params}, loss={loss:.6f}")

# 최적해: (1, 1)
```

### Learning Rate Scheduling

```python
"""
Learning Rate Schedule

의도: 학습 초기엔 빠르게, 후기엔 세밀하게

1. Step Decay
   η_t = η_0 · γ^(t/k)

2. Exponential Decay
   η_t = η_0 · e^(-λt)

3. Cosine Annealing
   η_t = η_min + (η_max - η_min) · (1 + cos(πt/T)) / 2

4. Warmup + Decay (Transformer)
   η_t = d_model^(-0.5) · min(t^(-0.5), t · warmup^(-1.5))
"""

def cosine_annealing_lr(epoch, total_epochs, lr_max=0.1, lr_min=0.001):
    """
    Cosine Annealing

    의도: 부드러운 learning rate 감소
    """
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(np.pi * epoch / total_epochs)
    )

# 시각화
epochs = np.arange(100)
lrs = [cosine_annealing_lr(e, 100) for e in epochs]

# epoch 0: 0.1
# epoch 50: 0.0505
# epoch 99: 0.001


def warmup_cosine_lr(step, warmup_steps=4000, d_model=512):
    """
    Warmup + Cosine (Transformer 논문)

    의도:
    - Warmup: 초기 불안정성 방지
    - Cosine: 부드러운 감소
    """
    arg1 = step ** (-0.5)
    arg2 = step * (warmup_steps ** (-1.5))

    return d_model ** (-0.5) * min(arg1, arg2)
```

---

## 🎮 5. Reinforcement Learning (강화학습)

### 기본 개념

```python
"""
Reinforcement Learning 핵심 개념

Environment:
- State (s): 현재 상태
- Action (a): 가능한 행동
- Reward (r): 보상
- Next state (s'): 다음 상태

Agent:
- Policy π(a|s): 상태 s에서 행동 a를 선택할 확률
- Value function V(s): 상태 s의 가치
- Q-function Q(s,a): 상태-행동 쌍의 가치

목표: 누적 보상 최대화
G_t = Σ γ^k · r_{t+k+1}
"""

class RLBasics:
    """강화학습 기본 개념 구현"""

    @staticmethod
    def bellman_equation(rewards, gamma=0.99):
        """
        Bellman Equation

        V(s) = R(s) + γ · max_a Σ P(s'|s,a) · V(s')

        의도: 현재 가치 = 즉각 보상 + 미래 가치
        """
        V = np.zeros(len(rewards))

        # Value Iteration
        for _ in range(100):
            V_new = rewards + gamma * V  # 단순화
            if np.allclose(V, V_new):
                break
            V = V_new

        return V

    @staticmethod
    def q_learning_update(Q, s, a, r, s_prime, alpha=0.1, gamma=0.99):
        """
        Q-Learning Update

        Q(s,a) ← Q(s,a) + α·[r + γ·max_a' Q(s',a') - Q(s,a)]

        의도: TD (Temporal Difference) 학습
        """
        td_target = r + gamma * np.max(Q[s_prime])
        td_error = td_target - Q[s, a]
        Q[s, a] += alpha * td_error

        return Q


"""
RLHF (Reinforcement Learning from Human Feedback)

Process:
1. Supervised Fine-Tuning (SFT)
   - 고품질 데이터로 base model 훈련

2. Reward Model Training
   - 인간 선호도 데이터 수집
   - (prompt, response_A, response_B) + 선호도
   - Reward model 훈련: R(prompt, response)

3. RL Fine-Tuning (PPO)
   - Policy: LLM
   - Reward: Reward model
   - 목표: E[R(prompt, LLM(prompt))] 최대화

수식:
max_θ E_{prompt}[R(prompt, π_θ(prompt))]
- KL(π_θ || π_ref)  ← Regularization
"""

def ppo_loss(old_probs, new_probs, advantages, epsilon=0.2):
    """
    Proximal Policy Optimization (PPO) Loss

    의도: Policy를 너무 급격히 바꾸지 않기

    수식:
    L = min(
        ratio · A,
        clip(ratio, 1-ε, 1+ε) · A
    )

    where ratio = π_new / π_old
    """
    ratio = new_probs / (old_probs + 1e-10)

    # Clipped ratio
    clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)

    # Loss
    loss = -np.minimum(
        ratio * advantages,
        clipped_ratio * advantages
    )

    return loss.mean()
```

---

## 📊 6. 논문 읽기 체크리스트

### 논문 구조 이해

```
표준 AI 논문 구조:

1. Abstract (초록)
   ✓ 문제 정의
   ✓ 제안 방법
   ✓ 주요 결과

2. Introduction (서론)
   ✓ 배경
   ✓ 기존 연구의 한계
   ✓ 본 논문의 기여

3. Related Work (관련 연구)
   ✓ 선행 연구 정리
   ✓ 본 논문과의 차이점

4. Method (방법론)
   ✓ 모델 아키텍처
   ✓ 알고리즘
   ✓ 수식 설명
   ← 가장 중요! 여기를 정확히 이해

5. Experiments (실험)
   ✓ 데이터셋
   ✓ 평가 지표
   ✓ Baseline 비교
   ✓ Ablation study

6. Results (결과)
   ✓ 정량적 결과
   ✓ 정성적 분석
   ✓ Failure cases

7. Conclusion (결론)
   ✓ 요약
   ✓ Future work
```

### 효율적 논문 읽기 전략

```
3-Pass 접근법:

Pass 1 (5-10분): 빠른 훑어보기
✓ Title, Abstract, Introduction
✓ Section headings
✓ Figures & Tables
✓ Conclusion
목표: "이 논문이 나와 관련 있나?" 판단

Pass 2 (1시간): 자세히 읽기
✓ 전체 내용 읽기
✓ 수식 이해 시도
✓ 중요한 그림 분석
✓ 관련 논문 마크
목표: "핵심 아이디어가 뭔가?" 파악

Pass 3 (4-5시간): 깊이 이해
✓ 수식 직접 유도
✓ 코드 구현 시도
✓ 실험 재현
✓ 한계점 분석
목표: "내가 이걸 개선할 수 있나?" 판단
```

### 수식 이해 팁

```python
"""
논문 수식 읽기 팁:

1. 차원 확인
   - 각 변수의 shape 파악
   - Matrix multiplication 가능한지 확인

   예: Attention
   Q: (batch, seq, d_k)
   K: (batch, seq, d_k)
   V: (batch, seq, d_v)

   QK^T: (batch, seq, d_k) @ (batch, d_k, seq)
       = (batch, seq, seq) ✓

2. 특수 기호
   ∇: Gradient
   ∂: Partial derivative
   Σ: Summation
   Π: Product
   argmax: 최대값을 주는 인자
   ⊙: Element-wise multiplication
   ⊗: Outer product / Kronecker product

3. 확률 표기
   P(x): x의 확률
   P(x|y): y가 주어졌을 때 x의 확률
   E[X]: Expectation
   Var(X): Variance

4. 최적화
   min_θ L(θ): θ에 대해 L 최소화
   argmin_θ L(θ): L을 최소화하는 θ 찾기
   s.t.: subject to (제약 조건)
"""

# 예제: Attention 수식 구현
def verify_attention_dimensions():
    """
    수식의 차원 확인

    의도: 논문 수식 → 코드로 검증
    """
    batch = 2
    seq_len = 10
    d_k = 64
    d_v = 64

    Q = np.random.randn(batch, seq_len, d_k)
    K = np.random.randn(batch, seq_len, d_k)
    V = np.random.randn(batch, seq_len, d_v)

    # QK^T
    K_T = np.transpose(K, (0, 2, 1))  # (batch, d_k, seq)
    scores = Q @ K_T  # (batch, seq, seq)

    print(f"Q shape: {Q.shape}")
    print(f"K^T shape: {K_T.shape}")
    print(f"QK^T shape: {scores.shape}")  # (2, 10, 10) ✓

    # Softmax
    attn = softmax(scores / np.sqrt(d_k), axis=-1)

    # AttnV
    output = attn @ V  # (batch, seq, seq) @ (batch, seq, d_v)
    print(f"Output shape: {output.shape}")  # (2, 10, 64) ✓

verify_attention_dimensions()
```

---

## 📚 추천 학습 순서

```
1단계: 수학 기초 (2-4주)
┌─────────────────────────────┐
│ □ Linear Algebra            │
│   - 벡터, 행렬 연산         │
│   - Eigenvalue, SVD         │
│ □ Probability               │
│   - 확률 분포               │
│   - Bayes' Theorem          │
│ □ Calculus                  │
│   - Gradient                │
│   - Chain rule              │
└─────────────────────────────┘

2단계: ML 기초 (4-6주)
┌─────────────────────────────┐
│ □ Optimization              │
│   - SGD, Adam               │
│   - Learning rate schedule  │
│ □ Loss Functions            │
│   - Cross-Entropy           │
│   - MSE, MAE                │
│ □ Regularization            │
│   - L1, L2                  │
│   - Dropout                 │
└─────────────────────────────┘

3단계: DL 기초 (4-6주)
┌─────────────────────────────┐
│ □ Neural Networks           │
│   - Backpropagation         │
│   - Activation functions    │
│ □ CNN                       │
│   - Convolution             │
│   - Pooling                 │
│ □ RNN/LSTM                  │
│   - Sequential modeling     │
│   - Vanishing gradient      │
└─────────────────────────────┘

4단계: Transformer (6-8주)
┌─────────────────────────────┐
│ □ Attention                 │
│   - Self-Attention          │
│   - Multi-Head Attention    │
│ □ Position Encoding         │
│   - Sinusoidal              │
│   - Learned                 │
│ □ Transformer Architecture  │
│   - Encoder-Decoder         │
│   - Decoder-only (GPT)      │
└─────────────────────────────┘

5단계: 고급 주제 (지속적)
┌─────────────────────────────┐
│ □ RL (RLHF)                 │
│ □ Diffusion Models          │
│ □ Multi-modal               │
│ □ Efficient Transformers    │
└─────────────────────────────┘
```

---

## 🔗 리소스

### 교과서
```
수학:
1. "Mathematics for Machine Learning" (Deisenroth et al.)
2. "Deep Learning" (Goodfellow et al.) - Part I

확률/통계:
1. "Pattern Recognition and Machine Learning" (Bishop)
2. "Probabilistic Machine Learning" (Murphy)

최적화:
1. "Convex Optimization" (Boyd & Vandenberghe)

강화학습:
1. "Reinforcement Learning" (Sutton & Barto)
```

### 온라인 강의
```
1. 3Blue1Brown (YouTube) - 시각적 수학
2. Stanford CS229 - Machine Learning
3. Fast.ai - Practical Deep Learning
4. Andrej Karpathy - Neural Networks
```

---

## ⏭️ Next Steps

배경지식을 마스터했다면:

1. **Transformer 논문 읽기**: "Attention Is All You Need"
2. **최신 연구 트렌드**: phase6-current-trends/02-research-trends-2024.md
3. **논문 구현**: GitHub에서 구현 찾기

👉 Continue to **02-research-trends-2024-2025.md**

**논문 읽기 배경지식 완료!** 🎓
