# Day 10-11: Information Theory

## 🎯 목표

**Loss function의 수학적 근거를 완벽히 이해하기!**

```python
# AI의 핵심 Loss functions:
- Cross Entropy Loss = Information Theory!
- KL Divergence = 분포 차이 측정
- Mutual Information = 상관관계 측정
```

---

## 📖 1. Entropy (엔트로피)

### 핵심 아이디어

**"불확실성을 수치화!"**

```python
# 낮은 entropy: 확실함
P = [1.0, 0.0, 0.0]  # "거의 확신"

# 높은 entropy: 불확실함
P = [0.33, 0.33, 0.34]  # "모르겠음"
```

### 수식

```python
H(P) = -Σ P(x) log P(x)

# Properties:
# - H(P) ≥ 0
# - Uniform distribution = maximum entropy
# - Delta distribution = minimum entropy (0)
```

### 직관: "평균 정보량"

```python
# 놀라움 = 정보량
I(x) = -log P(x)

# 확률 높음 → 낮은 놀라움 (정보 적음)
# 확률 낮음 → 높은 놀라움 (정보 많음)

# Entropy = 평균 놀라움!
H(P) = E[I(X)] = E[-log P(X)]
```

### 구현

```python
import numpy as np

def entropy(probs):
    """
    Shannon Entropy

    Args:
        probs: Probability distribution (sum to 1)
    """
    # Remove zeros (log(0) = undefined)
    probs = probs[probs > 0]

    return -np.sum(probs * np.log2(probs))

# 예시
p_certain = np.array([1.0, 0.0, 0.0])
p_uniform = np.array([1/3, 1/3, 1/3])
p_biased = np.array([0.7, 0.2, 0.1])

print(f"Certain: H = {entropy(p_certain):.3f} bits")     # 0
print(f"Uniform: H = {entropy(p_uniform):.3f} bits")     # 1.585 (max)
print(f"Biased:  H = {entropy(p_biased):.3f} bits")     # 1.157
```

---

## 🎯 2. Cross Entropy

### 정의

**"잘못된 분포로 encoding할 때의 평균 비트 수"**

```python
H(P, Q) = -Σ P(x) log Q(x)

# P: True distribution
# Q: Predicted distribution

# H(P, Q) ≥ H(P)
# Equality iff P = Q
```

### AI 연결: Classification Loss!

```python
# True label (one-hot):
y = [0, 1, 0]  # Class 1

# Predicted probabilities:
ŷ = [0.1, 0.7, 0.2]  # Model output

# Cross Entropy Loss:
L = -Σ y_i log ŷ_i
  = -log ŷ_1           # Only the true class contributes!
  = -log 0.7
  = 0.357
```

### 구현

```python
def cross_entropy(y_true, y_pred, epsilon=1e-12):
    """
    Cross entropy loss

    Args:
        y_true: True labels (one-hot or probabilities)
        y_pred: Predicted probabilities
    """
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -np.sum(y_true * np.log(y_pred))

# 예시: Binary classification
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

# Convert to probabilities (2 classes)
y_true_onehot = np.stack([1-y_true, y_true], axis=1)
y_pred_probs = np.stack([1-y_pred, y_pred], axis=1)

loss = sum(cross_entropy(yt, yp) for yt, yp in zip(y_true_onehot, y_pred_probs)) / len(y_true)
print(f"Cross Entropy Loss: {loss:.3f}")
```

### Binary Cross Entropy

```python
# 특수 케이스: 2개 클래스
BCE(y, ŷ) = -[y log ŷ + (1-y) log(1-ŷ)]

# 구현
def binary_cross_entropy(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

# 사용
y = np.array([1, 0, 1, 1, 0])
y_hat = np.array([0.9, 0.1, 0.8, 0.7, 0.3])

loss = binary_cross_entropy(y, y_hat)
print(f"BCE Loss: {loss:.3f}")
```

---

## 📊 3. KL Divergence

### 정의

**"두 분포 사이의 차이"**

```python
KL(P || Q) = Σ P(x) log(P(x) / Q(x))
           = Σ P(x) [log P(x) - log Q(x)]
           = H(P, Q) - H(P)

# Properties:
# - KL(P || Q) ≥ 0
# - KL(P || Q) = 0 iff P = Q
# - NOT symmetric: KL(P||Q) ≠ KL(Q||P)
```

### 직관

```python
# P에서 샘플링했는데, Q로 encoding할 때의 추가 비트 수

# P = Q: 추가 비트 없음 (KL = 0)
# P ≠ Q: 비효율 (KL > 0)
```

### AI 응용

**1. VAE Loss**

```python
# VAE objective:
L_VAE = Reconstruction_loss + β * KL(q(z|x) || p(z))

# KL term = Regularization
# q(z|x): Encoder distribution
# p(z): Prior (usually N(0, I))
```

**2. Distillation (Knowledge Distillation)**

```python
# Student가 Teacher를 모방
L_distill = KL(P_teacher || P_student)
```

**3. RL (Reinforcement Learning)**

```python
# Policy gradient with KL penalty
L = E[rewards] - α * KL(π_new || π_old)
```

### 구현

```python
def kl_divergence(p, q, epsilon=1e-12):
    """
    KL(P || Q)

    Args:
        p: True distribution
        q: Approximating distribution
    """
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    return np.sum(p * np.log(p / q))

# 예시
p = np.array([0.5, 0.3, 0.2])
q = np.array([0.4, 0.4, 0.2])

kl_pq = kl_divergence(p, q)
kl_qp = kl_divergence(q, p)

print(f"KL(P || Q) = {kl_pq:.3f}")
print(f"KL(Q || P) = {kl_qp:.3f}")
print(f"Asymmetric: {kl_pq != kl_qp}")
```

### Gaussian KL Divergence (Closed Form!)

```python
# P = N(μ₁, σ₁²), Q = N(μ₂, σ₂²)

KL(P || Q) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²) / (2σ₂²) - 1/2

# 특수 케이스: Q = N(0, 1)
KL(P || N(0,1)) = -1/2 * (1 + log σ² - μ² - σ²)

# VAE에서 많이 사용!
def kl_normal_standard(mu, log_var):
    """
    KL(N(mu, sigma^2) || N(0, 1))
    """
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

---

## 🔗 4. Mutual Information

### 정의

**"두 변수의 상관관계"**

```python
I(X; Y) = KL(P(X,Y) || P(X)P(Y))
        = H(X) + H(Y) - H(X, Y)

# I(X; Y) = 0: 독립
# I(X; Y) > 0: 상관있음
```

### AI 응용

**1. Feature Selection**

```python
# High I(X; Y): X는 Y를 잘 예측
# → X는 좋은 feature!
```

**2. InfoNCE (Contrastive Learning)**

```python
# Maximize I(image, augmented_image)
# → 같은 이미지의 다른 view를 가깝게!
```

---

## 📐 5. Loss Function 설계

### Classification: Cross Entropy

```python
# Why Cross Entropy?
# = Maximum Likelihood with Categorical distribution!

L = -Σ y_i log ŷ_i

# Gradient:
∂L/∂logits = ŷ - y  # 매우 깔끔!
```

### Regression: MSE

```python
# Why MSE?
# = Maximum Likelihood with Gaussian distribution!

L = (1/2N) Σ (y_i - ŷ_i)²

# Assumes: y ~ N(ŷ, σ²)
```

### VAE: ELBO (Evidence Lower Bound)

```python
# Maximize log p(x):
log p(x) ≥ ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

# Loss = -ELBO:
L_VAE = Reconstruction + KL_regularization
```

---

## 🎓 학습 목표

- [ ] Entropy의 직관 이해
- [ ] Cross entropy가 왜 classification loss인지 이해
- [ ] KL divergence 계산
- [ ] Mutual information 개념 이해
- [ ] Loss function의 정보이론적 해석

---

## 💡 직관적 이해

### Cross Entropy vs MSE

```python
# Classification (discrete):
# → Cross Entropy
# → Categorical/Bernoulli distribution

# Regression (continuous):
# → MSE
# → Gaussian distribution
```

### KL Divergence의 Asymmetry

```python
# KL(P || Q): P로 샘플링, Q로 encoding
# → P가 높고 Q가 낮은 곳 penalty 큼

# KL(Q || P): Q로 샘플링, P로 encoding
# → Q가 높고 P가 낮은 곳 penalty 큼

# Forward KL: Q가 P를 커버
# Reverse KL: Q가 P의 mode 하나만 선택 (mode-seeking)
```

---

## 🔬 실전 예제: Softmax + Cross Entropy

```python
import torch
import torch.nn as nn

# 방법 1: 수동 계산
def manual_softmax_ce(logits, targets):
    """
    Softmax + Cross Entropy (numerically unstable!)
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    loss = -torch.sum(targets * log_probs) / logits.size(0)
    return loss

# 방법 2: LogSoftmax + NLLLoss (stable!)
def stable_softmax_ce(logits, targets):
    """
    Numerically stable version
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = torch.nn.functional.nll_loss(log_probs, targets)
    return loss

# 방법 3: PyTorch built-in (most stable!)
def pytorch_ce(logits, targets):
    """
    CrossEntropyLoss combines LogSoftmax + NLLLoss
    """
    criterion = nn.CrossEntropyLoss()
    return criterion(logits, targets)

# 비교
logits = torch.randn(4, 3)  # (batch=4, classes=3)
targets = torch.tensor([0, 1, 2, 1])

loss1 = manual_softmax_ce(logits, torch.nn.functional.one_hot(targets, 3).float())
loss2 = stable_softmax_ce(logits, targets)
loss3 = pytorch_ce(logits, targets)

print(f"Manual: {loss1:.4f}")
print(f"Stable: {loss2:.4f}")
print(f"PyTorch: {loss3:.4f}")
# All similar!
```

---

## 📚 참고 자료

- **Information Theory, Inference, and Learning Algorithms** (MacKay)
- **Elements of Information Theory** (Cover & Thomas)

---

## ⏭️ 다음

👉 [Day 12-14: Putting It Together](./07-putting-it-together.md)

**이제 모든 수학을 통합하여 Neural Network를 완전히 이해해봅시다!** 🎓
