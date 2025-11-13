# Day 8-9: Probability & Statistics

## 🎯 목표

**AI의 불확실성을 수학적으로 다루기!**

```python
# AI = Probability!
- Dropout: 확률적으로 neuron 끄기
- Sampling: 확률 분포에서 샘플링
- Gaussian Noise: 정규 분포
- Softmax: 확률로 변환
```

---

## 📖 1. 확률 기초

### 확률 변수 (Random Variable)

```python
# 이산 (Discrete):
X ∈ {0, 1, 2, ...}
P(X = x) = probability

# 연속 (Continuous):
X ∈ ℝ
p(x) = probability density
P(a ≤ X ≤ b) = ∫_a^b p(x) dx
```

### 기댓값 (Expectation)

```python
# 평균!
E[X] = Σ x · P(X = x)     # Discrete
E[X] = ∫ x · p(x) dx      # Continuous

# AI 응용: Loss function의 기댓값
E[L(θ)] = (1/N) Σ L_i(θ)
```

### 분산 (Variance)

```python
# 퍼짐 정도
Var(X) = E[(X - E[X])²]
       = E[X²] - (E[X])²

σ = √Var(X)  # Standard deviation

# AI 응용: Gradient 안정성
```

---

## 🎲 2. 주요 확률 분포

### 2.1 Bernoulli Distribution

**동전 던지기 (0 or 1)**

```python
P(X = 1) = p
P(X = 0) = 1 - p

E[X] = p
Var(X) = p(1-p)
```

**AI 응용: Dropout**

```python
# Dropout: 각 neuron을 p 확률로 끄기
mask = np.random.bernoulli(p=0.5, size=hidden_size)
hidden = hidden * mask
```

### 2.2 Categorical Distribution

**주사위 (K개 선택지)**

```python
P(X = k) = p_k, where Σ p_k = 1

# AI 응용: Classification
```

**구현**

```python
import numpy as np

# Softmax → Categorical
logits = np.array([2.0, 1.0, 0.5])
probs = np.exp(logits) / np.sum(np.exp(logits))

# Sample
sample = np.random.choice(len(probs), p=probs)
print(f"Sampled class: {sample}")
```

### 2.3 Gaussian (Normal) Distribution

**가장 중요!**

```python
p(x) = (1/(σ√(2π))) * exp(-(x-μ)² / (2σ²))

# Parameters:
μ = mean (중심)
σ² = variance (퍼짐)

# Notation:
X ~ N(μ, σ²)
```

**중심 극한 정리 (Central Limit Theorem)**

```python
# 많은 독립 변수의 합 → Gaussian!
# → 자연에서 Gaussian이 흔한 이유
```

**AI 응용**

```python
# 1. Weight initialization
weights = np.random.normal(0, 0.01, size=(784, 128))

# 2. Noise injection (Diffusion)
noise = torch.randn_like(image)  # N(0, 1)

# 3. Gaussian prior (VAE)
z ~ N(0, I)
```

### Multivariate Gaussian

```python
# 다변수 정규 분포
X ~ N(μ, Σ)

# μ: mean vector
# Σ: covariance matrix

p(x) = (1/((2π)^(d/2)|Σ|^(1/2))) * exp(-1/2 (x-μ)ᵀΣ⁻¹(x-μ))
```

---

## 🧮 3. Bayes' Theorem

### 공식

```python
P(A|B) = P(B|A) P(A) / P(B)

# P(A|B): Posterior (사후 확률)
# P(B|A): Likelihood (가능도)
# P(A): Prior (사전 확률)
# P(B): Evidence (증거)
```

### AI 응용: Bayesian Inference

```python
# Model을 posterior로!
P(θ|D) = P(D|θ) P(θ) / P(D)

# P(θ|D): 데이터 D를 본 후 θ의 확률
# P(D|θ): θ가 주어졌을 때 D의 확률 (likelihood)
# P(θ): θ의 prior
```

### 예시: Spam Classification

```python
# P(Spam|"free money") = ?

# Bayes:
P(Spam|"free money") = P("free money"|Spam) * P(Spam) / P("free money")

# 구현
def bayes_classifier(word, p_word_spam, p_word_ham, p_spam):
    """
    Naive Bayes Classifier

    Args:
        word: 단어
        p_word_spam: P(word|Spam)
        p_word_ham: P(word|Ham)
        p_spam: P(Spam)
    """
    p_ham = 1 - p_spam

    # Posterior
    p_spam_word = (p_word_spam * p_spam) / \
                  (p_word_spam * p_spam + p_word_ham * p_ham)

    return p_spam_word

# 사용
p_spam = bayes_classifier("free money", p_word_spam=0.8, p_word_ham=0.1, p_spam=0.3)
print(f"P(Spam|'free money') = {p_spam:.3f}")
```

---

## 📈 4. Maximum Likelihood Estimation (MLE)

### 핵심 아이디어

**"데이터를 가장 잘 설명하는 파라미터는?"**

```python
# Likelihood:
L(θ) = P(D|θ) = ∏_{i=1}^N P(x_i|θ)

# MLE:
θ_MLE = argmax L(θ)
        θ

# Log-likelihood (더 편함):
log L(θ) = Σ log P(x_i|θ)

θ_MLE = argmax log L(θ)
        θ
```

### 예시: Gaussian MLE

```python
# Data: x_1, ..., x_N ~ N(μ, σ²)
# Find: μ, σ² that maximize likelihood

# Log-likelihood:
log L(μ, σ²) = Σ log p(x_i | μ, σ²)
             = -N/2 log(2πσ²) - (1/(2σ²)) Σ(x_i - μ)²

# MLE solution:
μ_MLE = (1/N) Σ x_i       # Sample mean!
σ²_MLE = (1/N) Σ(x_i - μ)²  # Sample variance!
```

### AI 연결: Training = MLE!

```python
# Neural network training:
θ* = argmax P(Y|X, θ)
     θ

# = argmin -log P(Y|X, θ)
# = argmin Loss(θ)

# Cross-entropy loss = Negative log-likelihood!
```

---

## 🎯 5. AI에서의 확률

### 5.1 Softmax as Probability

```python
# Logits → Probabilities
z = [z_1, ..., z_K]

softmax(z)_k = exp(z_k) / Σ_j exp(z_j)

# Properties:
# - All positive
# - Sum to 1
# → Valid probability distribution!
```

**구현**

```python
def softmax(logits):
    # Numerically stable
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

# Temperature scaling
def softmax_temperature(logits, temperature=1.0):
    """
    temperature < 1: 더 확실 (peaky)
    temperature > 1: 더 불확실 (smooth)
    """
    logits = logits / temperature
    return softmax(logits)

# 사용
logits = np.array([2.0, 1.0, 0.5])
probs = softmax(logits)
print(f"Probabilities: {probs}")

# Temperature 비교
probs_low = softmax_temperature(logits, temperature=0.5)
probs_high = softmax_temperature(logits, temperature=2.0)
print(f"Low temp (confident): {probs_low}")
print(f"High temp (uncertain): {probs_high}")
```

### 5.2 Sampling Strategies

**Greedy (Deterministic)**

```python
# 가장 높은 확률 선택
y = argmax P(y|x)
```

**Stochastic (Random)**

```python
# 확률에 따라 샘플링
y ~ P(y|x)

# 구현
probs = model.predict_proba(x)
y = np.random.choice(len(probs), p=probs)
```

**Top-k Sampling**

```python
# 상위 k개 중에서만 샘플링
def top_k_sampling(probs, k=5):
    topk_indices = np.argsort(probs)[-k:]
    topk_probs = probs[topk_indices]
    topk_probs = topk_probs / topk_probs.sum()  # Renormalize

    sample_idx = np.random.choice(topk_indices, p=topk_probs)
    return sample_idx
```

### 5.3 Reparameterization Trick

**문제: Sampling은 미분 불가!**

```python
# z ~ N(μ, σ²)
z = μ + σ * ε, where ε ~ N(0, 1)

# → μ, σ로 미분 가능!
```

**AI 응용: VAE**

```python
# VAE encoder
mu = encoder_mu(x)
log_var = encoder_logvar(x)
sigma = torch.exp(0.5 * log_var)

# Reparameterization
epsilon = torch.randn_like(sigma)
z = mu + sigma * epsilon  # Differentiable!

# Decoder
x_recon = decoder(z)
```

---

## 🎓 학습 목표

- [ ] 기댓값과 분산 계산
- [ ] Gaussian 분포 이해
- [ ] Bayes' theorem 적용
- [ ] MLE 이해
- [ ] Softmax를 확률로 이해
- [ ] Sampling 전략 구현

---

## 💡 직관적 이해

### 왜 Gaussian인가?

```python
# 1. Central Limit Theorem
#    많은 독립 변수의 합 → Gaussian

# 2. Maximum Entropy
#    평균과 분산만 알 때, 가장 "불확실한" 분포 = Gaussian

# 3. 계산 편함!
#    Closed-form solutions 많음
```

### MLE vs MAP

```python
# MLE (Maximum Likelihood):
θ_MLE = argmax P(D|θ)

# MAP (Maximum A Posteriori):
θ_MAP = argmax P(θ|D)
      = argmax P(D|θ) P(θ)

# MAP = MLE + Prior (regularization!)
```

---

## 📚 실전 예제: Dropout

```python
# Dropout = Bernoulli sampling
def dropout(x, p=0.5, training=True):
    """
    Dropout with probability p

    During training: randomly set neurons to 0
    During inference: scale by (1-p)
    """
    if not training:
        return x

    # Bernoulli mask
    mask = np.random.binomial(1, 1-p, size=x.shape)

    # Scale to maintain expected value
    return x * mask / (1-p)

# 사용
hidden = np.random.randn(128)
hidden_dropped = dropout(hidden, p=0.5, training=True)

print(f"Original mean: {hidden.mean():.3f}")
print(f"Dropped mean: {hidden_dropped.mean():.3f}")
# Should be similar!
```

---

## ⏭️ 다음

👉 [Day 10-11: Information Theory](./06-information-theory.md)

**이제 정보이론으로 Loss function의 수학을 이해해봅시다!** 📊
