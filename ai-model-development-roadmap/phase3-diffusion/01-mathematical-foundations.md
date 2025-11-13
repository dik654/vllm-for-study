# Week 5: Diffusion Models - Mathematical Foundations

## 🎯 목표

**Diffusion Models의 수학적 원리를 완벽히 이해하기!**

```python
# Forward: 이미지에 노이즈 점진적으로 추가
x_0 (clean) → x_1 → x_2 → ... → x_T (pure noise)

# Reverse: 노이즈에서 이미지 복원 (학습 필요!)
x_T (noise) → x_{T-1} → ... → x_1 → x_0 (clean) ✨
```

---

## 📖 1. 핵심 아이디어

### GANs vs Diffusion

**GANs**: 한 번에 생성
```python
z (noise) → Generator → x (image)
# 문제: 불안정한 훈련, mode collapse
```

**Diffusion**: 점진적으로 생성
```python
x_T → ... → x_t → x_{t-1} → ... → x_0
# 장점: 안정적 훈련, 고품질
```

---

## 🔬 2. Forward Diffusion Process

### 정의

**매 스텝마다 Gaussian noise를 조금씩 추가**

```python
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

# β_t: noise schedule (0.0001 → 0.02)
# √(1-β_t): scaling factor
```

**직관**:
- t가 증가하면 noise 비중 ↑, 이미지 비중 ↓
- T가 충분히 크면 x_T ≈ N(0, I) (pure noise)

### Reparameterization Trick

```python
# 샘플링을 미분 가능하게!
x_t = √(1-β_t) x_{t-1} + √β_t ε    where ε ~ N(0, I)

# 재귀적으로 전개:
α_t = 1 - β_t
ᾱ_t = ∏_{s=1}^t α_s

# 결과: x_0에서 x_t를 직접 샘플링!
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)

x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
```

### 구현

```python
import torch
import torch.nn as nn
import numpy as np

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # Precompute useful values
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # For sampling x_t from x_0
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    def add_noise(self, x_0, t, noise=None):
        """
        Forward diffusion: x_0 → x_t

        x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get coefficients for timestep t
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        # Add noise
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

# 사용 예시
scheduler = NoiseScheduler(num_timesteps=1000)

x_0 = torch.randn(1, 3, 64, 64)  # Clean image
t = torch.tensor([500])          # Timestep

x_t, noise = scheduler.add_noise(x_0, t)

print(f"x_0 shape: {x_0.shape}")
print(f"x_t shape: {x_t.shape}")
print(f"Added noise shape: {noise.shape}")
```

### Noise Schedule 종류

**1. Linear Schedule** (DDPM 원본)
```python
beta_t = linear_interpolate(0.0001, 0.02, t/T)
```

**2. Cosine Schedule** (더 좋음!)
```python
def cosine_schedule(t, T, s=0.008):
    f_t = cos((t/T + s) / (1 + s) * π/2)²
    alpha_bar_t = f_t / f_0
    return alpha_bar_t

# 장점: 초반에 노이즈 천천히, 후반에 빠르게
```

---

## 🔄 3. Reverse Diffusion Process

### 목표

**Noise → Image 복원!**

```python
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

# μ_θ, Σ_θ: 학습할 신경망!
```

### Bayes' Rule

```python
# 우리가 알고 있는 것:
q(x_t | x_{t-1}) = N(√α_t x_{t-1}, β_t I)  # Forward
q(x_{t-1} | x_t, x_0) = N(μ̃_t, β̃_t I)     # Posterior (x_0 조건부)

# Posterior mean (closed form!):
μ̃_t(x_t, x_0) = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) x_0 + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t

# Posterior variance:
β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) β_t
```

### Noise Prediction

**핵심 아이디어**: x_0를 직접 예측하는 대신 **noise를 예측!**

```python
# x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε

# Rearrange:
x_0 = (x_t - √(1-ᾱ_t) ε) / √ᾱ_t

# x_0를 μ̃_t 공식에 대입:
μ_θ(x_t, t) = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t, t))

# ε_θ: 신경망이 예측한 noise!
```

### Sampling (Reverse Process)

```python
def p_sample(model, x_t, t, scheduler):
    """
    Single reverse step: x_t → x_{t-1}
    """
    # Predict noise
    epsilon_pred = model(x_t, t)

    # Get coefficients
    alpha = scheduler.alphas[t]
    alpha_bar = scheduler.alpha_bars[t]
    beta = scheduler.betas[t]

    # Compute mean
    coef1 = 1 / torch.sqrt(alpha)
    coef2 = beta / torch.sqrt(1 - alpha_bar)
    mean = coef1 * (x_t - coef2 * epsilon_pred)

    # Add noise (except at t=0)
    if t > 0:
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta)
        x_t_minus_1 = mean + sigma * noise
    else:
        x_t_minus_1 = mean

    return x_t_minus_1

@torch.no_grad()
def sample(model, shape, num_timesteps, scheduler):
    """
    Complete sampling: x_T → x_0
    """
    # Start from pure noise
    x = torch.randn(shape)

    # Reverse diffusion
    for t in reversed(range(num_timesteps)):
        t_tensor = torch.full((shape[0],), t, dtype=torch.long)
        x = p_sample(model, x, t_tensor, scheduler)

    return x

# 사용
model = DiffusionModel()  # 다음 섹션에서 구현
images = sample(model, shape=(16, 3, 64, 64), num_timesteps=1000, scheduler=scheduler)
```

---

## 📊 4. Training Objective

### Variational Lower Bound (ELBO)

```python
# 목표: log p_θ(x_0) 최대화

# Variational lower bound:
L = E_q[log p_θ(x_0|x_1)] - ∑_{t=2}^T KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t)) - KL(q(x_T|x_0) || p(x_T))

# 복잡함! 😱
```

### Simplified Objective (DDPM)

**핵심 발견**: Noise prediction이 훨씬 간단!

```python
L_simple(θ) = E_{t, x_0, ε} [||ε - ε_θ(x_t, t)||²]

# ε: 실제 noise
# ε_θ: 모델이 예측한 noise
# x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
```

**이게 끝!** 단순한 MSE loss!

### Training Loop

```python
def train_step(model, x_0, scheduler, optimizer):
    """
    Single training step
    """
    batch_size = x_0.size(0)

    # 1. Random timestep
    t = torch.randint(0, scheduler.num_timesteps, (batch_size,))

    # 2. Sample noise
    noise = torch.randn_like(x_0)

    # 3. Add noise to x_0 → x_t
    x_t, _ = scheduler.add_noise(x_0, t, noise)

    # 4. Predict noise
    noise_pred = model(x_t, t)

    # 5. Loss
    loss = nn.functional.mse_loss(noise_pred, noise)

    # 6. Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# 훈련 루프
for epoch in range(num_epochs):
    for x_0 in dataloader:
        loss = train_step(model, x_0, scheduler, optimizer)

    print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

---

## 🎓 학습 목표

- [ ] Forward diffusion process 수식 이해
- [ ] Reparameterization trick으로 x_t 직접 샘플링
- [ ] Reverse process의 목표 이해
- [ ] Noise prediction objective 이해
- [ ] Simplified loss 유도 가능

---

## 🔑 핵심 수식 정리

### Forward Process
```python
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)
x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
```

### Reverse Process
```python
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)
μ_θ(x_t, t) = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t, t))
```

### Training
```python
L_simple = ||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²
```

---

## 💡 직관적 이해

### 왜 Diffusion이 작동하는가?

**1. Forward는 단순함**
- 노이즈 추가는 쉬움 (단순 Gaussian)
- 닫힌 형태의 식 존재

**2. Reverse는 학습 가능**
- 작은 스텝 → 예측 쉬움
- Gaussian 가정 → 간단한 모델

**3. Noise prediction이 효과적**
- x_0 직접 예측보다 쉬움
- Scale-invariant (크기에 무관)

---

## 📚 참고 논문

- **DDPM**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
  - [Paper](https://arxiv.org/abs/2006.11239)
  - Simplified objective 제안

- **Improved DDPM**: Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)
  - [Paper](https://arxiv.org/abs/2102.09672)
  - Cosine schedule, hybrid objective

---

## ⏭️ 다음

👉 [Week 6: DDPM Implementation](./02-ddpm-implementation.md)

**이제 U-Net으로 noise predictor를 구현해봅시다!** 🚀
