# Day 7: Optimization - Training Neural Networks

## 🎯 목표

**최적화 알고리즘의 수학적 원리를 이해하고 Optimizer를 직접 구현하기!**

```python
# AI Training = Optimization Problem
θ* = argmin L(θ)
     θ

# 목표: Loss L을 최소화하는 파라미터 θ 찾기
```

---

## 📖 1. Gradient Descent의 수학

### 직관

**"언덕을 내려가는 가장 빠른 방법은? 가장 가파른 방향으로!"**

```python
# Gradient = 가장 가파른 올라가는 방향
# -Gradient = 가장 가파른 내려가는 방향!

θ_new = θ_old - α * ∇L(θ)
```

### 1차 Taylor 근사

```python
# Loss를 현재 위치에서 1차 근사:
L(θ + Δθ) ≈ L(θ) + ∇L(θ)ᵀ · Δθ

# Gradient 반대 방향으로 이동:
Δθ = -α * ∇L(θ)

# → Loss 감소 보장 (α가 충분히 작으면)!
```

### 구현

```python
import numpy as np

def gradient_descent(loss_fn, grad_fn, theta_init, learning_rate=0.01, num_steps=100):
    """
    Basic gradient descent

    Args:
        loss_fn: Loss function L(θ)
        grad_fn: Gradient function ∇L(θ)
        theta_init: Initial parameters
        learning_rate: Step size α
    """
    theta = theta_init.copy()
    history = {'theta': [theta.copy()], 'loss': []}

    for step in range(num_steps):
        # Compute gradient
        grad = grad_fn(theta)

        # Update parameters
        theta = theta - learning_rate * grad

        # Record
        loss = loss_fn(theta)
        history['theta'].append(theta.copy())
        history['loss'].append(loss)

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")

    return theta, history

# 예시: f(x,y) = x^2 + 2y^2 최소화
def loss(theta):
    x, y = theta
    return x**2 + 2*y**2

def grad(theta):
    x, y = theta
    return np.array([2*x, 4*y])

theta_init = np.array([5.0, 3.0])
theta_opt, history = gradient_descent(loss, grad, theta_init, learning_rate=0.1, num_steps=50)

print(f"Optimal θ: {theta_opt}")  # Should be close to [0, 0]
```

---

## 🎯 2. Stochastic Gradient Descent (SGD)

### 문제: Batch GD는 느리다

```python
# Full Batch:
∇L(θ) = (1/N) Σ_{i=1}^N ∇L_i(θ)  # N개 전체 평균

# 문제: N=1M이면 매 step마다 1M개 계산!
```

### 해결: Stochastic Gradient

```python
# Mini-batch (batch_size=32):
∇L(θ) ≈ (1/32) Σ_{i∈batch} ∇L_i(θ)  # 32개만!

# 장점:
- 빠름: 32배 빠른 step
- Noise가 local minima 탈출에 도움
- 메모리 효율적
```

### 구현

```python
def sgd(model, data_loader, loss_fn, learning_rate=0.01, epochs=10):
    """
    Stochastic Gradient Descent
    """
    for epoch in range(epochs):
        total_loss = 0

        for batch in data_loader:
            # Forward pass
            x_batch, y_batch = batch
            predictions = model(x_batch)

            # Compute loss
            loss = loss_fn(predictions, y_batch)

            # Backward pass (gradient)
            grad = compute_gradient(loss)

            # Update parameters
            for param in model.parameters():
                param.data -= learning_rate * param.grad

            total_loss += loss.item()

        print(f"Epoch {epoch}: Avg Loss = {total_loss / len(data_loader):.4f}")
```

---

## 🚀 3. Momentum

### 문제: SGD oscillates (진동)

```python
# SGD는 noisy gradient 때문에 zigzag
# → 수렴 느림
```

### 해결: Momentum (관성)

```python
# Momentum: 이전 방향을 기억!
v_t = β * v_{t-1} + ∇L(θ_t)
θ_{t+1} = θ_t - α * v_t

# β = 0.9: 이전 gradient의 90% 유지
# → 안정적인 방향으로 가속!
```

### 직관

**"공을 언덕에서 굴릴 때, 관성으로 작은 언덕은 넘어감!"**

### 구현

```python
class MomentumOptimizer:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum

        # Initialize velocity
        self.velocity = {id(p): np.zeros_like(p) for p in params}

    def step(self):
        for param in self.params:
            # Get gradient
            grad = param.grad

            # Update velocity
            v = self.velocity[id(param)]
            v = self.momentum * v + grad
            self.velocity[id(param)] = v

            # Update parameter
            param.data -= self.lr * v

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data)

# 사용
optimizer = MomentumOptimizer(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(num_epochs):
    for batch in data_loader:
        loss = compute_loss(batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
```

---

## 🧠 4. Adam (Adaptive Moment Estimation)

### 핵심 아이디어

**"각 파라미터마다 다른 learning rate!"**

```python
# Adam = Momentum + RMSProp

# 1. First moment (momentum)
m_t = β_1 * m_{t-1} + (1-β_1) * ∇L(θ_t)

# 2. Second moment (adaptive LR)
v_t = β_2 * v_{t-1} + (1-β_2) * (∇L(θ_t))^2

# 3. Bias correction
m̂_t = m_t / (1 - β_1^t)
v̂_t = v_t / (1 - β_2^t)

# 4. Update
θ_{t+1} = θ_t - α * m̂_t / (√v̂_t + ε)
```

### 왜 효과적인가?

```python
# Gradient가 큰 파라미터: √v̂_t 큼 → 작은 update
# Gradient가 작은 파라미터: √v̂_t 작음 → 큰 update

# → 각 파라미터가 적절한 속도로 학습!
```

### 구현

```python
class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # Initialize moments
        self.m = {id(p): np.zeros_like(p) for p in params}
        self.v = {id(p): np.zeros_like(p) for p in params}

    def step(self):
        self.t += 1

        for param in self.params:
            grad = param.grad

            # Update biased first moment
            m = self.m[id(param)]
            m = self.beta1 * m + (1 - self.beta1) * grad
            self.m[id(param)] = m

            # Update biased second moment
            v = self.v[id(param)]
            v = self.beta2 * v + (1 - self.beta2) * grad**2
            self.v[id(param)] = v

            # Bias correction
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            # Update parameter
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data)

# 사용 (PyTorch 스타일)
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch in data_loader:
        optimizer.zero_grad()

        loss = compute_loss(batch)
        loss.backward()

        optimizer.step()
```

---

## 🎓 5. Learning Rate Scheduling

### 왜 필요한가?

```python
# 초반: 큰 LR로 빠르게 이동
# 후반: 작은 LR로 정밀하게 조정

# Fixed LR: 수렴 어려움
# Adaptive LR: 더 좋은 최적점 도달!
```

### Step Decay

```python
lr_t = lr_0 * (γ ** (epoch // step_size))

# 예: lr_0=0.1, γ=0.1, step_size=30
# Epoch 0-29: lr=0.1
# Epoch 30-59: lr=0.01
# Epoch 60-89: lr=0.001
```

### Cosine Annealing

```python
lr_t = lr_min + (lr_max - lr_min) * (1 + cos(π * t / T)) / 2

# 부드럽게 감소!
```

### Warmup

```python
# 초반에 LR 천천히 증가
# → 초기 불안정성 방지

if t < warmup_steps:
    lr_t = lr_max * (t / warmup_steps)
else:
    lr_t = lr_max * decay(t)

# Transformer에서 필수!
```

### 구현

```python
class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.t = 0

    def step(self):
        self.t += 1
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + np.cos(np.pi * self.t / self.T_max)) / 2
        self.optimizer.lr = lr

# 사용
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(num_epochs):
    train_epoch(model, optimizer)
    scheduler.step()
```

---

## 📊 6. Optimizer 비교

| Optimizer | 장점 | 단점 | 사용 |
|-----------|------|------|------|
| SGD | 간단, 안정적 | 느림 | 드물게 |
| SGD + Momentum | 빠름, 안정적 | 하이퍼파라미터 | 때때로 |
| Adam | 빠름, adaptive | 메모리 많음 | 기본 선택! |
| AdamW | Adam + weight decay | - | LLM 훈련 |

### 실전 팁

```python
# 기본 선택
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 큰 모델 (Transformer, LLM)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)
```

---

## 🎓 학습 목표

- [ ] Gradient descent의 수학적 원리 이해
- [ ] SGD의 필요성 이해
- [ ] Momentum의 작동 원리 이해
- [ ] Adam의 수식 이해
- [ ] Learning rate scheduling의 효과 이해
- [ ] Optimizer를 밑바닥부터 구현

---

## 💡 직관적 이해

### Optimization Landscape

```python
# Convex (쉬움):
#     *
#    / \
#   /   \
#  /  θ* \
# --------
# → 하나의 global minimum

# Non-convex (어려움):
#   *       *   *
#  / \  *  / \ / \
# /   \/  \/   V  \
# ----------------
# → Local minima, saddle points

# Neural Networks = Non-convex!
```

### 왜 Adam이 작동하는가?

```python
# Gradient가 일관된 방향: momentum 누적 → 빠르게 이동
# Gradient가 진동하는 방향: v̂ 커짐 → 작게 이동

# → 안정적이면서도 빠른 수렴!
```

---

## 📚 참고 자료

- **An Overview of Gradient Descent Optimization Algorithms** (Ruder, 2016)
- **Adam: A Method for Stochastic Optimization** (Kingma & Ba, 2015)

---

## ⏭️ 다음

👉 [Day 8-9: Probability & Statistics](./05-probability-statistics.md)

**이제 확률과 통계로 불확실성을 다루는 방법을 배워봅시다!** 🎲
