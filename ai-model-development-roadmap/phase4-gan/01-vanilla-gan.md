# Week 8: Vanilla GAN & Training Stabilization

## 🎯 목표

**GAN의 기초와 훈련 안정화 기법 완벽 마스터!**

```python
# GAN = Generative Adversarial Networks
# 핵심: Generator vs Discriminator (적대적 학습!)

Generator: noise → fake image
Discriminator: real or fake?

# 서로 경쟁하며 발전!
```

---

## 📖 1. GAN 기초

### Minimax Game

**두 플레이어 게임**:
- **Generator (G)**: D를 속이려고 함
- **Discriminator (D)**: Real과 Fake를 구분

```python
# Objective:
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]

# D의 목표: V를 최대화 (잘 분류)
# G의 목표: V를 최소화 (D를 속임)
```

### 구현

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Noise → Image
    """
    def __init__(self, z_dim=100, img_channels=1, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, img_channels * 28 * 28),
            nn.Tanh()  # [-1, 1]
        )

    def forward(self, z):
        # z: (batch, z_dim)
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    """
    Image → Real or Fake
    """
    def __init__(self, img_channels=1, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_channels * 28 * 28, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # [0, 1]
        )

    def forward(self, x):
        # x: (batch, channels, H, W)
        return self.net(x)

# 모델 생성
G = Generator(z_dim=100).cuda()
D = Discriminator().cuda()

print(f"Generator params: {sum(p.numel() for p in G.parameters()):,}")
print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")
```

---

## 🎭 2. Training GAN

### Minimax Loss

```python
def train_gan_minimax(G, D, real_images, z_dim=100):
    """
    Original GAN loss (minimax)
    """
    batch_size = real_images.size(0)

    # Labels
    real_labels = torch.ones(batch_size, 1).cuda()
    fake_labels = torch.zeros(batch_size, 1).cuda()

    # ============== Train Discriminator ==============
    # Real images
    real_output = D(real_images)
    d_loss_real = nn.functional.binary_cross_entropy(real_output, real_labels)

    # Fake images
    z = torch.randn(batch_size, z_dim).cuda()
    fake_images = G(z).detach()  # Detach! (G 업데이트 안함)
    fake_output = D(fake_images)
    d_loss_fake = nn.functional.binary_cross_entropy(fake_output, fake_labels)

    # Total D loss
    d_loss = d_loss_real + d_loss_fake

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # ============== Train Generator ==============
    z = torch.randn(batch_size, z_dim).cuda()
    fake_images = G(z)
    fake_output = D(fake_images)

    # G wants D(G(z)) → 1
    g_loss = nn.functional.binary_cross_entropy(fake_output, real_labels)

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    return d_loss.item(), g_loss.item()
```

### Non-Saturating Loss

**문제**: Minimax loss의 vanishing gradient

```python
# Minimax: G wants log(1 - D(G(z))) → 0
# → Early training: D(G(z)) ≈ 0, gradient 약함!

# Non-saturating: G wants log(D(G(z))) → ∞
# → Early training에도 gradient 강함!
```

```python
def train_gan_nonsaturating(G, D, real_images, z_dim=100):
    # ... (D training 동일) ...

    # ============== Train Generator ==============
    z = torch.randn(batch_size, z_dim).cuda()
    fake_images = G(z)
    fake_output = D(fake_images)

    # Non-saturating loss
    g_loss = -torch.log(fake_output + 1e-8).mean()
    # 또는: g_loss = nn.functional.binary_cross_entropy(fake_output, real_labels)

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    return d_loss.item(), g_loss.item()
```

---

## 🔧 3. DCGAN (Deep Convolutional GAN)

### Architecture Guidelines

1. **No fully connected layers** (except first/last)
2. **Use Conv/ConvTranspose** for up/downsampling
3. **BatchNorm** (except G output, D input)
4. **LeakyReLU** in D, **ReLU** in G
5. **Tanh** in G output

```python
class DCGANGenerator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, base_channels=64):
        super().__init__()

        self.net = nn.Sequential(
            # z_dim → 4×4
            nn.ConvTranspose2d(z_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),

            # 4×4 → 8×8
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),

            # 8×8 → 16×16
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),

            # 16×16 → 32×32
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),

            # 32×32 → 64×64
            nn.ConvTranspose2d(base_channels, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        # z: (batch, z_dim, 1, 1)
        return self.net(z)

class DCGANDiscriminator(nn.Module):
    def __init__(self, img_channels=3, base_channels=64):
        super().__init__()

        self.net = nn.Sequential(
            # 64×64 → 32×32
            nn.Conv2d(img_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32×32 → 16×16
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16×16 → 8×8
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8×8 → 4×4
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4×4 → 1×1
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)
```

---

## 💥 4. Training Problems & Solutions

### 4.1 Mode Collapse

**문제**: G가 다양성 없이 몇 개 샘플만 생성

```python
# G가 D를 속이는 하나의 샘플만 반복 생성
# 모든 z → 같은 이미지
```

**해결책**:

1. **Minibatch Discrimination**
```python
# D가 batch 내 다양성을 평가
```

2. **Unrolled GAN**
```python
# G 업데이트 시 D의 미래 상태 고려
```

3. **Feature Matching**
```python
# G가 D의 중간 feature를 매칭
g_loss = ||E[D.features(real)] - E[D.features(fake)]||²
```

### 4.2 Vanishing Gradients

**문제**: D가 너무 강하면 G의 gradient 소실

**해결책**:

1. **Label Smoothing**
```python
# Real labels: 1.0 → 0.9
# Fake labels: 0.0 → 0.1
real_labels = torch.FloatTensor(batch_size, 1).uniform_(0.8, 1.0)
fake_labels = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.2)
```

2. **Instance Noise**
```python
# 입력에 노이즈 추가 (D를 약화)
noise_std = 0.1 * (1 - epoch / total_epochs)  # Decay
real_images_noisy = real_images + torch.randn_like(real_images) * noise_std
```

3. **Two-Timescale Update Rule (TTUR)**
```python
# D와 G의 learning rate 다르게
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.0, 0.9))
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0004, betas=(0.0, 0.9))
```

### 4.3 Training Dynamics Visualization

```python
import matplotlib.pyplot as plt

def visualize_training(d_losses, g_losses, samples_per_epoch):
    """훈련 과정 시각화"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    axes[0].plot(d_losses, label='D loss', alpha=0.7)
    axes[0].plot(g_losses, label='G loss', alpha=0.7)
    axes[0].legend()
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Losses')

    # Generated samples
    grid = torchvision.utils.make_grid(samples_per_epoch[-1], nrow=8, normalize=True)
    axes[1].imshow(grid.permute(1, 2, 0).cpu())
    axes[1].axis('off')
    axes[1].set_title('Generated Samples')

    plt.tight_layout()
    plt.savefig('gan_training.png')
```

---

## 🛡️ 5. Wasserstein GAN (WGAN)

### 문제: JS Divergence

```python
# Original GAN: Minimize JS divergence
# 문제: Real과 Fake 분포가 겹치지 않으면 gradient 소실
```

### 해결: Wasserstein Distance (Earth-Mover Distance)

```python
# Wasserstein distance: 분포를 이동시키는 최소 비용
W(P_real, P_fake) = sup_{||f||_L ≤ 1} E[f(x)] - E[f(G(z))]

# f: 1-Lipschitz 함수 (Discriminator를 critic으로!)
```

### 구현

```python
def train_wgan(G, D, real_images, z_dim=100, lambda_gp=10):
    """
    WGAN with Gradient Penalty
    """
    batch_size = real_images.size(0)

    # ============== Train Critic (D) ==============
    # Real images
    real_output = D(real_images)

    # Fake images
    z = torch.randn(batch_size, z_dim).cuda()
    fake_images = G(z).detach()
    fake_output = D(fake_images)

    # Wasserstein loss
    d_loss = -torch.mean(real_output) + torch.mean(fake_output)

    # Gradient Penalty
    alpha = torch.rand(batch_size, 1, 1, 1).cuda()
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    interpolated_output = D(interpolated)

    gradients = torch.autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_output),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    d_loss_total = d_loss + lambda_gp * gradient_penalty

    d_optimizer.zero_grad()
    d_loss_total.backward()
    d_optimizer.step()

    # ============== Train Generator ==============
    z = torch.randn(batch_size, z_dim).cuda()
    fake_images = G(z)
    fake_output = D(fake_images)

    g_loss = -torch.mean(fake_output)

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    return d_loss_total.item(), g_loss.item()
```

---

## 🎓 학습 목표

- [ ] Generator와 Discriminator 구현
- [ ] Minimax vs Non-saturating loss 이해
- [ ] DCGAN 아키텍처 구현
- [ ] Mode collapse 경험 및 해결
- [ ] Label smoothing, instance noise 적용
- [ ] WGAN-GP 이해 및 구현

---

## 💡 실전 팁

### 1. Hyperparameters

```python
# Learning rate
lr_d = 2e-4  # Discriminator
lr_g = 2e-4  # Generator (또는 4e-4)

# Optimizer
betas = (0.5, 0.999)  # Adam (original: 0.9, 0.999)

# Training ratio
n_critic = 5  # D를 5번 훈련, G를 1번 (WGAN)
```

### 2. Monitoring

```python
# 정기적으로 이미지 생성 (고정된 z)
fixed_z = torch.randn(64, z_dim).cuda()

@torch.no_grad()
def generate_samples(G, fixed_z, epoch):
    G.eval()
    samples = G(fixed_z)
    grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True)
    torchvision.utils.save_image(grid, f'samples_epoch_{epoch}.png')
    G.train()
```

### 3. Debugging

```python
# D가 너무 강함 (D loss → 0)
# → G learning rate 증가 또는 D 약화

# G가 mode collapse
# → Feature matching, minibatch discrimination

# 훈련이 불안정
# → Spectral normalization, WGAN-GP
```

---

## ⏭️ 다음

👉 [Week 9: Progressive GAN](./02-progressive-gan.md)

**이제 점진적으로 고해상도 이미지를 생성해봅시다!** 🚀
