# Week 9: Progressive GAN - Growing to High Resolution

## 🎯 목표

**점진적으로 해상도를 늘려 고품질 이미지 생성!**

```python
# Progressive Growing:
4×4 → 8×8 → 16×16 → 32×32 → 64×64 → ... → 1024×1024

# 장점:
- 안정적 훈련
- 빠른 학습
- 고품질 (1024×1024!)
```

---

## 📖 1. Progressive Growing

### 핵심 아이디어

**작은 해상도부터 시작하여 점진적으로 layer 추가!**

```
Stage 1: 4×4 훈련
Stage 2: 8×8 layer 추가, 계속 훈련
Stage 3: 16×16 layer 추가
...
Stage 9: 1024×1024
```

**왜 효과적인가?**
1. 저해상도: 큰 구조 학습 (얼굴 위치, 전체 형태)
2. 고해상도: 세부사항 추가 (머리카락, 피부 질감)
3. 점진적 → 안정적!

### Smooth Fade-in

**새 layer를 갑자기 추가하면 불안정!**

```python
# α: 0 → 1로 천천히 증가
output = (1 - α) * low_res_upsampled + α * high_res

# α = 0: 이전 해상도만 사용
# α = 1: 새 해상도 완전히 사용
```

### 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProgressiveGenerator(nn.Module):
    """
    Progressive GAN Generator
    """

    def __init__(self, z_dim=512, base_channels=512):
        super().__init__()

        # Initial 4×4
        self.initial = nn.Sequential(
            # (z_dim,) → (base_channels, 4, 4)
            nn.ConvTranspose2d(z_dim, base_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )

        # Progressive blocks: 4→8→16→32→64→128→256→512→1024
        self.blocks = nn.ModuleList([
            self._make_block(base_channels, base_channels),      # 8×8
            self._make_block(base_channels, base_channels // 2), # 16×16
            self._make_block(base_channels // 2, base_channels // 4), # 32×32
            self._make_block(base_channels // 4, base_channels // 8), # 64×64
            self._make_block(base_channels // 8, base_channels // 16), # 128×128
            self._make_block(base_channels // 16, base_channels // 32), # 256×256
            self._make_block(base_channels // 32, base_channels // 32), # 512×512
            self._make_block(base_channels // 32, base_channels // 32), # 1024×1024
        ])

        # To RGB layers (각 해상도마다)
        self.to_rgb = nn.ModuleList([
            nn.Conv2d(base_channels, 3, 1),  # 4×4
            nn.Conv2d(base_channels, 3, 1),  # 8×8
            nn.Conv2d(base_channels // 2, 3, 1),  # 16×16
            nn.Conv2d(base_channels // 4, 3, 1),  # 32×32
            nn.Conv2d(base_channels // 8, 3, 1),  # 64×64
            nn.Conv2d(base_channels // 16, 3, 1), # 128×128
            nn.Conv2d(base_channels // 32, 3, 1), # 256×256
            nn.Conv2d(base_channels // 32, 3, 1), # 512×512
            nn.Conv2d(base_channels // 32, 3, 1), # 1024×1024
        ])

    def _make_block(self, in_channels, out_channels):
        """Single progressive block with upsampling"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, z, depth, alpha):
        """
        Args:
            z: latent code (batch, z_dim)
            depth: current depth (0=4×4, 1=8×8, ..., 8=1024×1024)
            alpha: fade-in parameter (0→1)
        """
        # Initial 4×4
        x = self.initial(z.view(z.size(0), -1, 1, 1))

        if depth == 0:
            return torch.tanh(self.to_rgb[0](x))

        # Progressive blocks
        for i in range(depth):
            x = self.blocks[i](x)

        # Fade-in
        if alpha < 1:
            # Previous resolution
            x_prev = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            x_prev = self.blocks[depth - 1](x_prev)
            rgb_prev = self.to_rgb[depth - 1](x_prev)
            rgb_prev = F.interpolate(rgb_prev, scale_factor=2, mode='nearest')

            # Current resolution
            rgb_curr = self.to_rgb[depth](x)

            # Blend
            rgb = (1 - alpha) * rgb_prev + alpha * rgb_curr
        else:
            rgb = self.to_rgb[depth](x)

        return torch.tanh(rgb)

class ProgressiveDiscriminator(nn.Module):
    """
    Progressive GAN Discriminator
    """

    def __init__(self, base_channels=512):
        super().__init__()

        # From RGB layers
        self.from_rgb = nn.ModuleList([
            nn.Conv2d(3, base_channels, 1),  # 4×4
            nn.Conv2d(3, base_channels, 1),  # 8×8
            nn.Conv2d(3, base_channels // 2, 1),  # 16×16
            nn.Conv2d(3, base_channels // 4, 1),  # 32×32
            nn.Conv2d(3, base_channels // 8, 1),  # 64×64
            nn.Conv2d(3, base_channels // 16, 1), # 128×128
            nn.Conv2d(3, base_channels // 32, 1), # 256×256
            nn.Conv2d(3, base_channels // 32, 1), # 512×512
            nn.Conv2d(3, base_channels // 32, 1), # 1024×1024
        ])

        # Progressive blocks (reverse order)
        self.blocks = nn.ModuleList([
            self._make_block(base_channels // 32, base_channels // 32), # 1024→512
            self._make_block(base_channels // 32, base_channels // 32), # 512→256
            self._make_block(base_channels // 32, base_channels // 16), # 256→128
            self._make_block(base_channels // 16, base_channels // 8),  # 128→64
            self._make_block(base_channels // 8, base_channels // 4),   # 64→32
            self._make_block(base_channels // 4, base_channels // 2),   # 32→16
            self._make_block(base_channels // 2, base_channels),        # 16→8
            self._make_block(base_channels, base_channels),             # 8→4
        ])

        # Final 4×4
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(base_channels, 1)
        )

    def _make_block(self, in_channels, out_channels):
        """Single progressive block with downsampling"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2)
        )

    def forward(self, x, depth, alpha):
        """
        Args:
            x: image (batch, 3, H, W)
            depth: current depth
            alpha: fade-in parameter
        """
        if depth == 0:
            x = self.from_rgb[0](x)
            return self.final(x)

        # Fade-in
        if alpha < 1:
            # Current resolution
            x_curr = self.from_rgb[depth](x)

            # Previous resolution
            x_prev = F.avg_pool2d(x, 2)
            x_prev = self.from_rgb[depth - 1](x_prev)

            # Blend
            x = (1 - alpha) * x_prev + alpha * x_curr
        else:
            x = self.from_rgb[depth](x)

        # Progressive blocks
        for i in range(depth, 0, -1):
            x = self.blocks[8 - i](x)

        return self.final(x)
```

---

## 🧪 2. Training Progressive GAN

### Training Schedule

```python
def train_progressive_gan():
    """
    Progressive training loop
    """
    G = ProgressiveGenerator().cuda()
    D = ProgressiveDiscriminator().cuda()

    # Training schedule
    depths = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 4×4 → 1024×1024
    images_per_depth = [800_000, 800_000, 800_000, 800_000, 600_000, 400_000, 200_000, 100_000, 50_000]

    for depth in depths:
        resolution = 4 * (2 ** depth)
        print(f"Training at {resolution}×{resolution}")

        # Fade-in phase
        fade_in_images = images_per_depth[depth] // 2
        for i in range(fade_in_images):
            alpha = i / fade_in_images  # 0 → 1

            # Load data at current resolution
            real_images = load_images(resolution)

            # Train
            d_loss, g_loss = train_step(G, D, real_images, depth, alpha)

        # Stabilization phase (alpha = 1)
        stabilize_images = images_per_depth[depth] // 2
        for i in range(stabilize_images):
            real_images = load_images(resolution)
            d_loss, g_loss = train_step(G, D, real_images, depth, alpha=1.0)

        # Save checkpoint
        torch.save({
            'G': G.state_dict(),
            'D': D.state_dict(),
            'depth': depth
        }, f'progressive_gan_depth_{depth}.pt')
```

---

## 🛡️ 3. Normalization Techniques

### 3.1 Pixel Normalization

**문제**: Generator의 activation이 폭주

**해결**: 각 픽셀을 normalize

```python
class PixelNorm(nn.Module):
    def forward(self, x):
        # x: (batch, channels, H, W)
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

# Generator에 추가
self.pixel_norm = PixelNorm()
x = self.pixel_norm(x)
```

### 3.2 Minibatch Standard Deviation

**문제**: Mode collapse (다양성 부족)

**해결**: Batch 통계를 D에 제공

```python
class MinibatchStdDev(nn.Module):
    def forward(self, x):
        # x: (batch, channels, H, W)
        batch_size, _, height, width = x.shape

        # Standard deviation across batch
        std = torch.std(x, dim=0, keepdim=False)  # (channels, H, W)
        mean_std = torch.mean(std)  # scalar

        # Append as extra channel
        mean_std = mean_std.expand(batch_size, 1, height, width)
        return torch.cat([x, mean_std], dim=1)  # (batch, channels+1, H, W)

# Discriminator final layer 전에 추가
self.minibatch_std = MinibatchStdDev()
x = self.minibatch_std(x)
```

### 3.3 Equalized Learning Rate

**아이디어**: 모든 layer가 같은 learning dynamics

```python
class EqualizedConv2d(nn.Module):
    """
    Conv2d with equalized learning rate
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        # He initialization
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        # Scale factor
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = np.sqrt(2 / fan_in)

    def forward(self, x):
        return self.conv(x * self.scale)
```

---

## 📊 4. Evaluation Metrics

### 4.1 Inception Score (IS)

**아이디어**: 생성 이미지가 명확하고 다양해야 함

```python
from torchvision.models import inception_v3
import numpy as np

def calculate_inception_score(images, splits=10):
    """
    Inception Score 계산

    Args:
        images: (N, 3, H, W) tensor
    """
    # Load Inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
    inception_model.eval()

    # Resize to 299×299
    images_resized = F.interpolate(images, size=(299, 299), mode='bilinear')

    # Get predictions
    with torch.no_grad():
        preds = []
        for i in range(0, len(images), 32):
            batch = images_resized[i:i+32]
            pred = F.softmax(inception_model(batch), dim=1).cpu().numpy()
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)

    # Split into groups
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits)]

        # KL divergence
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * (np.log(pyx + 1e-10) - np.log(py + 1e-10))))

        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# 사용
IS_mean, IS_std = calculate_inception_score(generated_images)
print(f"Inception Score: {IS_mean:.2f} ± {IS_std:.2f}")
# Higher is better! (>10 = good)
```

### 4.2 Fréchet Inception Distance (FID)

**아이디어**: Real과 Fake의 feature 분포 거리

```python
from scipy.linalg import sqrtm

def calculate_fid(real_images, fake_images, inception_model):
    """
    FID 계산
    """
    def get_activations(images, model):
        images = F.interpolate(images, size=(299, 299), mode='bilinear')
        with torch.no_grad():
            features = model(images)  # Use features before final layer
        return features.cpu().numpy()

    # Get features
    act_real = get_activations(real_images, inception_model)
    act_fake = get_activations(fake_images, inception_model)

    # Calculate statistics
    mu_real, sigma_real = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    mu_fake, sigma_fake = act_fake.mean(axis=0), np.cov(act_fake, rowvar=False)

    # FID
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

# 사용
fid_score = calculate_fid(real_images, generated_images, inception_model)
print(f"FID: {fid_score:.2f}")
# Lower is better! (< 10 = excellent)
```

---

## 🎓 학습 목표

- [ ] Progressive growing 개념 이해
- [ ] Smooth fade-in 구현
- [ ] Pixel normalization 이해
- [ ] Minibatch std dev 구현
- [ ] Equalized learning rate 이해
- [ ] IS, FID 계산 구현

---

## 💡 실전 팁

### Training Schedule

```python
# NVIDIA ProGAN 논문 스케줄
4×4:   800K images, ~1 day
8×8:   800K images, ~1 day
16×16: 800K images, ~1 day
32×32: 800K images, ~2 days
64×64: 600K images, ~3 days
128×128: 400K images, ~4 days
256×256: 200K images, ~5 days
512×512: 100K images, ~7 days
1024×1024: 50K images, ~10 days

# Total: ~35 days on 8× V100 GPUs
```

### Memory Optimization

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    fake = G(z, depth, alpha)
    d_loss = discriminator_loss(D, real, fake, depth, alpha)

scaler.scale(d_loss).backward()
scaler.step(d_optimizer)
scaler.update()
```

---

## ⏭️ 다음

👉 [Week 10: StyleGAN](./03-stylegan.md)

**이제 style-based generation으로 더 정교한 제어를 배워봅시다!** 🚀
