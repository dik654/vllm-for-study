# Week 6: DDPM Implementation with U-Net

## 🎯 목표

**Noise predictor (U-Net)를 구현하고 DDPM을 완전히 구현하기!**

```python
# Noise predictor
ε_θ(x_t, t) = U-Net(x_t, t)

# Input: Noisy image + timestep
# Output: Predicted noise
```

---

## 🏗️ 1. U-Net Architecture

### 핵심 아이디어

**Encoder-Decoder + Skip Connections**

```
Input (64×64)
     ↓
[Encoder] 64×64 → 32×32 → 16×16 → 8×8
     |        |        |        |
     └────────┼────────┼────────┘
              ↓        ↓        ↓
[Decoder] 8×8 → 16×16 → 32×32 → 64×64
     ↓
Output (64×64)
```

**Skip connections**: 고해상도 정보 보존!

### Components

1. **Residual Block**: Conv + GroupNorm + SiLU
2. **Self-Attention**: 전역 정보 통합
3. **Time Embedding**: Timestep 정보 주입
4. **Down/Upsampling**: 해상도 조절

---

## 🔧 2. Building Blocks

### Time Embedding

```python
import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    """Positional encoding style time embedding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (batch,)
        device = t.device
        half_dim = self.dim // 2

        # Frequency
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Time embedding
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb  # (batch, dim)

# 사용
time_emb = SinusoidalTimeEmbedding(dim=128)
t = torch.tensor([0, 100, 500, 999])
emb = time_emb(t)
print(f"Time embedding shape: {emb.shape}")  # (4, 128)
```

### Residual Block

```python
class ResidualBlock(nn.Module):
    """
    Residual block with time embedding
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()

        # First conv
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # Second conv
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        # x: (batch, in_channels, H, W)
        # time_emb: (batch, time_emb_dim)

        h = self.conv1(x)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)[:, :, None, None]  # (batch, out_channels, 1, 1)
        h = h + time_emb

        h = self.conv2(h)

        # Residual
        return h + self.residual_conv(x)
```

### Self-Attention

```python
class SelfAttention(nn.Module):
    """
    Multi-head self-attention for spatial features
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # x: (batch, channels, H, W)
        batch, channels, H, W = x.shape

        # Normalize
        h = self.norm(x)

        # QKV
        qkv = self.qkv(h)  # (batch, channels*3, H, W)
        qkv = qkv.reshape(batch, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, batch, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)

        # Apply
        out = torch.matmul(attn, v)  # (batch, num_heads, H*W, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(batch, channels, H, W)

        # Project
        out = self.proj(out)

        # Residual
        return x + out
```

### Down/Upsampling

```python
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
```

---

## 🎨 3. Complete U-Net

```python
class UNet(nn.Module):
    """
    U-Net for diffusion models
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=512,
        dropout=0.1,
        attention_resolutions=(16, 8)
    ):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        channels = [base_channels]
        in_ch = base_channels

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                block = ResidualBlock(in_ch, out_ch, time_emb_dim, dropout)
                self.encoder.append(block)
                channels.append(out_ch)
                in_ch = out_ch

            # Add attention at certain resolutions
            # (Assume input is 64×64, track resolution)
            resolution = 64 // (2 ** i)
            if resolution in attention_resolutions:
                self.encoder.append(SelfAttention(out_ch))
                channels.append(out_ch)

            # Downsample (except last)
            if i < len(channel_mults) - 1:
                self.downsamplers.append(Downsample(out_ch))
                channels.append(out_ch)
            else:
                self.downsamplers.append(nn.Identity())

        # Middle
        self.middle = nn.ModuleList([
            ResidualBlock(out_ch, out_ch, time_emb_dim, dropout),
            SelfAttention(out_ch),
            ResidualBlock(out_ch, out_ch, time_emb_dim, dropout)
        ])

        # Decoder
        self.decoder = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult

            for j in range(num_res_blocks + 1):
                # Skip connection
                skip_ch = channels.pop()
                block = ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout)
                self.decoder.append(block)
                in_ch = out_ch

            # Add attention
            resolution = 64 // (2 ** i)
            if resolution in attention_resolutions:
                self.decoder.append(SelfAttention(out_ch))

            # Upsample (except first)
            if i > 0:
                self.upsamplers.append(Upsample(out_ch))
            else:
                self.upsamplers.append(nn.Identity())

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # x: (batch, channels, H, W)
        # t: (batch,)

        # Time embedding
        time_emb = self.time_mlp(t)

        # Initial conv
        h = self.conv_in(x)
        skips = [h]

        # Encoder
        encoder_idx = 0
        for i, downsampler in enumerate(self.downsamplers):
            # Residual blocks + attention
            for _ in range(2):  # num_res_blocks
                h = self.encoder[encoder_idx](h, time_emb)
                skips.append(h)
                encoder_idx += 1

            # Attention (if exists)
            if isinstance(self.encoder[encoder_idx], SelfAttention):
                h = self.encoder[encoder_idx](h)
                skips.append(h)
                encoder_idx += 1

            # Downsample
            h = downsampler(h)
            if not isinstance(downsampler, nn.Identity):
                skips.append(h)

        # Middle
        for block in self.middle:
            if isinstance(block, SelfAttention):
                h = block(h)
            else:
                h = block(h, time_emb)

        # Decoder
        decoder_idx = 0
        for i, upsampler in enumerate(self.upsamplers):
            # Residual blocks + skip connections
            for _ in range(3):  # num_res_blocks + 1
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder[decoder_idx](h, time_emb)
                decoder_idx += 1

            # Attention (if exists)
            if decoder_idx < len(self.decoder) and isinstance(self.decoder[decoder_idx], SelfAttention):
                h = self.decoder[decoder_idx](h)
                decoder_idx += 1

            # Upsample
            h = upsampler(h)

        # Output
        return self.conv_out(h)

# 사용
model = UNet(in_channels=3, out_channels=3)
x = torch.randn(2, 3, 64, 64)
t = torch.tensor([0, 500])

noise_pred = model(x, t)
print(f"Input shape: {x.shape}")
print(f"Output shape: {noise_pred.shape}")
```

---

## 🎓 4. Training DDPM

### Complete Training Loop

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

# Model
model = UNet(in_channels=3, out_channels=3).cuda()

# Noise scheduler
scheduler = NoiseScheduler(num_timesteps=1000)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# EMA (Exponential Moving Average)
from copy import deepcopy

ema_model = deepcopy(model)
ema_decay = 0.9999

def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

# Training
num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0

    for x_0, _ in dataloader:
        x_0 = x_0.cuda()
        batch_size = x_0.size(0)

        # Random timestep
        t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=x_0.device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = scheduler.add_noise(x_0, t, noise)[0]

        # Predict noise
        noise_pred = model(x_t, t)

        # Loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update EMA
        update_ema(ema_model, model, ema_decay)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    # Sample images every 10 epochs
    if epoch % 10 == 0:
        with torch.no_grad():
            samples = sample(ema_model, (16, 3, 32, 32), 1000, scheduler)
            save_images(samples, f'samples_epoch_{epoch}.png')
```

### EMA (Exponential Moving Average)

**왜 EMA?**
```python
# 훈련 중 모델 가중치는 불안정
# EMA로 평활화된 가중치 유지

θ_ema ← β * θ_ema + (1-β) * θ

# β = 0.9999 (DDPM)
```

---

## 🎨 5. Sampling

### DDPM Sampling (1000 steps)

```python
@torch.no_grad()
def ddpm_sample(model, shape, num_timesteps, scheduler, device='cuda'):
    """
    Complete DDPM sampling
    """
    model.eval()

    # Start from pure noise
    x = torch.randn(shape, device=device)

    for t in reversed(range(num_timesteps)):
        # Predict noise
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch)

        # Get coefficients
        alpha = scheduler.alphas[t]
        alpha_bar = scheduler.alpha_bars[t]
        beta = scheduler.betas[t]

        # Compute mean
        coef1 = 1 / torch.sqrt(alpha)
        coef2 = beta / torch.sqrt(1 - alpha_bar)
        mean = coef1 * (x - coef2 * noise_pred)

        # Add noise (except last step)
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta)
            x = mean + sigma * noise
        else:
            x = mean

    return x

# 사용
samples = ddpm_sample(ema_model, (16, 3, 32, 32), 1000, scheduler)
```

---

## 🎓 학습 목표

- [ ] U-Net 아키텍처 이해 (Encoder-Decoder + Skip)
- [ ] Time embedding 구현
- [ ] Residual block + Self-attention 구현
- [ ] Complete U-Net 구현
- [ ] DDPM 훈련 루프 구현
- [ ] EMA 이해 및 적용
- [ ] Sampling 구현

---

## 💡 실전 팁

### 1. Hyperparameters

```python
# 작은 데이터셋 (MNIST, CIFAR-10)
base_channels = 128
channel_mults = (1, 2, 2, 2)
num_res_blocks = 2

# 큰 데이터셋 (ImageNet)
base_channels = 256
channel_mults = (1, 1, 2, 2, 4, 4)
num_res_blocks = 3
```

### 2. Memory Optimization

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    noise_pred = model(x_t, t)
    loss = mse_loss(noise_pred, noise)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Monitoring

```python
# FID (Fréchet Inception Distance)
from pytorch_fid import fid_score

fid = fid_score.calculate_fid_given_paths(
    [real_images_path, generated_images_path],
    batch_size=50,
    device='cuda',
    dims=2048
)

print(f"FID: {fid:.2f}")
# Lower is better! (< 10 = 매우 좋음)
```

---

## ⏭️ 다음

👉 [Week 7: Sampling & Conditioning](./03-sampling-conditioning.md)

**이제 더 빠른 sampling과 조건부 생성을 배워봅시다!** 🚀
