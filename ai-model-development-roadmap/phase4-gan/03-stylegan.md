# Week 10: StyleGAN - Style-Based Generation

## 🎯 목표

**Style-based generation으로 정교한 제어 가능한 이미지 생성!**

```python
# StyleGAN의 혁신:
1. Mapping Network: Z → W (disentangled latent space)
2. AdaIN: Style을 각 layer에 주입
3. Style Mixing: 다양한 조합 가능
4. Noise Injection: Stochastic variation

# 결과: SOTA 품질 + 뛰어난 제어성!
```

---

## 📖 1. Style-Based Generator

### 전통적 GAN vs StyleGAN

**전통적 GAN**:
```python
z → Generator → image
# z에서 직접 제어하기 어려움
```

**StyleGAN**:
```python
z → Mapping Network → w
w → AdaIN at each layer → image
# w 공간에서 쉽게 제어!
```

### Architecture Overview

```
Input: z ∈ Z (Gaussian noise)
  ↓
Mapping Network (8-layer MLP)
  ↓
w ∈ W (disentangled latent)
  ↓
Style vectors for each layer
  ↓
Synthesis Network (AdaIN at each resolution)
  ↓
Output: 1024×1024 image
```

---

## 🧠 2. Mapping Network

### 목적

**Z (Gaussian) → W (disentangled)**

```python
# Z 공간: 얽힌 features
# W 공간: 분리된 features
# → 더 쉬운 제어!
```

### 구현

```python
class MappingNetwork(nn.Module):
    """
    Z → W transformation
    """

    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.extend([
                nn.Linear(in_dim, w_dim),
                nn.LeakyReLU(0.2)
            ])

        self.mapping = nn.Sequential(*layers)

        # Normalize w
        self.normalize = lambda w: w / torch.sqrt(torch.mean(w ** 2, dim=1, keepdim=True) + 1e-8)

    def forward(self, z):
        # z: (batch, z_dim)
        w = self.mapping(z)
        w = self.normalize(w)
        return w  # (batch, w_dim)
```

---

## 🎨 3. Adaptive Instance Normalization (AdaIN)

### 핵심 아이디어

**Style을 각 layer의 통계로 주입!**

```python
# Instance Normalization:
x_norm = (x - μ(x)) / σ(x)

# Adaptive:
AdaIN(x, y) = y_scale * ((x - μ(x)) / σ(x)) + y_bias

# y_scale, y_bias: style vector로부터 계산!
```

### 구현

```python
class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization
    """

    def __init__(self, num_features, w_dim):
        super().__init__()

        # Affine transformation from w
        self.scale_transform = nn.Linear(w_dim, num_features)
        self.bias_transform = nn.Linear(w_dim, num_features)

    def forward(self, x, w):
        # x: (batch, channels, H, W)
        # w: (batch, w_dim)

        # Instance normalization
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-8
        x_norm = (x - mean) / std

        # Style modulation
        scale = self.scale_transform(w)[:, :, None, None]
        bias = self.bias_transform(w)[:, :, None, None]

        return scale * x_norm + bias

class StyleBlock(nn.Module):
    """
    Single StyleGAN block
    """

    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.activation1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.adain2 = AdaIN(out_channels, w_dim)
        self.activation2 = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        # First conv + AdaIN
        x = self.conv1(x)
        x = self.adain1(x, w)
        x = self.activation1(x)

        # Second conv + AdaIN
        x = self.conv2(x)
        x = self.adain2(x, w)
        x = self.activation2(x)

        return x
```

---

## 🎲 4. Noise Injection

### 목적

**Stochastic variation (머리카락, 피부 질감 등)**

```python
# Noise를 각 layer에 추가
# → 세부적인 randomness
```

### 구현

```python
class NoiseInjection(nn.Module):
    """
    Add learnable noise to features
    """

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        # x: (batch, channels, H, W)

        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)

        return x + self.weight * noise

# StyleBlock에 추가
class StyleBlockWithNoise(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.noise = NoiseInjection(out_channels)
        self.adain = AdaIN(out_channels, w_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w, noise=None):
        x = self.conv(x)
        x = self.noise(x, noise)
        x = self.adain(x, w)
        x = self.activation(x)
        return x
```

---

## 🏗️ 5. Complete StyleGAN Generator

```python
class StyleGANGenerator(nn.Module):
    """
    Complete StyleGAN Generator
    """

    def __init__(self, z_dim=512, w_dim=512, base_channels=512):
        super().__init__()

        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim)

        # Constant input (4×4)
        self.constant = nn.Parameter(torch.randn(1, base_channels, 4, 4))

        # Synthesis network (progressive)
        self.layers = nn.ModuleList()
        self.to_rgb = nn.ModuleList()

        # Resolutions: 4→8→16→32→64→128→256→512→1024
        channels = [base_channels, base_channels, base_channels, base_channels // 2,
                    base_channels // 4, base_channels // 8, base_channels // 16,
                    base_channels // 32, base_channels // 32]

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            # Style block
            self.layers.append(
                StyleBlockWithNoise(in_ch, out_ch, w_dim)
            )

            # To RGB
            self.to_rgb.append(nn.Conv2d(out_ch, 3, 1))

    def forward(self, z, return_latents=False):
        # Mapping
        w = self.mapping(z)  # (batch, w_dim)

        # Start from constant
        batch_size = z.size(0)
        x = self.constant.repeat(batch_size, 1, 1, 1)

        # Synthesis
        for i, (layer, to_rgb) in enumerate(zip(self.layers, self.to_rgb)):
            # Upsample (except first)
            if i > 0:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            # Apply style
            x = layer(x, w)

        # Final RGB
        image = torch.tanh(to_rgb(x))

        if return_latents:
            return image, w
        return image
```

---

## 🎭 6. Style Mixing

### 핵심 아이디어

**서로 다른 w를 다른 layer에 적용!**

```python
# Coarse styles (초반 layers): 얼굴 형태, 포즈
# Middle styles (중간 layers): 머리 스타일, 얼굴 특징
# Fine styles (후반 layers): 색상, 미세 질감

# Style mixing:
w1 → layers 0-3  (Person A의 얼굴 형태)
w2 → layers 4-7  (Person B의 머리 스타일)
w1 → layers 8-11 (Person A의 색상)
```

### 구현

```python
def style_mixing(G, z1, z2, mix_layer=4):
    """
    Style mixing generation
    """
    # Get w vectors
    w1 = G.mapping(z1)
    w2 = G.mapping(z2)

    # Start from constant
    x = G.constant.repeat(z1.size(0), 1, 1, 1)

    # Apply styles
    for i, layer in enumerate(G.layers):
        if i > 0:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')

        # Choose style
        w = w1 if i < mix_layer else w2
        x = layer(x, w)

    image = torch.tanh(G.to_rgb[-1](x))
    return image

# 사용
z1 = torch.randn(1, 512).cuda()  # Person A
z2 = torch.randn(1, 512).cuda()  # Person B

# A의 형태 + B의 머리
mixed = style_mixing(G, z1, z2, mix_layer=4)
```

---

## 🔍 7. Latent Space Exploration

### Interpolation in W Space

```python
@torch.no_grad()
def interpolate_w(G, z1, z2, num_steps=10):
    """
    Interpolate in W space
    """
    w1 = G.mapping(z1)
    w2 = G.mapping(z2)

    images = []
    for alpha in np.linspace(0, 1, num_steps):
        w = (1 - alpha) * w1 + alpha * w2
        image = G.synthesis(w)  # Generate from w directly
        images.append(image)

    return torch.cat(images, dim=0)

# 사용
z1 = torch.randn(1, 512).cuda()
z2 = torch.randn(1, 512).cuda()
interpolated = interpolate_w(G, z1, z2, num_steps=10)
```

### Truncation Trick

**생성 품질 개선!**

```python
# W 평균에 가까이 (다양성 ↓, 품질 ↑)
w_avg = compute_w_avg(G, num_samples=10000)

def truncation(w, w_avg, psi=0.7):
    """
    psi = 0: w_avg only (no diversity)
    psi = 1: original w (full diversity)
    psi = 0.7: balance (common choice)
    """
    return w_avg + psi * (w - w_avg)

# 사용
z = torch.randn(1, 512).cuda()
w = G.mapping(z)
w_truncated = truncation(w, w_avg, psi=0.7)
image = G.synthesis(w_truncated)
```

---

## 🎯 8. StyleGAN2 Improvements

### Weight Demodulation

**문제**: AdaIN의 artifact (불필요한 패턴)

**해결**: Weight demodulation

```python
class ModulatedConv2d(nn.Module):
    """
    Weight demodulation (StyleGAN2)
    """

    def __init__(self, in_channels, out_channels, kernel_size, w_dim):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.style_transform = nn.Linear(w_dim, in_channels)

    def forward(self, x, w):
        # Style modulation
        style = self.style_transform(w)[:, :, None, None]  # (batch, in_channels, 1, 1)

        # Modulate weights
        weight = self.weight[None, :, :, :, :] * (style + 1)[:, None, :, :, :]

        # Demodulate
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod[:, :, None, None, None]

        # Reshape for group convolution
        batch = x.size(0)
        x = x.reshape(1, -1, x.size(2), x.size(3))
        weight = weight.reshape(-1, weight.size(2), weight.size(3), weight.size(4))

        # Convolution
        out = F.conv2d(x, weight, padding=1, groups=batch)
        out = out.reshape(batch, -1, out.size(2), out.size(3))

        return out
```

### Path Length Regularization

**목적**: W 공간에서 부드러운 변화

```python
def path_length_regularization(G, w, y):
    """
    Perceptual path length (PPL) regularization
    """
    # Generate image
    img = G.synthesis(w)

    # Jacobian w.r.t w
    pl_noise = torch.randn_like(img) / np.sqrt(img.size(2) * img.size(3))
    pl_grads = torch.autograd.grad(
        outputs=(img * pl_noise).sum(),
        inputs=w,
        create_graph=True
    )[0]

    # Path length
    pl_lengths = torch.sqrt(pl_grads.pow(2).sum(dim=1).mean(dim=1))

    # Regularization
    pl_mean = y.lerp(pl_lengths.mean(), 0.01)  # Moving average
    pl_penalty = (pl_lengths - pl_mean).pow(2).mean()

    return pl_penalty, pl_mean
```

---

## 🎓 학습 목표

- [ ] Mapping network (Z → W) 이해
- [ ] AdaIN 구현
- [ ] Noise injection 이해
- [ ] Style mixing 실험
- [ ] Truncation trick 적용
- [ ] StyleGAN2 개선사항 이해

---

## 💡 실전 팁

### Training

```python
# Optimizer
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.002, betas=(0, 0.99))
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.002, betas=(0, 0.99))

# Loss
# R1 regularization (Discriminator)
def d_r1_loss(real_pred, real_img):
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=True
    )[0]
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

# Path length regularization (Generator)
pl_reg_weight = 2.0  # Every few iterations
```

### Pretrained Models

```python
# 사용 (HuggingFace)
from transformers import pipeline

generator = pipeline("image-generation", model="nvidia/stylegan2-ffhq-1024x1024")
image = generator("Generate a face")[0]

# 또는 직접 로드
import torch
G = torch.hub.load('NVlabs/stylegan2-ada-pytorch', 'generator').cuda()

z = torch.randn(1, 512).cuda()
image = G(z)
```

---

## 📚 참고 논문

- **StyleGAN**: A Style-Based Generator Architecture for GANs (Karras et al., 2019)
  - [Paper](https://arxiv.org/abs/1812.04948)

- **StyleGAN2**: Analyzing and Improving the Image Quality of StyleGAN (Karras et al., 2020)
  - [Paper](https://arxiv.org/abs/1912.04958)

- **StyleGAN3**: Alias-Free Generative Adversarial Networks (Karras et al., 2021)
  - [Paper](https://arxiv.org/abs/2106.12423)

---

## ⏭️ 다음 단계

Phase 4 완료! 🎉

**GAN의 모든 것을 마스터했습니다!**

👉 [Phase 5: Modern Techniques](../phase5-modern-techniques/)로 진행하여 최신 기술을 배워봅시다!

**"이제 production-ready AI를 만들 준비가 되었습니다!"** 🚀
