# Vision Transformers (ViT)

## 🎯 목표

**Transformer를 Vision에 적용하기**

이미지 = 시퀀스로 취급!

```
Image (224x224x3)
  ↓ Patch embedding
Sequence of patches (196 tokens)
  ↓ Transformer Encoder
Classification output
```

---

## 🖼️ 문제: CNN에서 Transformer로

### 기존 접근: CNN

```python
# Traditional approach
image → Conv layers → Pooling → FC → Class
```

**한계점**:
- Inductive bias (locality, translation invariance)
- 큰 receptive field 얻기 어려움
- Global context 부족

### 새로운 접근: Transformer

```python
# ViT approach
image → Patches → Linear projection → Transformer → Class
```

**장점**:
- Global attention from the start
- 적은 inductive bias (더 일반적)
- 대규모 데이터에서 더 좋은 scaling

---

## 📦 Vision Transformer (ViT) 아키텍처

### 핵심 아이디어

**"An Image is Worth 16x16 Words"**

이미지를 패치로 나누고, 각 패치를 token처럼 취급!

```
Original Image: (224, 224, 3)
  ↓
Patches: (14, 14, 16×16×3) = 196 patches
  ↓
Flattened: 196 tokens, each 768-dim
  ↓
Transformer Encoder
  ↓
[CLS] token for classification
```

### 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    이미지를 패치로 나누고 embedding

    핵심 아이디어: 각 패치를 하나의 token으로 취급
    의도: CNN의 convolution을 linear projection으로 대체
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """
        Args:
            img_size: 입력 이미지 크기 (정사각형 가정)
            patch_size: 패치 크기 (16x16이 일반적)
            in_channels: 입력 채널 (RGB = 3)
            embed_dim: 임베딩 차원
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 14×14 = 196

        # 패치 추출 및 projection을 conv2d로 구현
        # 의도: stride=patch_size인 convolution = 패치 단위 linear projection
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size  # non-overlapping patches
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224) - 입력 이미지
        Returns:
            patches: (batch, 196, 768) - 패치 임베딩
        """
        # Conv2d로 패치 추출 및 projection
        # (batch, 3, 224, 224) → (batch, 768, 14, 14)
        x = self.projection(x)

        # Flatten spatial dimensions
        # (batch, 768, 14, 14) → (batch, 768, 196)
        x = x.flatten(2)

        # Transpose to (batch, num_patches, embed_dim)
        # (batch, 768, 196) → (batch, 196, 768)
        x = x.transpose(1, 2)

        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT)

    Paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
    핵심: 이미지를 패치 시퀀스로 변환 후 Transformer 적용
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()

        # 패치 임베딩
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # [CLS] token: classification에 사용할 특수 token
        # 의도: BERT의 [CLS]와 동일한 역할
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding: 각 패치의 위치 정보
        # 의도: Transformer는 순서 정보가 없으므로 명시적으로 추가
        # num_patches + 1 (for [CLS] token)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder blocks
        # 의도: 패치 간 global attention으로 이미지 이해
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                dropout
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        # 의도: [CLS] token의 representation으로 classification
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        ViT forward pass

        Args:
            x: (batch, 3, 224, 224) - 입력 이미지
        Returns:
            logits: (batch, num_classes) - classification logits
        """
        batch_size = x.shape[0]

        # 1. 패치 임베딩
        # (batch, 3, 224, 224) → (batch, 196, 768)
        x = self.patch_embed(x)

        # 2. [CLS] token 추가
        # 의도: classification을 위한 특수 token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 197, 768)

        # 3. Positional embedding 추가
        # 의도: 각 패치의 공간적 위치 정보 제공
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Transformer blocks
        # 의도: 패치 간 관계 학습 (global context)
        for block in self.blocks:
            x = block(x)

        # 5. Normalization
        x = self.norm(x)

        # 6. [CLS] token으로 classification
        # 의도: [CLS] token이 전체 이미지 정보를 aggregate
        cls_token_final = x[:, 0]  # (batch, 768)

        # 7. Classification head
        logits = self.head(cls_token_final)  # (batch, num_classes)

        return logits


class TransformerBlock(nn.Module):
    """
    ViT Transformer Block

    구조: Pre-LN
    - LayerNorm → Multi-Head Attention → Residual
    - LayerNorm → MLP → Residual
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()

        # Layer norms (Pre-LN 방식)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # MLP (Feed-Forward)
        # 의도: 비선형 변환으로 표현력 증가
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),  # ViT는 GELU 사용
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches+1, embed_dim)
        """
        # Pre-LN: norm 후 attention
        attn_out, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x)
        )
        x = x + attn_out  # Residual connection

        # Pre-LN: norm 후 MLP
        x = x + self.mlp(self.norm2(x))  # Residual connection

        return x


# ViT 모델 variants
def vit_tiny(num_classes=1000):
    """ViT-Tiny: 5.7M params"""
    return ViT(
        patch_size=16,
        embed_dim=192,
        num_heads=3,
        num_layers=12,
        num_classes=num_classes
    )

def vit_small(num_classes=1000):
    """ViT-Small: 22M params"""
    return ViT(
        patch_size=16,
        embed_dim=384,
        num_heads=6,
        num_layers=12,
        num_classes=num_classes
    )

def vit_base(num_classes=1000):
    """ViT-Base: 86M params (original paper)"""
    return ViT(
        patch_size=16,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        num_classes=num_classes
    )

def vit_large(num_classes=1000):
    """ViT-Large: 307M params"""
    return ViT(
        patch_size=16,
        embed_dim=1024,
        num_heads=16,
        num_layers=24,
        num_classes=num_classes
    )

def vit_huge(num_classes=1000):
    """ViT-Huge: 632M params"""
    return ViT(
        patch_size=14,
        embed_dim=1280,
        num_heads=16,
        num_layers=32,
        num_classes=num_classes
    )


# 사용 예제
if __name__ == "__main__":
    # 모델 생성
    model = vit_base(num_classes=1000)

    # 이미지 입력
    images = torch.randn(2, 3, 224, 224)  # batch=2

    # Forward pass
    logits = model(images)
    print(f"Output shape: {logits.shape}")  # (2, 1000)

    # 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")  # ~86M
```

---

## 🎯 훈련 전략

### Pre-training

**ViT의 핵심: 대규모 데이터 필요!**

```python
# 1. ImageNet-1K (1.3M 이미지) - 부족!
# 성능: ResNet보다 낮음

# 2. ImageNet-21K (14M 이미지) - 충분
# 성능: ResNet과 비슷

# 3. JFT-300M (300M 이미지) - 최고!
# 성능: ResNet 능가
```

**왜?**
- CNN: Strong inductive bias (locality)
- ViT: Weak inductive bias → 더 많은 데이터 필요

### Fine-tuning

```python
def finetune_vit(pretrained_model, num_classes_new):
    """
    Pre-trained ViT를 새로운 task에 fine-tuning

    전략:
    1. Classification head만 교체
    2. 작은 learning rate 사용
    3. High resolution fine-tuning (선택)
    """
    # 1. Head 교체
    # 의도: 새로운 class 수에 맞게 조정
    pretrained_model.head = nn.Linear(
        pretrained_model.head.in_features,
        num_classes_new
    )

    # 2. Learning rate 설정
    # 의도: Pre-trained weights 보존하면서 adaptation
    optimizer = torch.optim.AdamW([
        {'params': pretrained_model.blocks.parameters(), 'lr': 1e-5},  # Backbone: 작은 LR
        {'params': pretrained_model.head.parameters(), 'lr': 1e-3}     # Head: 큰 LR
    ])

    return pretrained_model, optimizer


# High-resolution fine-tuning
def high_res_finetune(model, new_img_size=384):
    """
    더 높은 해상도로 fine-tuning

    핵심: Positional embedding 보간
    의도: 더 많은 패치 → 더 세밀한 표현
    """
    old_num_patches = model.patch_embed.num_patches
    new_num_patches = (new_img_size // model.patch_embed.patch_size) ** 2

    if old_num_patches != new_num_patches:
        # Positional embedding 보간 (interpolation)
        # 의도: 기존 위치 정보를 새로운 크기에 맞게 확장
        old_pos_embed = model.pos_embed[:, 1:, :]  # [CLS] 제외

        # 2D interpolation
        # (1, 196, 768) → (1, 768, 14, 14) → interpolate → (1, 768, 24, 24) → (1, 576, 768)
        old_pos_embed = old_pos_embed.transpose(1, 2)
        old_h = old_w = int(old_num_patches ** 0.5)
        old_pos_embed = old_pos_embed.reshape(1, -1, old_h, old_w)

        new_h = new_w = int(new_num_patches ** 0.5)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False
        )

        new_pos_embed = new_pos_embed.reshape(1, -1, new_num_patches).transpose(1, 2)

        # [CLS] token 다시 추가
        model.pos_embed = nn.Parameter(
            torch.cat([model.pos_embed[:, :1, :], new_pos_embed], dim=1)
        )

    return model
```

---

## 📊 ViT vs CNN

### 성능 비교

```
Dataset: ImageNet-1K (1.3M images)
Training: From scratch

Model              | Top-1 Acc | Params
-------------------|-----------|--------
ResNet-50          | 76.5%     | 25M
ResNet-152         | 78.3%     | 60M
ViT-Base/16        | 77.9%     | 86M     ← CNN과 비슷 (충분한 데이터)


Dataset: JFT-300M (300M images) → ImageNet-1K fine-tune

Model              | Top-1 Acc | Params
-------------------|-----------|--------
BiT-ResNet-152x4   | 87.5%     | 928M
ViT-Huge/14        | 88.5%     | 632M    ← CNN 능가!
```

### 계산 효율성

```python
# FLOPs 비교 (ImageNet inference)

ResNet-50:    ~4 GFLOPs
ViT-Base/16:  ~17 GFLOPs  ← 더 많은 연산

# 하지만!
# - ViT는 parallelizable (Transformer)
# - CNN은 sequential (layer by layer)
# → 실제 throughput은 비슷하거나 ViT가 더 빠름
```

---

## 🎯 ViT의 장단점

### 장점

1. **Global Receptive Field**
   - 첫 layer부터 전체 이미지 볼 수 있음
   - Long-range dependencies 학습 용이

2. **Scalability**
   - 모델 크기 증가 시 성능 향상 지속
   - CNN보다 더 나은 scaling law

3. **Transfer Learning**
   - Pre-trained ViT는 다양한 task에 잘 전이
   - Fine-tuning efficiency

4. **Interpretability**
   - Attention maps으로 "어디를 보는지" 시각화 가능

### 단점

1. **Data Hungry**
   - 작은 데이터셋에서 CNN보다 성능 낮음
   - ImageNet-1K로는 부족

2. **Computational Cost**
   - Attention은 O(n²) complexity
   - 큰 이미지에서 비효율적

3. **Inductive Bias 부족**
   - Locality, translation equivariance 없음
   - 더 많은 데이터로 학습해야 함

---

## 🚀 ViT 변형들

### DeiT (Data-efficient ViT)

**문제**: ViT는 JFT-300M 같은 대규모 데이터 필요

**해결**: Knowledge distillation으로 ImageNet-1K에서도 잘 동작

```python
class DeiT(nn.Module):
    """
    DeiT: Data-efficient Image Transformer

    핵심 아이디어: Teacher-student distillation
    - CNN teacher (ResNet 등)의 지식을 ViT student에 전달
    """
    def __init__(self, ...):
        super().__init__()
        # ViT와 동일한 구조
        # + distillation token 추가
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        # [CLS] token과 [DIST] token 모두 사용
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        dist_tokens = self.dist_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)

        # ... transformer blocks ...

        # 두 token 모두 classification에 사용
        cls_output = x[:, 0]
        dist_output = x[:, 1]

        return cls_output, dist_output
```

**결과**: ImageNet-1K만으로 ViT-Base 수준 달성!

### Swin Transformer

**문제**: ViT는 patch 수가 많으면 O(n²) 때문에 느림

**해결**: Hierarchical structure + Shifted windows

```
ViT:
  모든 patch가 서로 attention (전역)
  O(n²) complexity

Swin:
  Local window 내에서만 attention (지역)
  + Shifted windows로 global connection
  O(n) complexity!
```

**특징**:
- Multi-scale features (CNN처럼)
- Efficient for high-resolution images
- Object detection, segmentation에 적합

---

## 💻 실습: ViT로 CIFAR-10 훈련

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 데이터 준비
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# ViT 모델 (작은 버전, CIFAR-10용)
model = ViT(
    img_size=32,           # CIFAR-10은 32x32
    patch_size=4,          # 작은 이미지이므로 작은 patch
    in_channels=3,
    num_classes=10,
    embed_dim=256,         # 작은 embedding
    num_heads=4,
    num_layers=6,
    mlp_ratio=4,
    dropout=0.1
).cuda()

# Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Loss
criterion = nn.CrossEntropyLoss()

# 훈련 루프
for epoch in range(200):
    model.train()
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.cuda(), labels.cuda()

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/200], Loss: {running_loss/len(trainloader):.4f}')

print("훈련 완료!")
```

---

## 📝 Summary

### Key Takeaways

1. **ViT = Transformer for Images**
   - 이미지를 패치로 나누고 시퀀스로 취급
   - [CLS] token으로 classification

2. **Data Requirements**
   - 작은 데이터: CNN이 더 좋음
   - 큰 데이터 (14M+): ViT가 CNN 능가

3. **장점: Global Context**
   - 첫 layer부터 전체 이미지 볼 수 있음
   - Long-range dependencies 학습 용이

4. **단점: Computational Cost**
   - O(n²) attention complexity
   - 큰 이미지에서 비효율적

5. **실전 팁**:
   - Pre-trained 모델 사용 (ImageNet-21K or JFT)
   - Fine-tuning with high resolution
   - DeiT/Swin 같은 efficient variants 고려

---

## 📚 References

**Papers**:
1. **An Image is Worth 16x16 Words** (Dosovitskiy et al., 2020)
   - Original ViT paper
2. **Training data-efficient image transformers** (Touvron et al., 2021)
   - DeiT
3. **Swin Transformer** (Liu et al., 2021)
   - Hierarchical vision transformer

**Code**:
- HuggingFace: `transformers.ViTModel`
- PyTorch Image Models (timm): `timm.models.vision_transformer`

---

## ⏭️ Next Steps

ViT를 이해했으니:

1. **CLIP** 학습 → Vision-Language pre-training
2. **Swin Transformer** → Efficient hierarchical ViT
3. **Object Detection** → DETR (Detection Transformer)

👉 Continue to [Multi-Modal Models](../phase2-bert-gpt/06-multimodal-models.md)

**Vision Transformer 마스터 완료!** 🎨
