# Week 7: Sampling & Conditioning

## 🎯 목표

**빠른 sampling과 조건부 생성 마스터!**

```python
# DDPM: 1000 steps (느림 😢)
# DDIM: 50 steps (빠름! ⚡)

# Unconditional: 랜덤 생성
# Conditional: "A cat" → 고양이 이미지 ✨
```

---

## ⚡ 1. DDIM (Denoising Diffusion Implicit Models)

### 문제: DDPM은 느리다

```python
# DDPM: 1000 steps
for t in range(1000, 0, -1):
    x = denoise_step(x, t)  # 1000번 반복!

# → 이미지 1장 생성에 수십 초...
```

### 해결: DDIM - 비마르코프 프로세스

**핵심 아이디어**: 일부 timestep만 사용!

```python
# DDIM: 50 steps (20배 빠름!)
timesteps = [0, 20, 40, 60, ..., 980, 1000]  # Subset!

for t in timesteps[::-1]:
    x = ddim_step(x, t)  # 50번만!
```

### DDIM 수식

```python
# DDPM (stochastic):
x_{t-1} = μ + σ * noise

# DDIM (deterministic):
x_{t-1} = √ᾱ_{t-1} * pred_x_0 + √(1-ᾱ_{t-1}) * ε_θ(x_t, t)

# pred_x_0 = (x_t - √(1-ᾱ_t) * ε_θ(x_t, t)) / √ᾱ_t
```

### 구현

```python
@torch.no_grad()
def ddim_sample(model, shape, num_inference_steps=50, scheduler=None, eta=0.0):
    """
    DDIM sampling with fewer steps

    Args:
        eta: 0 = deterministic, 1 = DDPM
    """
    device = next(model.parameters()).device

    # Timestep schedule
    timesteps = torch.linspace(
        scheduler.num_timesteps - 1, 0, num_inference_steps
    ).long().to(device)

    # Start from noise
    x = torch.randn(shape, device=device)

    for i, t in enumerate(timesteps):
        # Current and previous timestep
        t_curr = t
        t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0)

        # Predict noise
        t_batch = t_curr.repeat(shape[0])
        noise_pred = model(x, t_batch)

        # Get alpha values
        alpha_bar_curr = scheduler.alpha_bars[t_curr]
        alpha_bar_prev = scheduler.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict x_0
        pred_x_0 = (x - torch.sqrt(1 - alpha_bar_curr) * noise_pred) / torch.sqrt(alpha_bar_curr)

        # Clip for stability
        pred_x_0 = torch.clamp(pred_x_0, -1, 1)

        # Compute direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_prev - eta ** 2 * (
            (1 - alpha_bar_curr) / (1 - alpha_bar_prev) *
            (1 - alpha_bar_prev / alpha_bar_curr)
        )) * noise_pred

        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x_0 + dir_xt

        # Add noise (if eta > 0)
        if eta > 0 and t_prev > 0:
            noise = torch.randn_like(x)
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_curr) *
                (1 - alpha_bar_curr / alpha_bar_prev)
            )
            x_prev = x_prev + sigma * noise

        x = x_prev

    return x

# 사용
samples = ddim_sample(model, (16, 3, 64, 64), num_inference_steps=50, scheduler=scheduler)
```

### DDPM vs DDIM 비교

| Method | Steps | Speed | Deterministic | Quality |
|--------|-------|-------|---------------|---------|
| DDPM | 1000 | 1x | ❌ | Excellent |
| DDIM | 50 | 20x | ✅ | Excellent |

**결론**: DDIM이 압도적! (Stable Diffusion도 DDIM 사용)

---

## 🎨 2. Conditional Generation

### 2.1 Class Conditioning

**목표**: 특정 클래스 이미지 생성

```python
"Generate a cat" → 🐱
"Generate a dog" → 🐶
```

### Class Embedding

```python
class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()

        self.unet = UNet(**kwargs)

        # Class embedding
        self.class_emb = nn.Embedding(num_classes, kwargs['time_emb_dim'])

    def forward(self, x, t, class_labels):
        # Time embedding
        time_emb = self.unet.time_mlp(t)

        # Class embedding
        class_emb = self.class_emb(class_labels)

        # Combine
        cond_emb = time_emb + class_emb

        # Forward with conditional embedding
        return self.unet.forward_with_emb(x, cond_emb)

# 훈련
for x_0, labels in dataloader:
    t = torch.randint(0, num_timesteps, (batch_size,))
    noise = torch.randn_like(x_0)
    x_t = scheduler.add_noise(x_0, t, noise)[0]

    # Conditional prediction
    noise_pred = model(x_t, t, labels)

    loss = mse_loss(noise_pred, noise)
    loss.backward()
```

---

### 2.2 Classifier Guidance

**아이디어**: 학습된 분류기로 생성 방향 조정

```python
# Score: ∇_x log p(x_t)
# Conditional score: ∇_x log p(x_t|y)

# Bayes' rule:
∇_x log p(x_t|y) = ∇_x log p(x_t) + ∇_x log p(y|x_t)
                   ε_θ(x_t, t)   +  ∇_x log p_φ(y|x_t)
```

### 구현

```python
def classifier_guided_sampling(
    model,
    classifier,
    shape,
    class_label,
    guidance_scale=1.0,
    num_steps=50
):
    """
    Classifier-guided sampling
    """
    x = torch.randn(shape)

    for t in reversed(range(num_steps)):
        # Require gradient for classifier
        x.requires_grad = True

        # Predict noise
        t_batch = torch.full((shape[0],), t)
        noise_pred = model(x, t_batch)

        # Classifier gradient
        logits = classifier(x, t_batch)
        log_prob = torch.log_softmax(logits, dim=-1)[:, class_label]
        class_grad = torch.autograd.grad(log_prob.sum(), x)[0]

        # Guided noise prediction
        noise_pred = noise_pred - guidance_scale * class_grad

        # Denoise step
        x = denoise_step(x, t, noise_pred)
        x = x.detach()

    return x

# 사용
samples = classifier_guided_sampling(
    model,
    classifier,
    shape=(16, 3, 64, 64),
    class_label=3,  # Cat
    guidance_scale=2.0
)
```

**문제**: 별도의 classifier 훈련 필요 😢

---

### 2.3 Classifier-Free Guidance (CFG)

**핵심 아이디어**: Classifier 없이 guidance!

```python
# 훈련 시 10-20% 확률로 condition 제거
if random.random() < 0.1:
    class_label = None  # Unconditional

# 추론 시 두 예측 결합
ε_uncond = ε_θ(x_t, t, ∅)       # Unconditional
ε_cond = ε_θ(x_t, t, y)         # Conditional

ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
#          (무조건)   +  w * (방향)

# w > 1: 강한 guidance (더 명확, 덜 다양)
# w = 1: 조건부 생성
# w = 0: 무조건 생성
```

### 구현

```python
class CFGUNet(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.unet = UNet(**kwargs)

        # Class embedding (+ null token)
        self.class_emb = nn.Embedding(num_classes + 1, kwargs['time_emb_dim'])
        self.null_class = num_classes  # Index for null

    def forward(self, x, t, class_labels=None):
        if class_labels is None:
            # Unconditional
            class_labels = torch.full(
                (x.size(0),),
                self.null_class,
                device=x.device
            )

        return self.unet(x, t, class_labels)

# 훈련
for x_0, labels in dataloader:
    # Random drop condition
    mask = torch.rand(labels.size(0)) < 0.1
    labels[mask] = model.null_class

    noise_pred = model(x_t, t, labels)
    loss = mse_loss(noise_pred, noise)

# Sampling with CFG
@torch.no_grad()
def cfg_sample(model, shape, class_label, guidance_scale=7.5, num_steps=50):
    x = torch.randn(shape)

    for t in reversed(range(num_steps)):
        t_batch = torch.full((shape[0],), t)

        # Unconditional prediction
        noise_uncond = model(x, t_batch, class_labels=None)

        # Conditional prediction
        class_labels = torch.full((shape[0],), class_label)
        noise_cond = model(x, t_batch, class_labels)

        # Classifier-free guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Denoise
        x = denoise_step(x, t, noise_pred)

    return x

# 사용
samples = cfg_sample(model, (16, 3, 64, 64), class_label=5, guidance_scale=7.5)
```

**장점**:
- Classifier 불필요!
- 더 좋은 품질
- Stable Diffusion의 핵심 기법!

---

## 📝 3. Text Conditioning (Stable Diffusion Style)

### CLIP Embeddings

```python
from transformers import CLIPTextModel, CLIPTokenizer

# CLIP text encoder
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Encode text
text = "A beautiful sunset over the ocean"
tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
text_embeddings = text_encoder(**tokens).last_hidden_state  # (1, seq_len, 768)
```

### Cross-Attention

```python
class CrossAttention(nn.Module):
    """
    Cross-attention to text embeddings
    """

    def __init__(self, query_dim, context_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        # x: (batch, H*W, query_dim) - image features
        # context: (batch, seq_len, context_dim) - text embeddings

        batch, hw, _ = x.shape

        # Q from image, K/V from text
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head
        q = q.reshape(batch, hw, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, hw, -1)

        return self.to_out(out)

# U-Net에 Cross-Attention 추가
class TextConditionedUNet(nn.Module):
    def __init__(self, ...):
        # ...
        self.cross_attention = CrossAttention(
            query_dim=d_model,
            context_dim=768  # CLIP embedding dim
        )

    def forward(self, x, t, text_embeddings):
        # ... encoder ...

        # Cross-attention to text
        h = h.reshape(batch, -1, channels)  # (batch, H*W, channels)
        h = self.cross_attention(h, text_embeddings)
        h = h.reshape(batch, channels, H, W)

        # ... decoder ...
```

### CFG with Text

```python
@torch.no_grad()
def text_to_image(
    model,
    prompt,
    negative_prompt="",
    guidance_scale=7.5,
    num_steps=50
):
    # Encode prompts
    text_emb = encode_text(prompt)
    uncond_emb = encode_text(negative_prompt)  # Usually ""

    x = torch.randn(1, 3, 512, 512)

    for t in reversed(range(num_steps)):
        # Unconditional
        noise_uncond = model(x, t, uncond_emb)

        # Conditional
        noise_cond = model(x, t, text_emb)

        # CFG
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        x = denoise_step(x, t, noise_pred)

    return x

# 사용
image = text_to_image(
    model,
    prompt="A cat wearing a hat, digital art",
    negative_prompt="blurry, low quality",
    guidance_scale=7.5
)
```

---

## 🎨 4. Advanced Techniques

### 4.1 ControlNet

**아이디어**: 구조적 조건 추가 (pose, edge, depth)

```python
# U-Net을 복사하여 control branch 생성
control_net = copy.deepcopy(unet.encoder)

# Input: image + control (e.g., canny edge)
control_features = control_net(control_image, t)

# U-Net의 각 layer에 control 더하기
for i, (unet_feat, ctrl_feat) in enumerate(zip(unet_features, control_features)):
    unet_features[i] = unet_feat + ctrl_feat

# 결과: Edge를 따르는 이미지 생성!
```

### 4.2 Inpainting

**마스크 영역만 다시 생성**

```python
def inpaint(model, image, mask, prompt, num_steps=50):
    """
    image: 원본 이미지
    mask: 1 = 생성, 0 = 유지
    """
    x = torch.randn_like(image)

    for t in reversed(range(num_steps)):
        # Denoise
        noise_pred = model(x, t, prompt)
        x_denoised = denoise_step(x, t, noise_pred)

        # Apply mask
        x = mask * x_denoised + (1 - mask) * image

    return x
```

---

## 🎓 학습 목표

- [ ] DDIM 이해 및 구현 (50 steps vs 1000 steps)
- [ ] Classifier-free guidance 이해
- [ ] Text conditioning with CLIP embeddings
- [ ] Cross-attention 구현
- [ ] CFG sampling 구현
- [ ] Guidance scale의 영향 이해

---

## 📊 Guidance Scale 비교

```python
w = 1.0:  "A cat" → 🐱 (다양하지만 약간 모호)
w = 5.0:  "A cat" → 🐱 (명확한 고양이)
w = 10.0: "A cat" → 🐱 (매우 명확, 덜 다양, 과포화)

# Stable Diffusion 기본값: 7.5
```

---

## 💡 실전: HuggingFace Diffusers

```python
from diffusers import StableDiffusionPipeline

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

# Generate
image = pipe(
    prompt="A cat wearing a wizard hat, digital art, trending on artstation",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=50,  # DDIM steps
    guidance_scale=7.5,      # CFG scale
    height=512,
    width=512
).images[0]

image.save("cat_wizard.png")
```

### ControlNet

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import cv2

# Load ControlNet
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet
).to("cuda")

# Canny edge
image = Image.open("input.jpg")
image_np = np.array(image)
edges = cv2.Canny(image_np, 100, 200)
edges = Image.fromarray(edges)

# Generate
output = pipe(
    prompt="A beautiful painting",
    image=edges,
    num_inference_steps=50
).images[0]
```

---

## 📚 참고 논문

- **DDIM**: Denoising Diffusion Implicit Models (Song et al., 2021)
  - [Paper](https://arxiv.org/abs/2010.02502)

- **Classifier-Free Guidance**: Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)
  - [Paper](https://arxiv.org/abs/2207.12598)

- **ControlNet**: Adding Conditional Control to Text-to-Image Diffusion Models (Zhang et al., 2023)
  - [Paper](https://arxiv.org/abs/2302.05543)

---

## ⏭️ 다음 단계

Phase 3 완료! 🎉

**Diffusion Models를 완전히 마스터했습니다!**

👉 [Phase 4: GAN](../phase4-gan/)로 진행하여 또 다른 생성 모델을 배워봅시다!

**"Diffusion으로 이미지 생성, 이제 GAN으로!"** 🎨
