# Day 1-3: Video Generation

## 🎯 목표

**이미지 생성을 넘어, 시간축을 다루는 영상 생성!**

```python
# 이미지: (H, W, C)
image = diffusion_model.generate("A cat")

# 영상: (T, H, W, C)  
video = video_diffusion_model.generate("A cat walking")
# → Temporal consistency가 핵심!
```

---

## 📖 핵심 개념

### 왜 Video Generation이 어려운가?
## 🚀 기술 발전의 역사

### 2014-2017: GAN 기반 초기 시도

**VGAN (2016)**
```python
# Generator: 3D convolutions
class VideoGAN(nn.Module):
    def __init__(self):
        # Noise → 3D conv → Video
        self.gen = nn.Sequential(
            nn.ConvTranspose3d(100, 512, 4, 1, 0),  # (T, H, W)
            nn.ConvTranspose3d(512, 256, 4, 2, 1),
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.ConvTranspose3d(128, 3, 4, 2, 1),
        )

# 문제점:
# - 매우 짧은 영상 (16프레임, 1초 미만)
# - 낮은 해상도 (64x64)
# - 훈련 불안정 (GAN 특성상)
# - Temporal coherence 부족
```

**왜 실패?**
- GAN discriminator가 long-term dependency 학습 어려움
- 3D conv는 computational cost 너무 높음

---

### 2018-2020: Two-stream + Recurrent Models

**DVD-GAN (2019)**

**핵심 혁신**: Spatial + Temporal discriminator 분리
```python
class DVDGAN:
    def __init__(self):
        # Spatial discriminator (각 프레임)
        self.D_spatial = Discriminator2D()
        
        # Temporal discriminator (프레임 간)
        self.D_temporal = Discriminator3D()
    
    def discriminator_loss(self, fake_video, real_video):
        # 1. Spatial: 프레임이 realistic한가?
        loss_spatial = self.D_spatial(fake_video.reshape(-1, 3, H, W))
        
        # 2. Temporal: 움직임이 natural한가?
        loss_temporal = self.D_temporal(fake_video)
        
        return loss_spatial + loss_temporal
```

**개선점**: 128x128, 48 프레임 가능
**한계**: 여전히 GAN 불안정성, 짧은 영상

---

### 2021: VQ-VAE + Transformer (GODIVA)

**핵심 혁신**: Discrete latent space + Autoregressive

```python
# 1. VQ-VAE로 video → discrete codes
class VideoVQVAE:
    def encode(self, video):
        # Spatial compression
        z = self.encoder(video)  # (T, H/8, W/8, C)
        
        # Quantize (continuous → discrete)
        codes = self.codebook.quantize(z)  # (T, H/8, W/8) integers
        return codes

# 2. Transformer로 codes 생성
class GODIVA(nn.Module):
    def generate(self, text):
        text_tokens = tokenize(text)
        
        # Auto-regressive generation
        video_codes = []
        for t in range(num_frames):
            for h in range(H_codes):
                for w in range(W_codes):
                    next_code = self.transformer(
                        torch.cat([text_tokens, video_codes])
                    )
                    video_codes.append(next_code)
        
        # Decode
        video = self.vqvae.decode(video_codes)
        return video
```

**개선점**: 
- Text conditioning 가능
- 더 긴 영상 (수 초)

**한계**:
- Autoregressive → 매우 느림
- Discrete bottleneck → 품질 한계
- 여전히 저해상도

---

### 2022: Video Diffusion - 패러다임 전환!

**Make-A-Video (Meta)**

**핵심 혁신 #1**: Spatial + Temporal 분리

```python
class MakeAVideoModel:
    def __init__(self):
        # Pre-trained image diffusion (Spatial)
        self.spatial_layers = StableDiffusion()  # Frozen!
        
        # NEW: Temporal layers (학습 필요)
        self.temporal_layers = TemporalAttention()
    
    def forward(self, zt, t, text):
        # 1. Spatial (per-frame, pre-trained!)
        zt = self.spatial_layers(zt, t, text)
        
        # 2. Temporal (across frames, new!)
        zt = self.temporal_layers(zt)
        
        return zt
```

**왜 획기적?**
```
기존 이미지 diffusion 재사용!
→ Billions of images로 학습된 지식 활용
→ Temporal layers만 video data로 학습

결과:
- 훨씬 적은 video data 필요
- 더 높은 품질 (이미지 모델 덕분)
- 더 빠른 수렴
```

**핵심 혁신 #2**: Frame interpolation

```python
# Low frame rate로 생성 → Interpolation으로 up-sampling
def generate_video(text):
    # 1. Generate keyframes (8 FPS)
    keyframes = diffusion_model.generate(text, num_frames=8)
    
    # 2. Interpolate (8 FPS → 24 FPS)
    full_video = frame_interpolation_model(keyframes)
    
    return full_video

# 3배 적은 계산량으로 같은 결과!
```

---

### 2023: Latent Video Diffusion - 효율성 극대화

**Stable Video Diffusion**

**핵신 혁신**: Latent space에서 diffusion!

```python
# Imagen Video: Pixel space diffusion
for t in timesteps:
    video_pixels = unet(video_pixels, t)  # (T, 3, H, W)
# → 메모리/속도 문제!

# SVD: Latent space diffusion
# 1. Encode to latent
z = vae.encode(video)  # (T, 3, H, W) → (T, C, H/8, W/8)

# 2. Diffusion in latent (8x8 = 64배 작음!)
for t in timesteps:
    z = unet(z, t)

# 3. Decode
video = vae.decode(z)
```

**결과**:
- 메모리: 64배 감소
- 속도: 10배 향상
- 해상도: 1024x576 가능!

---

### 2024: Sora - World Simulator

**핵심 혁신 #1**: Diffusion Transformer (DiT)

```python
# 기존: U-Net (CNN 기반)
class UNet:
    # 고정된 downsampling/upsampling
    # → 해상도에 제약

# Sora: Transformer
class DiffusionTransformer:
    # Fully attention-based
    # → 임의의 해상도/duration 가능!
```

**핵심 혁신 #2**: Spacetime Patches

```python
# 기존: (T, H, W, C) → separate처리
# Spatial: (H, W)
# Temporal: T

# Sora: Unified spacetime patches
def spacetime_patchify(video):
    # (T, H, W, C) → patches
    patches = video.reshape(
        num_patches_t, patch_size_t,
        num_patches_h, patch_size_h,
        num_patches_w, patch_size_w,
        C
    )
    
    # Flatten: (num_patches, patch_size_t * patch_size_h * patch_size_w * C)
    patches = patches.reshape(num_patches, -1)
    
    # 이제 patches = "tokens"!
    # → Transformer로 직접 처리
    return patches

# 장점:
# - Space와 time을 동등하게 처리
# - Long-range dependency (temporal + spatial 모두)
# - Flexible resolution/duration
```

**핵심 혁신 #3**: Native resolution training

```python
# 기존: 모든 video를 256x256으로 resize
# → Aspect ratio 깨짐, 정보 손실

# Sora: Native resolution 유지
videos = [
    (16, 1920, 1080),  # 16:9
    (24, 1080, 1920),  # 9:16 (vertical)
    (32, 1024, 1024),  # 1:1 (square)
]

# Patch로 만들면 다 같은 방식으로 처리 가능!
for video in videos:
    patches = spacetime_patchify(video)
    # Transformer는 sequence length만 다를 뿐, 동일하게 처리
```

**결과**:
- 1분 영상 생성 가능
- 1080p 해상도
- 다양한 aspect ratios
- 놀라운 temporal consistency
- 물리 법칙 이해 (world model처럼!)

---

## 🔬 발전의 핵심 패턴

### 패턴 1: Pre-training 활용

```
2016: Video GAN (scratch 훈련)
  → 실패 (데이터 부족)

2022: Make-A-Video (이미지 모델 재사용)
  → 성공!
  
교훈: Billions of images로 학습된 지식을 활용하라!
```

### 패턴 2: Representation Learning

```
2016: Raw pixels (H×W×T)
  → 너무 high-dimensional

2021: VQ-VAE codes (discrete)
  → Autoregressive 느림

2023: Latent diffusion (continuous latent)
  → 최적의 balance!
  
교훈: 적절한 representation이 핵심!
```

### 패턴 3: Scaling Laws

```
Model Size:
2019: ~100M parameters
2022: ~1B parameters  
2024: ~10B+ parameters (Sora 추정)

Data:
2019: 수천 videos
2022: 수십만 videos
2024: 수백만 videos

결과: 
Size ↑, Data ↑ → Quality ↑ (일관되게!)
```

---

## 📈 성능 비교 (발전 추이)

| Year | Model | Duration | Resolution | FVD ↓ (낮을수록 좋음) |
|------|-------|----------|------------|----------------------|
| 2016 | VGAN | 0.5s | 64x64 | ~1000 |
| 2019 | DVD-GAN | 1.5s | 128x128 | ~500 |
| 2021 | GODIVA | 3s | 128x128 | ~300 |
| 2022 | Make-A-Video | 5s | 256x256 | ~100 |
| 2023 | SVD | 3s | 1024x576 | ~50 |
| 2024 | Sora | 60s | 1920x1080 | ~20 (추정) |

**8년간 50배 성능 향상!**

**1. Temporal Consistency**
```python
# Naive: 프레임별로 독립 생성
frames = []
for t in range(30):  # 1 second at 30 FPS
    frame = image_model.generate("A cat walking")
    frames.append(frame)

# 문제:
# - 프레임 간 고양이 위치가 점프
# - 움직임이 부자연스러움
# - 배경이 계속 바뀜
```

**2. Computational Cost**
```
1초 영상 (30 FPS) = 30 프레임
10초 영상 = 300 프레임!

Diffusion 50 steps × 300 프레임 = 15,000 forward passes
→ 엄청난 계산량!
```

**3. Data Scarcity**
```
이미지: LAION-5B (50억 개)
영상: WebVid-10M (1000만 개)

→ 영상 데이터가 500배 적음!
```

---

## 🎬 주요 모델들

### 1. Stable Video Diffusion (SVD)

**Stability AI, 2023**

**핵심**: Image-to-Video
- 단일 이미지를 입력으로 받아 짧은 비디오 생성
- 14 or 25 프레임 (약 2초)
- 576x1024 해상도

**아키텍처**

```python
# Stable Diffusion + Temporal Layers
class SVDModel(nn.Module):
    def __init__(self):
        # Spatial layers (from Stable Diffusion)
        self.spatial_transformer = SpatialTransformer()
        
        # NEW: Temporal layers
        self.temporal_transformer = TemporalTransformer()
    
    def forward(self, x):
        # x: (batch, frames, channels, height, width)
        batch, frames, c, h, w = x.shape
        
        # 1. Spatial processing (per-frame)
        x_reshaped = x.view(batch * frames, c, h, w)
        x_spatial = self.spatial_transformer(x_reshaped)
        x_spatial = x_spatial.view(batch, frames, c, h, w)
        
        # 2. Temporal processing (across frames)
        x_temporal = self.temporal_transformer(x_spatial)
        
        return x_temporal

class TemporalTransformer(nn.Module):
    def forward(self, x):
        # x: (batch, frames, channels, h, w)
        batch, frames, c, h, w = x.shape
        
        # Reshape for attention across time
        x = x.permute(0, 2, 3, 4, 1)  # (batch, c, h, w, frames)
        x = x.reshape(batch * c * h * w, frames, 1)
        
        # Temporal attention
        attn_output = self.temporal_attention(x, x, x)
        
        # Reshape back
        attn_output = attn_output.reshape(batch, c, h, w, frames)
        attn_output = attn_output.permute(0, 4, 1, 2, 3)
        
        return attn_output
```

**사용 예시**

```python
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

# Load pipeline
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# Load conditioning image
image = load_image("path/to/image.png")
image = image.resize((1024, 576))

# Generate video
frames = pipe(
    image,
    num_frames=25,
    decode_chunk_size=8,  # Memory optimization
    motion_bucket_id=127,  # Controls motion amount (0-255)
    noise_aug_strength=0.02
).frames[0]

# Export
export_to_video(frames, "output.mp4", fps=7)
```

**Motion Bucket**
```python
# Low motion (static scenes)
frames = pipe(image, motion_bucket_id=50)

# Medium motion
frames = pipe(image, motion_bucket_id=127)

# High motion (dynamic scenes)
frames = pipe(image, motion_bucket_id=200)
```

---

### 2. AnimateDiff

**Community-driven, 2023**

**핵심**: Stable Diffusion + Motion Module
- 기존 SD 모델에 motion module 추가
- LoRA와 호환
- 커뮤니티 모델 활용 가능

**Motion Module**

```python
class MotionModule(nn.Module):
    """Temporal layers for animation"""
    def __init__(self, in_channels, temporal_position_encoding_max_len=32):
        super().__init__()
        
        # Temporal position encoding
        self.pos_encoder = PositionalEncoding(temporal_position_encoding_max_len)
        
        # Temporal attention
        self.temporal_transformer = TemporalTransformerBlock(
            dim=in_channels,
            num_attention_heads=8,
            attention_head_dim=in_channels // 8,
        )
    
    def forward(self, hidden_states, num_frames):
        # hidden_states: (batch*frames, channels, h, w)
        batch_frames, channels, height, width = hidden_states.shape
        batch = batch_frames // num_frames
        
        # Reshape to separate frames
        hidden_states = hidden_states.view(batch, num_frames, channels, height, width)
        
        # Add positional encoding
        hidden_states = self.pos_encoder(hidden_states)
        
        # Temporal attention
        hidden_states = self.temporal_transformer(hidden_states)
        
        # Reshape back
        hidden_states = hidden_states.view(batch_frames, channels, height, width)
        
        return hidden_states
```

**사용 예시**

```python
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_video

# Load motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5")

# Load base model (any SD 1.5 model!)
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# Generate
output = pipe(
    prompt="A cat walking on the beach, sunset, photorealistic",
    negative_prompt="blur, distorted",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
)

export_to_video(output.frames[0], "cat_walking.mp4")
```

**LoRA 결합**

```python
# Load custom LoRA for style
pipe.load_lora_weights("path/to/anime-style-lora.safetensors")

# Generate with custom style
output = pipe(
    prompt="A girl dancing, anime style",
    num_frames=16
)
```

---

### 3. Sora (OpenAI)

**2024년, 게임 체인저!**

**주요 특징**
- 텍스트 → 최대 1분 영상
- 해상도: 최대 1080p
- 다양한 aspect ratios
- Long-term temporal consistency

**아키텍처 (추정)**

```python
# Diffusion Transformer (DiT) 기반
class SoraModel(nn.Module):
    """
    추정 아키텍처:
    1. Video VAE (spacetime compression)
    2. Diffusion Transformer
    3. Conditioning (text, camera control)
    """
    
    def __init__(self):
        # 1. Spacetime VAE
        self.vae = SpacetimeVAE(
            spatial_compression=8,
            temporal_compression=4
        )
        
        # 2. Diffusion Transformer (DiT)
        self.transformer = DiffusionTransformer(
            num_layers=32,
            hidden_size=1280,
            num_heads=16,
        )
        
        # 3. Text encoder
        self.text_encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")
    
    def forward(self, video, text, timesteps):
        # Encode video to latent
        # (T, H, W, C) → (T/4, H/8, W/8, latent_dim)
        z = self.vae.encode(video)
        
        # Text conditioning
        text_embed = self.text_encoder(text)
        
        # Diffusion process
        noise_pred = self.transformer(z, timesteps, text_embed)
        
        return noise_pred
```

**Patch-based Representation**

```python
# Sora는 video를 spacetime patches로 분할
def patchify_video(video, patch_size=(2, 16, 16)):
    """
    video: (T, H, W, C)
    patch_size: (temporal, spatial_h, spatial_w)
    """
    T, H, W, C = video.shape
    t_patch, h_patch, w_patch = patch_size
    
    # Divide into patches
    patches = video.reshape(
        T // t_patch, t_patch,
        H // h_patch, h_patch,
        W // w_patch, w_patch,
        C
    )
    
    # Flatten patches
    patches = patches.permute(0, 2, 4, 1, 3, 5, 6)
    patches = patches.reshape(-1, t_patch * h_patch * w_patch * C)
    
    # Now patches can be treated as "tokens" for Transformer!
    return patches
```

**Why Sora is powerful?**

1. **Variable resolutions & durations**
   - Native training at multiple resolutions
   - 1초 ~ 1분 영상

2. **Camera control**
   ```python
   prompt = """
   A drone shot of a futuristic city.
   Camera: aerial view, slowly panning right, 
   descending from 100m to 50m altitude.
   """
   # → Sora가 카메라 움직임도 이해!
   ```

3. **World simulation**
   - 물리 법칙 이해 (중력, 충돌)
   - Object permanence (물체가 가려져도 다시 나타남)

---

### 4. Runway Gen-2

**Commercial-focused, 2023**

**특징**
- Text-to-video
- Image-to-video
- Video-to-video (style transfer)
- 4초 clips (commercial 길이)

**사용 (API)**

```python
import requests

API_KEY = "your_runway_api_key"

# Text-to-video
response = requests.post(
    "https://api.runwayml.com/v1/generate",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "gen2",
        "prompt": "A sunset over mountains, cinematic",
        "duration": 4,
        "resolution": "1280x768"
    }
)

video_url = response.json()["url"]
```

---

## 🔧 실습: Temporal Layer 구현

### Simple Temporal Attention

```python
import torch
import torch.nn as nn

class SimpleTemporalAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        
        # Q, K, V projections
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        
        # Output projection
        self.to_out = nn.Linear(channels, channels)
    
    def forward(self, x):
        """
        x: (batch, frames, channels, height, width)
        """
        batch, frames, channels, h, w = x.shape
        
        # Reshape: treat (h, w) as sequence length
        x = x.permute(0, 3, 4, 1, 2)  # (batch, h, w, frames, channels)
        x = x.reshape(batch * h * w, frames, channels)
        
        # Attention across frames (for each spatial position)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Multi-head attention
        q = q.view(batch * h * w, frames, self.num_heads, channels // self.num_heads).transpose(1, 2)
        k = k.view(batch * h * w, frames, self.num_heads, channels // self.num_heads).transpose(1, 2)
        v = v.view(batch * h * w, frames, self.num_heads, channels // self.num_heads).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (channels // self.num_heads) ** 0.5
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch * h * w, frames, channels)
        
        # Output projection
        out = self.to_out(out)
        
        # Reshape back
        out = out.view(batch, h, w, frames, channels)
        out = out.permute(0, 3, 4, 1, 2)  # (batch, frames, channels, h, w)
        
        return out

# Test
temporal_attn = SimpleTemporalAttention(channels=512)
video_latent = torch.randn(2, 16, 512, 32, 32)  # (batch, frames, channels, h, w)
output = temporal_attn(video_latent)
print(output.shape)  # (2, 16, 512, 32, 32)
```

---

## 📊 모델 비교

| Model | Input | Output Duration | Resolution | Key Feature |
|-------|-------|-----------------|------------|-------------|
| SVD | Image | 2-3 sec | 1024x576 | Image-to-video |
| AnimateDiff | Text | 1-2 sec | 512x512 | SD compatible |
| Sora | Text | Up to 60 sec | Up to 1080p | Long-term consistency |
| Runway Gen-2 | Text/Image | 4 sec | 1280x768 | Commercial quality |

---

## 🎓 학습 목표 체크리스트

- [ ] Video diffusion의 temporal layer 이해
- [ ] SVD로 image-to-video 생성
- [ ] AnimateDiff로 텍스트 기반 애니메이션
- [ ] Sora의 spacetime patches 개념 이해
- [ ] Temporal attention 구현

---

## 📚 참고 자료

- [Stable Video Diffusion Paper](https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets)
- [AnimateDiff](https://github.com/guoyww/AnimateDiff)
- [Sora Technical Report](https://openai.com/research/video-generation-models-as-world-simulators)

---

## ⏭️ 다음

영상 생성을 마스터했습니다!

👉 [Day 4-6: Audio/Music Generation](./02-audio-music-generation.md)에서 **소리를 생성**합니다!

**"이제 영상을 만들 수 있습니다!"** 🎬
