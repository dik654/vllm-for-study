# Day 7-8: Multi-modal & Controllable Generation

## 🎯 목표

**모든 modality를 연결하고 정교하게 제어하기!**

```python
# Text → Image (basic)
image = sd.generate("A dog")

# Image + Pose → Image (controlled!)
image = controlnet.generate("A dog", pose=pose_skeleton)

# Humming → Music (controlled!)
music = musicgen.generate_with_melody(humming_audio)

# Image → Music → Video (multi-modal!)
music = image_to_music(image)
video = music_to_video(music)
```

---

## 🎨 Part 1: 이미지/영상 제어

### ControlNet - 정교한 제어의 핵심

**문제**: Text만으로는 정확한 제어 어려움
```python
# "A person standing with arms raised"
# → 어떤 자세? 어디서? 어떤 각도?
```

**해결**: ControlNet (Zhang et al., 2023)

**핵심 아이디어**: Spatial control + Text

```python
class ControlNet(nn.Module):
    """
    Stable Diffusion에 추가 conditioning 제공
    """
    def __init__(self, sd_unet):
        # SD U-Net의 encoder를 복사 (frozen SD는 그대로!)
        self.control_net = copy_encoder(sd_unet)
        
        # Zero convolutions (학습 초기 영향 0)
        self.zero_convs = nn.ModuleList([
            zero_module(nn.Conv2d(...)) for _ in range(num_layers)
        ])
    
    def forward(self, x, hint, timestep, text_embed):
        # hint: conditioning image (pose, depth, canny, etc.)
        
        # ControlNet forward
        control_features = []
        h = torch.cat([x, hint], dim=1)  # Concatenate hint
        
        for layer, zero_conv in zip(self.control_net, self.zero_convs):
            h = layer(h, timestep, text_embed)
            # Zero conv로 feature 추출
            control_feat = zero_conv(h)
            control_features.append(control_feat)
        
        return control_features

# Stable Diffusion with ControlNet
class SDwithControl:
    def __init__(self):
        self.sd_unet = UNet()  # Frozen!
        self.controlnet = ControlNet(self.sd_unet)
    
    def forward(self, noisy_latent, t, text_embed, control_image):
        # ControlNet features
        control_feats = self.controlnet(
            noisy_latent, control_image, t, text_embed
        )
        
        # SD U-Net with injected control
        noise_pred = self.sd_unet(
            noisy_latent, t, text_embed,
            additional_residuals=control_feats  # Inject!
        )
        
        return noise_pred
```

**다양한 Control Types**

1. **Canny Edge (윤곽선)**
```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import cv2

# Load ControlNet
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet
)

# Detect edges
image = cv2.imread("input.jpg")
edges = cv2.Canny(image, 100, 200)

# Generate with edge control
output = pipe(
    prompt="A beautiful landscape, high quality",
    image=edges,
    num_inference_steps=20
).images[0]

# → 윤곽선 유지하면서 새로운 이미지 생성!
```

2. **Pose (자세 제어)**
```python
from controlnet_aux import OpenposeDetector

# Detect pose from image
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
pose_image = openpose(source_image)

# Generate with pose control
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
pipe = StableDiffusionControlNetPipeline.from_pretrained(..., controlnet=controlnet)

output = pipe(
    prompt="A superhero in action pose",
    image=pose_image
).images[0]

# → 정확히 같은 자세로 새로운 캐릭터 생성!
```

3. **Depth (깊이 제어)**
```python
from transformers import DPTForDepthEstimation

# Estimate depth
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
depth_map = depth_estimator(image)

# Generate with depth control
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
output = pipe(
    prompt="A futuristic city",
    image=depth_map
).images[0]

# → 같은 depth structure로 새로운 장면!
```

4. **Scribble (낙서로 제어!)**
```python
# 손그림 → 고품질 이미지
scribble = draw_scribble()  # MS Paint로 그린 낙서!

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
output = pipe(
    prompt="A fantasy castle",
    image=scribble
).images[0]

# → 낙서가 photorealistic 이미지로!
```

**Multi-ControlNet (여러 제어 동시 적용)**

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, MultiControlNetModel

# Multiple ControlNets
controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")

controlnet = MultiControlNetModel([controlnet_canny, controlnet_depth])

pipe = StableDiffusionControlNetPipeline.from_pretrained(..., controlnet=controlnet)

# Generate with both controls!
output = pipe(
    prompt="A castle",
    image=[canny_edges, depth_map],
    controlnet_conditioning_scale=[0.5, 0.8]  # Weights
).images[0]

# → Canny + Depth 동시에 만족!
```

---

### Image-to-Image (img2img)

**개념**: 기존 이미지를 리메이크

```python
class Img2ImgPipeline:
    def __call__(self, prompt, init_image, strength=0.8):
        # 1. Encode image to latent
        latent = self.vae.encode(init_image)
        
        # 2. Add noise (strength 조절)
        noise = torch.randn_like(latent)
        timestep = int(1000 * strength)  # 0.8 → 800
        noisy_latent = self.scheduler.add_noise(latent, noise, timestep)
        
        # 3. Denoise (partial denoising!)
        for t in reversed(range(0, timestep)):
            noise_pred = self.unet(noisy_latent, t, text_embed)
            noisy_latent = self.scheduler.step(noise_pred, t, noisy_latent)
        
        # 4. Decode
        image = self.vae.decode(noisy_latent)
        return image

# 사용
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(...)

# 사진 → 그림 스타일
output = pipe(
    prompt="Oil painting style",
    image=photo,
    strength=0.75  # 0=원본, 1=완전 새로 생성
).images[0]

# strength가 핵심!
# 0.3: 약간만 변경 (색감, 디테일)
# 0.7: 많이 변경 (스타일 완전 전환)
# 1.0: 거의 새로 생성
```

---

### Inpainting & Outpainting

**Inpainting (부분 편집)**

```python
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(...)

# 마스크: 1=수정할 부분, 0=유지
mask = create_mask(image, region="person's_face")

output = pipe(
    prompt="A dog face",
    image=image,
    mask_image=mask
).images[0]

# → 사람 얼굴만 강아지로 교체!
```

**Outpainting (이미지 확장)**

```python
# 이미지 크기 확장하고 빈 공간 채우기
extended_image = expand_canvas(image, new_size=(1024, 1024))
mask = create_outpainting_mask(original_size, new_size)

output = pipe(
    prompt="Continuation of the scene",
    image=extended_image,
    mask_image=mask
).images[0]

# → 이미지가 자연스럽게 확장!
```

---

## 🎵 Part 2: 음악 제어

### Melody Conditioning (허밍 → 음악)

**개념**: Melody는 유지하고 style만 변경

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained('facebook/musicgen-melody')

# 1. 허밍/노래 녹음
humming, sr = torchaudio.load("humming.wav")

# 2. Chroma features 추출 (음계 정보)
import librosa

# Chromagram: 12-dimensional (C, C#, D, ..., B)
chroma = librosa.feature.chroma_cqt(
    y=humming.numpy(),
    sr=sr,
    n_chroma=12
)
# Shape: (12, time_frames)

# 3. MusicGen으로 생성
music = model.generate_with_chroma(
    descriptions=["80s pop rock, energetic drums"],
    melody_wavs=humming,
    melody_sample_rate=sr,
    progress=True
)

# → 허밍 melody는 유지, style은 80s pop rock으로!
```

**내부 동작**

```python
class MusicGenWithMelody(MusicGen):
    def generate_with_chroma(self, text, melody_audio):
        # 1. Text embedding
        text_embed = self.text_encoder(text)
        
        # 2. Melody → Chroma
        chroma = extract_chroma(melody_audio)  # (12, T)
        
        # 3. Chroma embedding
        chroma_embed = self.chroma_encoder(chroma)  # (T, D)
        
        # 4. Combined conditioning
        combined = torch.cat([text_embed, chroma_embed], dim=-1)
        
        # 5. Generate with conditioning
        audio_codes = self.transformer.generate(
            max_length=duration_in_tokens,
            condition=combined
        )
        
        # 6. Decode
        audio = self.audio_decoder(audio_codes)
        return audio

# 핵심: Chroma가 pitch information만 유지
# → Rhythm, timbre는 자유롭게 생성!
```

**Chroma Features 시각화**

```python
import matplotlib.pyplot as plt

# Chromagram
plt.figure(figsize=(10, 4))
librosa.display.specshow(
    chroma,
    y_axis='chroma',
    x_axis='time',
    cmap='coolwarm'
)
plt.title('Chromagram')
plt.colorbar()

# 각 행 = 음계 (C, C#, D, ..., B)
# 밝은 부분 = 해당 음이 들림
```

---

### Music Style Transfer

**개념**: 기존 음악의 structure 유지하면서 style만 변경

```python
def music_style_transfer(original_audio, target_style):
    """
    Original: Jazz piano
    Target: Heavy metal
    → Jazz structure를 유지한 Heavy metal!
    """
    
    # 1. Structure 분석
    # - Tempo (BPM)
    # - Beat structure
    # - Harmonic content (chroma)
    
    tempo, beats = librosa.beat.beat_track(original_audio)
    chroma = librosa.feature.chroma_cqt(original_audio)
    
    # 2. Style transfer
    # Option A: MusicGen with conditioning
    output = musicgen.generate_with_chroma(
        descriptions=[f"{target_style}, {tempo} BPM"],
        melody_wavs=original_audio
    )
    
    # Option B: Timbre transfer (Tone Transfer)
    output = tone_transfer_model(
        original_audio,
        target_instrument="electric guitar"
    )
    
    return output

# 사용
jazz_piano = load_audio("jazz_piano.wav")
metal_version = music_style_transfer(jazz_piano, "heavy metal")

# → Same notes, different instruments!
```

---

### Stem Separation & Remix

**Stem Separation (악기 분리)**

```python
from demucs import pretrained
from demucs.apply import apply_model

# Load Demucs model
model = pretrained.get_model('htdemucs')

# Separate stems
sources = apply_model(
    model,
    audio_tensor,
    device='cuda'
)

# sources: dict with keys
# - 'drums'
# - 'bass'
# - 'vocals'
# - 'other' (guitar, piano, etc.)

drums = sources['drums']
vocals = sources['vocals']
bass = sources['bass']
other = sources['other']
```

**Remix with AI**

```python
# 1. Separate original song
stems = separate_stems(original_song)

# 2. Generate new drums
new_drums = musicgen.generate("Aggressive trap drums, 140 BPM")

# 3. Mix
remix = mix_stems({
    'vocals': stems['vocals'],
    'bass': stems['bass'],
    'drums': new_drums,  # AI-generated!
    'other': stems['other']
})

# → AI가 드럼만 새로 만듦!
```

---

## 🌐 Part 3: Multi-modal Embeddings

### ImageBind (Meta)

**핵심**: 6 modalities를 하나의 embedding space에!

```python
from imagebind import imagebind_model
from imagebind.models.imagebind_model import ModalityType

model = imagebind_model.imagebind_huge()

# Embed different modalities
inputs = {
    ModalityType.TEXT: ["A dog playing fetch"],
    ModalityType.VISION: [dog_image],
    ModalityType.AUDIO: [dog_bark_audio],
    ModalityType.THERMAL: [thermal_image],
    ModalityType.DEPTH: [depth_map],
    ModalityType.IMU: [motion_data],
}

embeddings = model(inputs)

# embeddings: 모든 modality가 같은 1024-dim space에!
text_embed = embeddings[ModalityType.TEXT]      # (1, 1024)
image_embed = embeddings[ModalityType.VISION]   # (1, 1024)
audio_embed = embeddings[ModalityType.AUDIO]    # (1, 1024)

# Similarity 계산 가능!
similarity = torch.cosine_similarity(text_embed, image_embed)
```

**How it works?**

```python
class ImageBind(nn.Module):
    def __init__(self):
        # Modality-specific encoders
        self.text_encoder = Transformer()
        self.image_encoder = VisionTransformer()
        self.audio_encoder = AudioTransformer()
        # ...
        
        # All project to same dimension
        self.projection_dim = 1024
    
    def forward(self, inputs):
        embeddings = {}
        
        # Encode each modality
        if ModalityType.TEXT in inputs:
            text_feats = self.text_encoder(inputs[ModalityType.TEXT])
            embeddings[ModalityType.TEXT] = self.project(text_feats)
        
        if ModalityType.VISION in inputs:
            img_feats = self.image_encoder(inputs[ModalityType.VISION])
            embeddings[ModalityType.VISION] = self.project(img_feats)
        
        # ...
        
        return embeddings

# Training: Contrastive learning
# Image-Text pairs: CLIP-style
# Image-Audio pairs: Match image with corresponding sound
# ...
# → 모든 modality가 align!
```

---

### Cross-modal Generation

**1. Image → Music**

```python
def image_to_music(image):
    # 1. Image → Caption
    caption = blip_model.generate_caption(image)
    # "A peaceful beach at sunset"
    
    # 2. Image → Mood
    mood = analyze_mood(image)  # {'calm': 0.9, 'happy': 0.7}
    
    # 3. Generate music
    prompt = f"{caption}, {mood['dominant']} mood, ambient music"
    music = musicgen.generate(prompt)
    
    return music

# Advanced: ImageBind embedding
def image_to_music_v2(image):
    # Image embedding
    img_embed = imagebind.encode_image(image)
    
    # Find closest audio in database
    audio_db_embeds = load_audio_embeddings()
    similarities = img_embed @ audio_db_embeds.T
    closest_audio_idx = similarities.argmax()
    
    reference_audio = load_audio(closest_audio_idx)
    
    # Generate similar music
    music = musicgen.generate_continuation(reference_audio)
    return music
```

**2. Audio → Video**

```python
def audio_to_video(audio):
    # 1. Audio analysis
    tempo, beats = librosa.beat.beat_track(audio)
    energy = librosa.feature.rms(audio).mean()
    
    # 2. Visual concept from audio
    # High energy → Fast movement
    # Low energy → Slow, calm
    
    if energy > 0.5:
        prompt = "Fast moving particles, energetic, abstract"
    else:
        prompt = "Calm waves, slow motion, peaceful"
    
    # 3. Generate video synced to audio
    video = sora.generate(
        prompt=prompt,
        audio_conditioning=audio,  # Beat-sync!
        duration=len(audio) / sample_rate
    )
    
    return video
```

**3. Text → Image + Music + Video (Complete pipeline)**

```python
def text_to_multimedia(text):
    # 1. Image
    image = stable_diffusion.generate(text)
    
    # 2. Music from image
    music = image_to_music(image)
    
    # 3. Video from image + music
    video = animatediff.generate(
        text=text,
        image_conditioning=image,
        audio_conditioning=music
    )
    
    return {
        'image': image,
        'music': music,
        'video': video
    }

# 사용
result = text_to_multimedia("A futuristic city at night")
# → Image, Music, Video 모두 일관성 있게 생성!
```

---

## 🎓 학습 목표 체크리스트

**이미지/영상 제어**
- [ ] ControlNet 원리 이해 (pose, depth, canny)
- [ ] Multi-ControlNet으로 복합 제어
- [ ] img2img로 이미지 리메이크
- [ ] Inpainting/Outpainting 활용

**음악 제어**
- [ ] Chroma features 이해
- [ ] Melody conditioning (허밍 → 음악)
- [ ] Music style transfer
- [ ] Stem separation & remix

**Multi-modal**
- [ ] ImageBind embedding space 이해
- [ ] Cross-modal similarity 계산
- [ ] Image → Music 생성
- [ ] Audio → Video 생성
- [ ] 전체 multi-modal pipeline 구현

---

## 🔥 실전 프로젝트 아이디어

1. **AI Music Video Generator**
   ```
   Input: Song lyrics
   → Generate music (MusicGen)
   → Generate key frames (Stable Diffusion)
   → Animate (AnimateDiff)
   → Sync with music
   ```

2. **Photo Animation Tool**
   ```
   Input: Photo + Pose sequence
   → ControlNet (pose) + AnimateDiff
   → Character moves according to pose!
   ```

3. **AI Remix Studio**
   ```
   Input: Original song
   → Separate stems (Demucs)
   → Generate new drums (MusicGen)
   → User-controlled mixing
   ```

---

## 📚 참고 자료

- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [ImageBind](https://github.com/facebookresearch/ImageBind)
- [MusicGen](https://github.com/facebookresearch/audiocraft)
- [Demucs (Stem Separation)](https://github.com/facebookresearch/demucs)

---

## 🎯 완료!

Phase 3.5를 완료했습니다! 이제:
- ✅ 영상 생성 (Sora, AnimateDiff)
- ✅ 음악 생성 (MusicGen, AudioLDM)
- ✅ 정교한 제어 (ControlNet, Melody conditioning)
- ✅ Multi-modal 연결 (ImageBind)

**모든 미디어를 자유자재로 생성하고 제어할 수 있습니다!** 🎬🎵🎨
