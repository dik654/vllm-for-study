# Phase 3.5: Video & Audio Generation

## 🎯 왜 이 Phase가 필요한가?

**"이미지 생성을 넘어, 시간축을 다루다!"**

```
이미지 생성 (Stable Diffusion):
  → (height, width, channels)
  → 2D spatial

영상 생성:
  → (time, height, width, channels)
  → 3D: 2D spatial + 1D temporal
  → 훨씬 복잡!

음악 생성:
  → (time, frequency)
  → Waveform or spectrogram
  → 완전히 다른 modality!
```

### ❓ 실사용 사례

**Video Generation**
- Sora (OpenAI): 텍스트 → 고품질 1분 영상
- Runway Gen-2: 영화 제작, 광고
- Pika Labs: 소셜 미디어 콘텐츠

**Audio/Music Generation**
- MusicGen (Meta): 텍스트 → 음악
- AudioLDM: 사운드 이펙트 생성
- Stable Audio: 고품질 음악 생성

---

## 📚 커리큘럼 (1.5주, 60시간)

### Week 1: Video Generation

#### [Day 1-3: Video Diffusion Models](./01-video-generation.md)

**핵심 개념**

영상 = 연속된 프레임들
```python
# Naive: 각 프레임을 독립적으로 생성
for i in range(num_frames):
    frame_i = diffusion_model.generate(prompt)
# 문제: 프레임 간 일관성 없음!

# Video Diffusion: temporal consistency 유지
video = video_diffusion_model.generate(prompt)
# → 부드러운 움직임!
```

**주요 모델**

1. **Stable Video Diffusion (SVD)**
   - Stability AI
   - 이미지 → 비디오 (image-to-video)
   - 14-25 프레임 생성

2. **AnimateDiff**
   - Stable Diffusion에 temporal layer 추가
   - Motion module로 애니메이션
   - LoRA와 결합 가능

3. **Sora (OpenAI)**
   - 텍스트 → 1분 영상
   - Diffusion Transformer (DiT) 기반
   - World simulator처럼 작동

4. **Runway Gen-2**
   - Text/image-to-video
   - 상업용으로 최적화
   - 4초 clips

**기술적 챌린지**

```
1. Temporal Consistency
   - 프레임 간 일관성 유지
   - Motion smoothness

2. Computational Cost
   - Video = 30 FPS × 10초 = 300 프레임!
   - Diffusion steps × 300 = 엄청난 계산량

3. Long-term Coherence
   - 긴 영상에서 일관성 유지
   - Object tracking
```

💻 **실습**:
- Stable Video Diffusion으로 image-to-video
- AnimateDiff로 애니메이션 생성
- Temporal layer 구현 이해

---

### Week 2: Audio & Music Generation

#### [Day 4-6: Audio/Music Generation](./02-audio-music-generation.md)

**핵심 개념**

**Audio Representations**
```python
# 1. Raw waveform
# → 직접 생성 (어려움)
waveform: (channels, samples)  # 44100 samples/sec

# 2. Spectrogram
# → 시각화, 처리 쉬움
spectrogram: (time, frequency)

# 3. Latent space (best!)
# → Diffusion in latent space
latent: (time_compressed, latent_dim)
```

**주요 모델**

1. **MusicGen (Meta)**
   - 텍스트 → 음악
   - Transformer + EnCodec
   - 30초 고품질 음악

2. **AudioLDM**
   - Latent Diffusion for Audio
   - Text/image → sound effects
   - Environmental sounds

3. **Stable Audio (Stability AI)**
   - 텍스트 → 음악 (최대 3분)
   - Latent diffusion
   - 상업용 품질

4. **Jukebox (OpenAI)**
   - VQ-VAE + Transformer
   - 가사까지 생성
   - 장르 제어 가능

**Audio Tokenization**

```python
# EnCodec (Meta)
from encodec import EncodecModel

model = EncodecModel.encodec_model_24khz()
audio_tensor = ...  # (batch, channels, samples)

# Encode
encoded = model.encode(audio_tensor)
# → Discrete codes (like tokenizing text!)

# Decode
reconstructed = model.decode(encoded)
# → High quality reconstruction
```

💻 **실습**:
- MusicGen으로 텍스트 → 음악
- AudioLDM으로 사운드 이펙트
- Spectrogram ↔ Waveform 변환 이해
- EnCodec tokenization

---

#### [Day 7-8: Multi-modal & Advanced Topics](./03-multimodal.md)

**ImageBind (Meta)**

**핵심**: 모든 modality를 하나의 embedding space에!

```python
# 6 modalities in one space:
# - Image
# - Text  
# - Audio
# - Video
# - Depth
# - IMU (motion)

from imagebind import imagebind_model
from imagebind.models.imagebind_model import ModalityType

model = imagebind_model.imagebind_huge()

# Embed different modalities
embeddings = model({
    ModalityType.TEXT: ["A dog playing"],
    ModalityType.VISION: [dog_image],
    ModalityType.AUDIO: [dog_audio],
})

# They're in the same space!
# → Cross-modal retrieval, generation
```

**응용**

1. **Audio-driven Video Generation**
   - 음악 → 뮤직비디오
   
2. **Text → Audio → Video**
   - 텍스트 → 사운드 생성 → 사운드에 맞는 영상

3. **Zero-shot Classification**
   - 모든 modality 간 유사도 계산 가능

💻 **실습**:
- ImageBind로 cross-modal retrieval
- Audio + Image → Video generation
- Multi-modal embedding 시각화

---

## 🎓 학습 목표

### 이론 이해
- [ ] Video diffusion의 temporal layer 이해
- [ ] Audio representation (waveform, spectrogram, latent) 비교
- [ ] EnCodec tokenization 원리
- [ ] ImageBind의 unified embedding space 이해

### 실습 완료
- [ ] Stable Video Diffusion 실행
- [ ] AnimateDiff로 애니메이션 생성
- [ ] MusicGen으로 음악 생성
- [ ] AudioLDM으로 사운드 이펙트 생성
- [ ] ImageBind로 cross-modal 작업

### 최신 모델 이해
- [ ] Sora의 작동 원리 (DiT, world model)
- [ ] Runway Gen-2의 상업화 전략
- [ ] Stable Audio의 긴 음악 생성 방법

---

## 📊 Timeline: Video/Audio Generation 역사

```
2020: Jukebox (OpenAI)
  → VQ-VAE + Transformer
  → 음악 생성 가능성 제시

2021: Make-A-Video (Meta)
  → Text-to-video diffusion
  → Temporal layers

2022: AudioLDM
  → Latent diffusion for audio
  → Text-to-sound

2023: 
  - MusicGen (Meta)
  - Stable Video Diffusion
  - ImageBind
  - AnimateDiff

2024:
  - Sora (OpenAI) - 게임 체인저!
  - Stable Audio
  - Runway Gen-2 commercial

→ 이제 실사용 단계!
```

---

## 🔥 실전 응용

### 1. 영상 제작 파이프라인

```python
# 1. 스토리보드 (텍스트)
scenes = [
    "A serene sunset over ocean",
    "Waves crashing on beach",
    "Seagulls flying"
]

# 2. 각 장면 생성
videos = []
for scene_text in scenes:
    video = sora.generate(scene_text, duration=4)
    videos.append(video)

# 3. 음악 생성
music = musicgen.generate("Calm, cinematic, ocean theme")

# 4. 결합
final_video = combine_videos(videos, background_music=music)
```

### 2. 게임 Asset 생성

```python
# Sound effects
footstep_sound = audioldm.generate("Footsteps on wooden floor")
explosion_sound = audioldm.generate("Large explosion")

# Animations
walk_animation = animatediff.generate("Character walking cycle")
```

### 3. 음악 제작

```python
# Base track
base = musicgen.generate("Lo-fi hip hop beat, 120 BPM")

# Melody
melody = musicgen.generate("Piano melody, jazz chords", continuation_of=base)

# Final mix
final = mix_tracks([base, melody])
```

---

## 📚 필수 논문

### Video Generation
- **Video Diffusion Models** (Ho et al., 2022)
- **Align your Latents** (SVD, Stability AI, 2023)
- **AnimateDiff** (Guo et al., 2023)
- **Sora Technical Report** (OpenAI, 2024)

### Audio/Music Generation
- **Jukebox** (Dhariwal et al., 2020)
- **AudioLDM** (Liu et al., 2023)
- **MusicGen** (Copet et al., 2023)
- **Stable Audio** (Evans et al., 2024)

### Multi-modal
- **ImageBind** (Girdhar et al., 2023)
- **CLIP** (Radford et al., 2021) - 기반 기술

---

## 🎯 완료 기준

이 Phase를 완료하면:
- ✅ Video diffusion 모델 작동 원리 이해
- ✅ Sora, Runway 등 최신 모델 이해
- ✅ 텍스트로 음악 생성 가능
- ✅ Audio tokenization (EnCodec) 이해
- ✅ Multi-modal embedding space 이해
- ✅ 실제로 영상/음악 생성 프로젝트 완수

---

## ⏭️ 다음 단계

Phase 3.5를 완료했다면, 모든 생성 AI를 마스터했습니다!
- ✅ 이미지 (Stable Diffusion, GAN)
- ✅ 영상 (Sora, AnimateDiff)
- ✅ 음악/오디오 (MusicGen, AudioLDM)

👉 [Phase 4: GAN](../phase4-gan/)으로 돌아가거나
👉 [Phase 5: Modern Techniques](../phase5-modern-techniques/)로 진행하세요!

**"이제 모든 미디어를 생성할 수 있습니다!"** 🎬🎵
