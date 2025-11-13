# Day 4-6: Audio & Music Generation

## 🎯 목표

**텍스트나 이미지로 음악과 사운드 생성!**

```python
# Text → Music
music = musicgen.generate("Lo-fi hip hop, chill, 120 BPM")

# Text → Sound Effect
sound = audioldm.generate("Thunder and rain")

# Image → Music (multi-modal!)
music = image_to_music(beach_photo)  # → Calm, ocean waves음악
```

---

## 📖 Audio Representations

### 1. Raw Waveform

```python
# 시간 영역 (time domain)
waveform: (channels, samples)

# CD quality
sampling_rate = 44100 Hz
duration = 10 seconds
## 🚀 음악 생성 기술의 발전

### 2016: WaveNet - Raw Waveform의 시작

**DeepMind WaveNet**

**핵심 혁신**: Dilated Causal Convolutions

```python
class WaveNet(nn.Module):
    def __init__(self):
        # Dilated convolutions for large receptive field
        self.dilated_convs = nn.ModuleList([
            CausalConv1d(dilation=1),      # receptive field: 1
            CausalConv1d(dilation=2),      # receptive field: 2  
            CausalConv1d(dilation=4),      # receptive field: 4
            CausalConv1d(dilation=8),      # receptive field: 8
            # ...
            CausalConv1d(dilation=512),    # receptive field: 512
        ])
    
    def forward(self, x):
        # Autoregressive: 한 sample씩 생성
        for conv in self.dilated_convs:
            x = conv(x)
        
        # Predict next sample
        return x

# 1초 생성 시간: ~90초 (!!!)
# → Real-time 불가능
```

**왜 느린가?**
```python
# 44100 samples/second
# 각 sample을 순차적으로 생성
for t in range(44100):
    next_sample = model.predict(waveform[:t])
    waveform.append(next_sample)

# O(T) complexity, T=44100 → 매우 느림!
```

**의의**: Raw waveform 생성 가능성 제시
**한계**: 실용성 없음 (속도)

---

### 2018-2019: Vocoder Revolution

**문제**: WaveNet은 느림
**해결**: 2-stage approach

```python
# Stage 1: Mel-spectrogram 생성 (빠름!)
mel_spec = tacotron2.generate(text)  # TTS

# Stage 2: Mel-spec → Waveform (Vocoder)
waveform = vocoder.generate(mel_spec)
```

**주요 Vocoders**:

1. **WaveGlow (NVIDIA, 2018)**: Flow-based
```python
# Invertible transformations
z = flow_model.forward(waveform)  # Waveform → Latent
waveform = flow_model.inverse(z)   # Latent → Waveform

# 장점: Parallel generation (빠름!)
# WaveNet 대비 1000배 빠름!
```

2. **MelGAN (2019)**: GAN-based
```python
# Generator: Mel-spec → Waveform
class MelGAN:
    def generate(self, mel_spec):
        # Transposed convolutions
        waveform = self.generator(mel_spec)
        return waveform

# Real-time 가능! (CPU에서도)
```

---

### 2020: Jukebox - Long-form Music

**OpenAI Jukebox**

**핵심 혁신 #1**: Hierarchical VQ-VAE

```python
class JukeboxVQVAE:
    def __init__(self):
        # 3 levels of compression
        self.encoder_top = Encoder(hop_length=128)     # Coarse
        self.encoder_mid = Encoder(hop_length=32)      # Medium  
        self.encoder_bottom = Encoder(hop_length=8)    # Fine
    
    def encode(self, audio):
        # Multi-scale encoding
        z_top = self.encoder_top(audio)       # (T/128, D)
        z_mid = self.encoder_mid(audio)       # (T/32, D)
        z_bottom = self.encoder_bottom(audio) # (T/8, D)
        
        # Quantize each level
        codes_top = self.quantize(z_top)
        codes_mid = self.quantize(z_mid)
        codes_bottom = self.quantize(z_bottom)
        
        return codes_top, codes_mid, codes_bottom
```

**핵심 혁신 #2**: Autoregressive with conditioning

```python
# Top level: Genre, artist conditioning
codes_top = transformer_top.generate(
    conditioning={"genre": "jazz", "artist": "Miles Davis"}
)

# Mid level: Conditioned on top
codes_mid = transformer_mid.generate(
    conditioning=codes_top
)

# Bottom level: Conditioned on mid
codes_bottom = transformer_bottom.generate(
    conditioning=codes_mid
)

# Decode
audio = vqvae.decode(codes_top, codes_mid, codes_bottom)
```

**성과**: 
- 수 분 길이 음악 생성 가능
- 가사도 생성!
- Artist style 모방

**한계**:
- 여전히 느림 (autoregressive)
- 품질 한계 (VQ bottleneck)
- 샘플링에 5시간 이상 소요!

---

### 2022: EnCodec - Audio Compression 혁신

**Meta EnCodec**

**핵심 혁신**: Residual Vector Quantization (RVQ)

```python
class ResidualVQ(nn.Module):
    """기존 VQ의 문제: 하나의 codebook → 정보 손실"""
    
    def __init__(self, num_quantizers=8):
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size=1024, embedding_dim=128)
            for _ in range(num_quantizers)
        ])
    
    def forward(self, z):
        # z: continuous latent
        
        quantized = 0
        residual = z
        codes = []
        
        # 순차적으로 residual quantize
        for quantizer in self.quantizers:
            q, code = quantizer(residual)
            quantized += q
            codes.append(code)
            
            # 남은 residual
            residual = residual - q
        
        return quantized, codes

# 8 codebooks → 훨씬 높은 품질!
```

**결과**:
```python
# Compression
# 48 kHz audio → 1.5 kbps (32배 압축!)

# Quality
# PESQ: 4.2 (거의 원본과 구별 불가)

# Speed
# Real-time encoding/decoding
```

**의의**: Audio를 language model처럼 처리 가능!

---

### 2023: MusicGen - Transformer 기반 음악 생성

**Meta MusicGen**

**핵심 혁신 #1**: Parallel prediction

```python
# 기존 (Jukebox): Sequential
for t in range(T):
    code_0 = predict_codebook_0(codes[:t])
    code_1 = predict_codebook_1(codes[:t])
    # ...
    code_7 = predict_codebook_7(codes[:t])
# → 8x slower!

# MusicGen: Parallel (다 같이!)
class MusicGenTransformer:
    def forward(self, codes, timestep):
        # codes: (batch, num_codebooks, time)
        
        # Pattern: delay pattern for parallel prediction
        # Codebook 0: t
        # Codebook 1: t-1
        # Codebook 2: t-2
        # ...
        
        # 한 forward pass로 모든 codebook 예측!
        logits = self.transformer(codes)
        
        return logits  # (batch, num_codebooks, time, vocab_size)
```

**핵심 혁신 #2**: Melody conditioning

```python
# Chroma features로 melody 추출
def extract_chroma(melody_audio):
    # 12-dimensional chroma (음계)
    chroma = librosa.feature.chroma_cqt(melody_audio)
    return chroma  # (12, T)

# Conditioning
generated = musicgen.generate_with_chroma(
    text="Epic orchestral music",
    melody_chroma=chroma
)
# → Melody는 유지하면서 style 변경!
```

**성과**:
- 30초 음악을 ~10초에 생성
- Jukebox 대비 1000배 빠름!
- 더 높은 품질

---

### 2023: AudioLDM - Latent Diffusion for Audio

**핵심 혁신 #1**: CLAP (Contrastive Language-Audio Pretraining)

```python
# CLIP처럼, but for audio!
class CLAP:
    def __init__(self):
        self.text_encoder = RoBERTa()
        self.audio_encoder = HTSAT()  # Audio transformer
    
    def contrastive_loss(self, texts, audios):
        # Encode
        text_embeds = self.text_encoder(texts)      # (batch, 512)
        audio_embeds = self.audio_encoder(audios)   # (batch, 512)
        
        # Normalize
        text_embeds = F.normalize(text_embeds)
        audio_embeds = F.normalize(audio_embeds)
        
        # Contrastive
        logits = text_embeds @ audio_embeds.T
        labels = torch.arange(len(texts))
        
        loss = F.cross_entropy(logits, labels)
        return loss

# 결과: Text와 Audio를 같은 embedding space에!
# → Text-to-audio generation 가능
```

**핵심 혁신 #2**: Latent Diffusion

```python
# Pixel-space diffusion (느림)
for t in timesteps:
    spectrogram = denoise(spectrogram, t)  # (H, W)

# Latent diffusion (빠름!)
# 1. VAE encode
z = vae.encode(spectrogram)  # (H/8, W/8)

# 2. Diffusion in latent
for t in timesteps:
    z = denoise(z, t)  # 64배 작음!

# 3. Decode
spectrogram = vae.decode(z)
```

---

### 2024: Stable Audio - Long-form Generation

**Stability AI Stable Audio**

**핵심 혁신 #1**: Timing conditioning

```python
# Diffusion transformer with timing tokens
class StableAudioDiT:
    def forward(self, z, text, start_time, end_time):
        # Timing embeddings
        time_embed = self.time_encoder(start_time, end_time)
        
        # Text embeddings
        text_embed = self.text_encoder(text)
        
        # Combined conditioning
        cond = torch.cat([text_embed, time_embed], dim=-1)
        
        # DiT
        output = self.transformer(z, cond)
        return output

# 사용
audio = model.generate(
    text="Jazz piano, slow intro (0-10s), fast solo (10-60s)",
    start_time=0,
    end_time=60
)
# → 구조 제어 가능!
```

**핵심 혁신 #2**: Variable-length training

```python
# 기존: 고정 길이 (10초)
# → 10초 이상은 불가능

# Stable Audio: Variable length training
# 5초, 10초, 30초, 60초, 180초... 모두 섞어서 훈련!

for batch in dataloader:
    # Random duration
    duration = random.choice([5, 10, 30, 60, 180])
    audio_batch = crop_or_pad(batch, duration)
    
    loss = model.train_step(audio_batch)
```

**결과**:
- 최대 3분 (180초) 음악 생성!
- 구조 제어 가능
- High-quality stereo

---

## 🔬 발전의 핵심 패턴

### 패턴 1: Representation의 진화

```
2016: Raw waveform (44100 Hz)
  → 너무 high-dimensional, 느림

2020: VQ codes (discrete tokens)
  → Bottleneck, 품질 한계

2022: Residual VQ (EnCodec)
  → 8x redundancy, 고품질!

2023: Latent diffusion
  → Continuous, flexible
```

### 패턴 2: 속도 개선

```
2016 WaveNet: 
  1초 생성에 90초 소요 (90x slower than real-time)

2019 MelGAN:
  1초 생성에 0.001초 (1000x faster than real-time!)

2023 MusicGen:
  30초 음악을 10초에 생성

발전: 100,000배!
```

### 패턴 3: 제어 가능성

```
2016: 제어 불가능 (random 생성만)

2020: Genre, artist conditioning

2023: Text, melody conditioning  

2024: Timing, structure 제어

→ 점점 세밀한 제어 가능!
```

---

## 📊 성능 비교

| Year | Model | Quality (FAD ↓) | Speed | Max Duration | Control |
|------|-------|------------------|-------|--------------|---------|
| 2016 | WaveNet | ~10 | 0.01x RT | 1s | None |
| 2019 | MelGAN | ~5 | 1000x RT | 10s | None |
| 2020 | Jukebox | ~3 | 0.001x RT | 4min | Genre, artist |
| 2023 | MusicGen | ~1.5 | 3x RT | 30s | Text, melody |
| 2023 | AudioLDM | ~2 | 10x RT | 10s | Text, image |
| 2024 | Stable Audio | ~1 | 50x RT | 3min | Text, timing |

**FAD (Fréchet Audio Distance)**: 낮을수록 실제 음악과 유사
**RT**: Real-time (1x = 실시간)

**8년간 품질 10배 향상, 속도 5000배 향상!**
total_samples = 44100 × 10 = 441,000

# 문제: 너무 high-dimensional!
# → 생성 모델이 직접 다루기 어려움
```

### 2. Spectrogram

```python
import librosa
import numpy as np

# Waveform → Spectrogram
waveform, sr = librosa.load("audio.wav")
spectrogram = librosa.stft(waveform)  # Short-Time Fourier Transform
spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

# Shape: (frequency_bins, time_frames)
# 예: (1025, 1000) for 10 second audio

# Spectrogram → Waveform (Griffin-Lim algorithm)
waveform_reconstructed = librosa.griffinlim(spectrogram)
```

**장점**: 시각화 가능, 2D image처럼 처리
**단점**: Phase 정보 손실

### 3. Mel Spectrogram

```python
# Human perception에 맞춘 frequency scale
mel_spec = librosa.feature.melspectrogram(
    y=waveform,
    sr=sr,
    n_mels=128  # Mel frequency bins
)

# Shape: (128, time_frames)
# → 더 compact!
```

### 4. Latent Representation (Best!)

```python
# EnCodec (Meta)
from encodec import EncodecModel

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # 6 kbps

# Audio → Discrete codes
audio = torch.randn(1, 1, 24000)  # 1 second
encoded_frames = model.encode(audio)

# Codes: VQ-VAE style discrete tokens
# → Can be modeled with Transformer!

# Codes → Audio
reconstructed = model.decode(encoded_frames)
```

---

## 🎵 주요 모델들

### 1. MusicGen (Meta, 2023)

**아키텍처**: Transformer + EnCodec

```python
from audiocraft.models import MusicGen

# Load model
model = MusicGen.get_pretrained('facebook/musicgen-medium')

# Generate music
model.set_generation_params(
    duration=10,  # seconds
    temperature=1.0,
    top_k=250,
)

descriptions = [
    "80s pop track with bassy drums and synth",
    "90s rock song with loud guitars and heavy drums",
    "Calm meditation music with soft piano"
]

wav = model.generate(descriptions, progress=True)

# Save
import scipy
for idx, one_wav in enumerate(wav):
    scipy.io.wavfile.write(f"generated_{idx}.wav", 32000, one_wav.cpu().numpy())
```

**Melody conditioning**

```python
# Use melody as conditioning
import torchaudio

melody, sr = torchaudio.load("melody.mp3")

# Generate music with melody
wav = model.generate_with_chroma(
    descriptions=["Epic orchestral music"],
    melody_wavs=melody,
    melody_sample_rate=sr,
    progress=True
)
```

**내부 동작**

```python
class MusicGen(nn.Module):
    def __init__(self):
        # 1. Text encoder
        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")
        
        # 2. Audio tokenizer (EnCodec)
        self.audio_tokenizer = EncodecModel.encodec_model_24khz()
        
        # 3. Transformer decoder
        self.transformer = TransformerDecoder(
            num_layers=24,
            hidden_size=1024,
            num_heads=16,
        )
    
    def generate(self, text):
        # Encode text
        text_embed = self.text_encoder(text)
        
        # Auto-regressive generation
        audio_codes = []
        for t in range(max_length):
            # Predict next audio token
            logits = self.transformer(audio_codes, text_embed)
            next_token = logits.argmax(dim=-1)
            audio_codes.append(next_token)
        
        # Decode to waveform
        audio = self.audio_tokenizer.decode(audio_codes)
        return audio
```

---

### 2. AudioLDM (2023)

**Latent Diffusion for Audio**

```python
from audioldm import AudioLDM

model = AudioLDM.from_pretrained("cvssp/audioldm-s-full-v2")

# Text-to-audio
audio = model.generate_audio(
    text="Dog barking in the distance",
    duration=5.0,
    guidance_scale=3.5
)

# Image-to-audio (multi-modal!)
from PIL import Image
image = Image.open("thunderstorm.jpg")

audio = model.generate_audio_from_image(
    image=image,
    duration=5.0
)
```

**아키텍처**

```python
class AudioLDM(nn.Module):
    def __init__(self):
        # 1. VAE for audio
        self.vae = AudioVAE()
        
        # 2. CLAP (Contrastive Language-Audio Pretraining)
        self.clap = CLAPModel()
        
        # 3. Latent diffusion (like Stable Diffusion but for audio)
        self.unet = UNet1D(
            in_channels=8,  # VAE latent channels
            model_channels=256,
            num_res_blocks=2,
        )
    
    def generate(self, text, num_steps=50):
        # Text → CLAP embedding
        text_embed = self.clap.encode_text(text)
        
        # Start from noise
        latent = torch.randn(1, 8, latent_length)
        
        # Diffusion denoising
        for t in reversed(range(num_steps)):
            noise_pred = self.unet(latent, t, context=text_embed)
            latent = self.scheduler.step(noise_pred, t, latent)
        
        # Latent → Audio
        audio = self.vae.decode(latent)
        return audio
```

---

### 3. Stable Audio (Stability AI, 2024)

**Long-form music generation (up to 3 minutes!)**

```python
from stable_audio_tools import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0")

# Generate long music
audio = pipe(
    prompt="Energetic EDM track, 128 BPM, with build-ups and drops",
    negative_prompt="low quality, distorted",
    duration=95,  # seconds
    guidance_scale=7.0
).audios[0]

# Save
import scipy
scipy.io.wavfile.write("edm_track.wav", 44100, audio)
```

**Timing control**

```python
# Control structure with timing
audio = pipe(
    prompt="Jazz piano, intro (0-10s), main theme (10-60s), outro (60-70s)",
    duration=70,
    start_time=0.0,
    end_time=70.0
)
```

---

### 4. Jukebox (OpenAI, 2020)

**VQ-VAE + Transformer (초기 모델)**

```python
import jukebox

# Load model
model, hps = jukebox.load_model('5b_lyrics')

# Generate
sample_length = 393216  # ~8 seconds at 48kHz
metas = [
    dict(
        artist="Frank Sinatra",
        genre="jazz",
        lyrics="I did it my way"
    )
]

# Ancestral sampling
audio = model.sample(sample_length, metas)
```

---

## 🔬 실습: Spectrogram Diffusion

### Simple Audio Diffusion

```python
import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet1DModel

class SimpleAudioDiffusion:
    def __init__(self):
        # 1D U-Net for spectrograms
        self.model = UNet1DModel(
            sample_size=65536,  # Audio length
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock1D",
                "DownBlock1D",
                "DownBlock1D",
                "AttnDownBlock1D",
            ),
            up_block_types=(
                "AttnUpBlock1D",
                "UpBlock1D",
                "UpBlock1D",
                "UpBlock1D",
            ),
        )
        
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    def train_step(self, audio_batch):
        # audio_batch: (batch, 1, 65536)
        
        # Random timestep
        timesteps = torch.randint(0, 1000, (audio_batch.shape[0],))
        
        # Add noise
        noise = torch.randn_like(audio_batch)
        noisy_audio = self.scheduler.add_noise(audio_batch, noise, timesteps)
        
        # Predict noise
        noise_pred = self.model(noisy_audio, timesteps).sample
        
        # Loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    def generate(self, batch_size=1):
        # Start from noise
        audio = torch.randn(batch_size, 1, 65536)
        
        # Denoise
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.model(audio, t).sample
            audio = self.scheduler.step(noise_pred, t, audio).prev_sample
        
        return audio

# Train
diffusion = SimpleAudioDiffusion()
for audio_batch in dataloader:
    loss = diffusion.train_step(audio_batch)
    loss.backward()
    optimizer.step()

# Generate
generated_audio = diffusion.generate(batch_size=4)
```

---

## 🎼 Music Theory for AI

### BPM (Beats Per Minute)

```python
def adjust_bpm(audio, current_bpm, target_bpm):
    """Stretch audio to match target BPM"""
    import librosa
    
    rate = current_bpm / target_bpm
    audio_stretched = librosa.effects.time_stretch(audio, rate=rate)
    return audio_stretched

# Generate 120 BPM music
music_120 = musicgen.generate("House music, 120 BPM")

# Convert to 140 BPM
music_140 = adjust_bpm(music_120, current_bpm=120, target_bpm=140)
```

### Key & Scale

```python
# Generate in specific key
music = musicgen.generate("Piano ballad in C minor, emotional, slow")

# Pitch shift
import librosa

# Transpose up 2 semitones (C → D)
music_transposed = librosa.effects.pitch_shift(
    music,
    sr=32000,
    n_steps=2
)
```

---

## 📊 모델 비교

| Model | Type | Max Duration | Quality | Control |
|-------|------|--------------|---------|---------|
| MusicGen | Autoregressive | 30s | High | Text, melody |
| AudioLDM | Diffusion | 10s | Medium | Text, image |
| Stable Audio | Diffusion | 3min | High | Text, timing |
| Jukebox | VQ-VAE + AR | Minutes | Medium | Artist, genre, lyrics |

---

## 🎓 학습 목표 체크리스트

- [ ] Waveform, spectrogram, latent representation 이해
- [ ] EnCodec audio tokenization 원리 이해
- [ ] MusicGen으로 텍스트 → 음악 생성
- [ ] AudioLDM으로 사운드 이펙트 생성
- [ ] Stable Audio로 긴 음악 생성
- [ ] Spectrogram diffusion 구현

---

## 📚 참고 자료

- [MusicGen](https://github.com/facebookresearch/audiocraft)
- [AudioLDM](https://github.com/haoheliu/AudioLDM)
- [Stable Audio](https://github.com/Stability-AI/stable-audio-tools)
- [Jukebox](https://github.com/openai/jukebox)

---

## ⏭️ 다음

음악 생성을 마스터했습니다!

👉 [Day 7-8: Multi-modal](./03-multimodal.md)에서 **모든 modality를 연결**합니다!

**"이제 음악을 만들 수 있습니다!"** 🎵
