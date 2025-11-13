# Phase 3: Diffusion Models 정복

## 🎯 학습 목표

이미지 생성의 최신 기술인 Diffusion Models를 완전히 이해합니다.

1. ✅ Forward/Reverse diffusion process의 수학적 원리
2. ✅ DDPM (Denoising Diffusion Probabilistic Models) 구현
3. ✅ U-Net 아키텍처 완벽 이해
4. ✅ Sampling 알고리즘 (DDIM, Guidance) 마스터
5. ✅ Text-to-Image 조건부 생성

## 📚 학습 내용

### Week 5: [수학적 기초](./01-mathematical-foundations.md)
- 확률론 기초 (Gaussian, Markov Chain)
- Variational Lower Bound
- Reparameterization Trick
- Forward Diffusion Process
  - Noise Schedule (Linear, Cosine)
  - q(x_t|x_0) 직접 샘플링
- Reverse Process
  - p_θ(x_{t-1}|x_t) 학습

### Week 6: [DDPM 구현](./02-ddpm-implementation.md)
- U-Net 아키텍처
  - Residual Blocks
  - Self-Attention Layers
  - Time Embedding (Sinusoidal)
  - Skip Connections
- Training Loop
  - Noise Prediction Objective
  - Simplified Loss: ||ε - ε_θ(x_t, t)||²
  - EMA (Exponential Moving Average)
- Dataset: MNIST → CIFAR-10 → CelebA

### Week 7: [Sampling & Conditioning](./03-sampling-conditioning.md)
- Sampling 알고리즘
  - DDPM (1000 steps)
  - DDIM (50 steps, deterministic)
  - PLMS, DPM-Solver
- Guidance Techniques
  - Classifier Guidance
  - Classifier-Free Guidance (CFG)
- Conditional Generation
  - Class Conditioning
  - Text Conditioning (CLIP embeddings)
  - ControlNet
  - Inpainting/Outpainting

## 🎓 완료 기준

- ✅ Forward process 수식 유도 가능
- ✅ U-Net 아키텍처 밑바닥 구현
- ✅ DDPM으로 MNIST 생성 성공
- ✅ Classifier-free guidance 이해 및 구현
- ✅ Stable Diffusion 아키텍처 이해

## 🛠️ 실전 프로젝트

### 프로젝트 1: MNIST Diffusion
MNIST 숫자 생성 모델 완전 구현

### 프로젝트 2: Conditional Image Generator
클래스 조건부 이미지 생성

### 프로젝트 3: Mini Stable Diffusion
간단한 text-to-image 모델 구축

## ⏭️ 다음 단계

[Phase 4: GAN 마스터리](../phase4-gan/)로 진행!
