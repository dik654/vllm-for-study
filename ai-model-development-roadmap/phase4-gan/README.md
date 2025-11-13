# Phase 4: GAN 마스터리

## 🎯 학습 목표

적대적 학습(Adversarial Learning)의 모든 것을 마스터합니다.

1. ✅ GAN의 minimax objective 완벽 이해
2. ✅ 훈련 불안정성 해결 기법
3. ✅ Progressive GAN의 growing 메커니즘
4. ✅ StyleGAN의 style-based generation

## 📚 학습 내용

### Week 8: [Vanilla GAN & 안정화](./01-vanilla-gan.md)
- GAN 기초
  - Generator와 Discriminator 구조
  - Minimax Loss vs Non-saturating Loss
  - BCE Loss vs Wasserstein Loss
- 훈련 문제들
  - Mode Collapse 경험 및 해결
  - Vanishing Gradients
  - Training Dynamics 시각화
- 안정화 기법
  - Spectral Normalization
  - Gradient Penalty (WGAN-GP)
  - Instance Noise
  - Label Smoothing
  - Two-Timescale Update Rule (TTUR)

### Week 9: [Progressive GAN](./02-progressive-gan.md)
- Progressive Growing
  - Resolution Ladder (4×4 → 8×8 → ... → 1024×1024)
  - Smooth Fade-in Mechanism
  - Why it works?
- 정규화 기법
  - Pixel Normalization
  - Minibatch Standard Deviation
  - Equalized Learning Rate
- 평가 지표
  - Inception Score (IS)
  - Fréchet Inception Distance (FID)
  - Precision & Recall
  - Perceptual Path Length (PPL)

### Week 10: [StyleGAN](./03-stylegan.md)
- Style-Based Generator
  - Mapping Network (Z → W)
  - AdaIN (Adaptive Instance Normalization)
  - Style Mixing
  - Stochastic Variation (Noise Injection)
- 이론
  - Disentanglement in W space
  - Controllable Generation
  - Path Length Regularization
- StyleGAN2/3 개선
  - Weight Demodulation
  - Skip Connections
  - Lazy Regularization
  - Alias-Free Generator

## 🎓 완료 기준

- ✅ Vanilla GAN 훈련 성공 (mode collapse 해결)
- ✅ FID, IS 계산 구현
- ✅ Progressive training 구현
- ✅ StyleGAN의 style mixing 이해
- ✅ 고품질 얼굴 이미지 생성 가능

## 🛠️ 실전 프로젝트

### 프로젝트 1: DCGAN
MNIST/CIFAR-10 이미지 생성

### 프로젝트 2: Progressive Face Generator
점진적 해상도 증가로 얼굴 생성

### 프로젝트 3: Style Transfer GAN
StyleGAN 기반 스타일 변환

## ⏭️ 다음 단계

[Phase 5: 최신 기술 통합](../phase5-modern-techniques/)로 진행!
