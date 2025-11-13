# Phase 1: Transformer 완전 구현

## 🎯 학습 목표

"Attention is All You Need" 논문을 완벽히 이해하고 밑바닥부터 구현합니다.

1. ✅ Attention 메커니즘의 수학적 원리 완벽 이해
2. ✅ Positional Encoding의 다양한 변형 구현
3. ✅ Encoder-Decoder 아키텍처 완전 구현
4. ✅ 실제 번역 태스크 훈련

## 📚 학습 내용

### Week 1: [기초 구현](./week1-basics/)

#### [Day 1-2: Attention 메커니즘](./01-attention-mechanism.md)
- Scaled Dot-Product Attention numpy 구현
- Multi-Head Attention PyTorch 구현
- Causal Mask와 Attention Pattern 시각화
- Attention의 계산 복잡도 분석

#### [Day 3-4: Positional Encoding](./02-positional-encoding.md)
- Sinusoidal Position Encoding 구현
- Learned Position Embedding
- RoPE (Rotary Position Embedding)
- ALiBi (Attention with Linear Biases)

### Week 2: [전체 모델 구축](./week2-full-model/)

#### [Day 5-6: Transformer 블록](./03-transformer-blocks.md)
- Encoder Block (Self-Attention + FFN)
- Decoder Block (Masked Attention + Cross-Attention)
- Layer Normalization (Pre-LN vs Post-LN)
- Residual Connections

#### [Day 7: 훈련 파이프라인](./04-training-pipeline.md)
- Teacher Forcing
- Learning Rate Scheduling (Warmup)
- Label Smoothing
- Beam Search 디코딩

## 🎓 완료 기준

이 Phase를 완료하면:
- ✅ "Attention is All You Need" 논문을 2시간 내 완전 구현 가능
- ✅ Attention의 모든 변형을 이해하고 설명 가능
- ✅ Custom Transformer 아키텍처 설계 가능
- ✅ 간단한 번역 모델 훈련 가능

## 🛠️ 프로젝트

### Mini Project 1: Attention Visualizer
Attention weights를 시각화하는 인터랙티브 도구

### Mini Project 2: Multi-lingual Translator
영어-한국어 번역 모델 구축

### Final Project: Custom Transformer
자신만의 Transformer 변형 설계 및 구현

## ⏭️ 다음 단계

[Phase 2: BERT/GPT 마스터](../phase2-bert-gpt/)로 진행!
