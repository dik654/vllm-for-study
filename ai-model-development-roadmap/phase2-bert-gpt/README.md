# Phase 2: BERT/GPT 마스터

## 🎯 학습 목표

Pre-training과 Fine-tuning의 핵심을 완벽히 이해합니다.

1. ✅ 토크나이저 알고리즘 (BPE, WordPiece) 밑바닥 구현
2. ✅ BERT의 MLM + NSP 이해 및 구현
3. ✅ GPT의 Causal LM 이해 및 구현
4. ✅ Generation 전략 마스터 (Greedy, Beam Search, Sampling)

## 📚 학습 내용

### Week 3: [토크나이저 구현](./01-tokenizers.md)
- BPE (Byte Pair Encoding) 알고리즘 직접 구현
- WordPiece vs SentencePiece
- Vocab 구축 및 Merge rules
- Special tokens 처리
- Fast Tokenizer 최적화

### Week 4: Pre-training 구현

#### [BERT 구현](./02-bert-implementation.md)
- Masked Language Model (MLM)
- Next Sentence Prediction (NSP)
- BERT 아키텍처 (Encoder-only)
- Fine-tuning for downstream tasks
- [CLS] token의 역할

#### [GPT 구현](./03-gpt-implementation.md)
- Causal Language Model
- GPT 아키텍처 (Decoder-only)
- Generation 전략
  - Greedy Search
  - Beam Search
  - Top-k Sampling
  - Top-p (Nucleus) Sampling
- Few-shot Prompting

## 🎓 완료 기준

- ✅ BPE 알고리즘을 30분 내 코딩 가능
- ✅ BERT와 GPT의 차이점 명확히 설명 가능
- ✅ Custom 토크나이저로 언어 모델 훈련 가능
- ✅ 다양한 generation 전략의 trade-off 이해

## ⏭️ 다음 단계

[Phase 3: Diffusion Models](../phase3-diffusion/)로 진행!
