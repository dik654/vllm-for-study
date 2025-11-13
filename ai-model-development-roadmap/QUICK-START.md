# 🚀 Quick Start Guide

**AI 모델 개발 로드맵을 15분 안에 시작하기**

이 가이드는 완전 초보자를 위한 빠른 시작 가이드입니다.

---

## ⚡ 0단계: 환경 설정 (5분)

### Python 설치 확인

```bash
python --version  # Python 3.8+ 필요
```

없다면: https://www.python.org/downloads/

### PyTorch 설치

```bash
# CPU only (학습용)
pip install torch torchvision

# GPU (NVIDIA)
# https://pytorch.org/ 에서 자신의 CUDA 버전에 맞게 선택
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 기본 라이브러리

```bash
pip install numpy matplotlib jupyter transformers datasets
```

---

## 🎯 1단계: 첫 번째 Attention 구현 (10분)

### 코드 작성 및 실행

```python
# attention_first.py
import torch
import torch.nn.functional as F

def attention(Q, K, V):
    """
    가장 간단한 Attention 구현

    Args:
        Q: Query (batch, seq_len, d_k)
        K: Key (batch, seq_len, d_k)
        V: Value (batch, seq_len, d_k)
    """
    # 1. Q와 K의 유사도 계산
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # 2. Softmax로 확률 분포로 변환
    attention_weights = F.softmax(scores, dim=-1)

    # 3. V에 가중치 적용
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# 테스트
batch = 1
seq_len = 4
d_k = 8

Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_k)

output, weights = attention(Q, K, V)

print(f"Input shape: {Q.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\nAttention weights (어떤 token에 attention하는지):")
print(weights[0])
```

실행:
```bash
python attention_first.py
```

**축하합니다! Attention을 구현했습니다! 🎉**

---

## 📚 2단계: 학습 경로 선택

### 옵션 A: 빠른 경로 (실용적)

**목표**: 빠르게 실전 프로젝트 시작

```
1주차: Phase 0 (HuggingFace 기초)
  ↓
2주차: Phase 1.1-1.4 (Transformer 기본)
  ↓
3주차: Phase 2.1-2.3 (GPT 기본)
  ↓
4주차: 첫 프로젝트 (Text generation with GPT-2)
```

**시작점**: `phase0-foundations/01-핵심-용어.md`

### 옵션 B: 탄탄한 기초 (수학 포함)

**목표**: 수학부터 차근차근

```
2주차: Phase -1 (선형대수, 미적분)
  ↓
1주차: Phase 0 (HuggingFace)
  ↓
2주차: Phase 1 (Transformer)
  ↓
2주차: Phase 2 (BERT/GPT)
  ↓
  ... 계속
```

**시작점**: `phase-1-math-foundations/01-linear-algebra-basics.md`

### 옵션 C: 특정 목표 집중

**현대 LLM 기술만 배우고 싶다면**:
```
Phase 1.5: Modern Attention Variants (MQA, GQA)
  ↓
Phase 1.6: Context Length Extension
  ↓
Phase 2.4: Modern LLM Architectures (Llama 2, Mistral)
  ↓
Phase 5.3: Speculative Decoding
```

**시작점**: `phase1-transformer/05-modern-attention-variants.md`

**Vision AI에 관심있다면**:
```
Phase 1 (Transformer 기본)
  ↓
Phase 1.7: Vision Transformers (ViT)
  ↓
Phase 3: Diffusion Models
  ↓
프로젝트: Stable Diffusion fine-tuning
```

**시작점**: `phase1-transformer/07-vision-transformers.md`

---

## 🎓 3단계: 첫 주 학습 계획

### Day 1 (월요일): 환경 & Attention

**오전 (2시간)**:
- [ ] 환경 설정 완료
- [ ] Attention 구현 및 이해
- [ ] 시각화 코드 작성

**오후 (2시간)**:
- [ ] `phase1-transformer/01-attention-mechanism.md` 읽기
- [ ] Multi-Head Attention 구현

**저녁 (1시간)**:
- [ ] 오늘 배운 내용 정리
- [ ] 내일 학습 계획

### Day 2 (화요일): Positional Encoding

**오전 (2시간)**:
- [ ] `phase1-transformer/02-positional-encoding.md` 읽기
- [ ] Sinusoidal PE 구현

**오후 (2시간)**:
- [ ] RoPE 이해 및 구현
- [ ] PE visualization

**저녁 (1시간)**:
- [ ] 복습 및 정리

### Day 3 (수요일): Transformer 블록

**오전 (3시간)**:
- [ ] `phase1-transformer/03-transformer-blocks.md` 읽기
- [ ] EncoderBlock 구현

**오후 (2시간)**:
- [ ] DecoderBlock 구현
- [ ] Pre-LN vs Post-LN 실험

### Day 4 (목요일): 전체 Transformer

**전체 (5시간)**:
- [ ] 전체 Transformer 조립
- [ ] 간단한 데이터로 훈련
- [ ] Loss 그래프 확인

### Day 5 (금요일): 복습 & 프로젝트

**오전 (2시간)**:
- [ ] 이번 주 배운 내용 복습
- [ ] 막혔던 부분 다시 보기

**오후 (3시간)**:
- [ ] Mini 프로젝트: "Shakespeare 텍스트 생성"
- [ ] 모델 훈련 및 샘플 생성

### 주말: 휴식 & 선택적 학습

**토요일**:
- 평일 못다한 부분 마무리
- 관련 논문 읽기 (선택)

**일요일**:
- 휴식! 🎮
- 다음 주 계획

---

## 💡 학습 팁

### 1. 작게 시작하기

```python
# ❌ 나쁜 예: 처음부터 큰 모델
model = GPT(
    vocab_size=50000,
    d_model=4096,
    num_layers=48
)  # 메모리 부족!

# ✅ 좋은 예: 작은 모델로 시작
model = GPT(
    vocab_size=1000,   # 작은 vocab
    d_model=128,       # 작은 dimension
    num_layers=2       # 적은 layers
)  # 빠르게 실험 가능!
```

### 2. 시각화하기

```python
import matplotlib.pyplot as plt

# Attention weights 시각화
plt.imshow(attention_weights[0].detach().numpy(), cmap='hot')
plt.title('Attention Weights')
plt.xlabel('Key')
plt.ylabel('Query')
plt.colorbar()
plt.show()
```

### 3. 디버깅은 Shape부터

```python
# 항상 shape 확인!
print(f"Q shape: {Q.shape}")  # (batch, seq_len, d_k)
print(f"K shape: {K.shape}")  # (batch, seq_len, d_k)
print(f"Scores shape: {scores.shape}")  # (batch, seq_len, seq_len)
```

### 4. 작은 성공 축하하기

- Attention 구현 완료 → ✅ 체크!
- 첫 loss 감소 → 🎉 축하!
- 모델이 의미있는 텍스트 생성 → 🏆 대성공!

---

## 🆘 문제 해결

### "CUDA out of memory" 오류

```python
# 해결책 1: Batch size 줄이기
batch_size = 32  # → 16 또는 8로

# 해결책 2: 모델 크기 줄이기
d_model = 512  # → 256 또는 128로

# 해결책 3: Gradient checkpointing
model.gradient_checkpointing_enable()
```

### "Loss가 감소하지 않아요"

```python
# 체크리스트:
# 1. Learning rate가 너무 높거나 낮지 않은지
lr = 1e-4  # 시작은 1e-4가 안전

# 2. Data가 올바르게 로드되는지
for batch in dataloader:
    print(batch.shape)  # 확인!
    break

# 3. Forward pass가 올바른지
output = model(dummy_input)
print(output.shape)  # 예상과 일치하는지 확인

# 4. Loss가 계산되는지
loss = criterion(output, target)
print(loss.item())  # NaN이 아닌지 확인
```

### "이해가 안 돼요"

**단계별 접근**:

1. **수식 → 코드 매핑**
   ```python
   # 수식: Attention(Q,K,V) = softmax(QK^T/√d_k)V
   # 코드:
   scores = Q @ K.T / sqrt(d_k)  # QK^T/√d_k
   weights = softmax(scores)      # softmax(...)
   output = weights @ V           # ...V
   ```

2. **작은 예제로 확인**
   ```python
   # 2×2 예제로 손으로 계산
   Q = torch.tensor([[1, 0], [0, 1]])
   K = torch.tensor([[1, 0], [0, 1]])
   # ... 계산 과정을 종이에 적어보기
   ```

3. **시각화**
   - Attention weights를 그림으로
   - Loss curve를 그래프로
   - Embedding을 t-SNE로

---

## 📖 추천 리소스

### 필수 읽기

1. **The Illustrated Transformer** (Jay Alammar)
   - https://jalammar.github.io/illustrated-transformer/
   - 그림으로 보는 Transformer

2. **Annotated Transformer** (Harvard NLP)
   - http://nlp.seas.harvard.edu/annotated-transformer/
   - 코드와 설명이 함께

### 선택적 읽기

3. **3Blue1Brown - Neural Networks**
   - YouTube: 시각적 설명 최고

4. **Fast.ai Course**
   - 실용적 딥러닝 입문

### 커뮤니티

- Reddit: r/MachineLearning
- HuggingFace Forums
- Twitter/X: #AI, #ML 태그

---

## ✅ 첫 주 완료 체크리스트

첫 주를 마쳤다면 다음을 할 수 있어야 합니다:

- [ ] Attention 메커니즘을 코드로 구현
- [ ] Multi-Head Attention 이해
- [ ] Positional Encoding 구현
- [ ] 간단한 Transformer block 구현
- [ ] 작은 데이터로 모델 훈련
- [ ] Loss가 감소하는 것 확인
- [ ] 생성된 텍스트 샘플 확인

**모두 체크했다면**: 🎉 **축하합니다! Phase 1 완료!**

---

## 🚀 다음 단계

### Week 2: GPT & Text Generation

```
Day 1-2: GPT 아키텍처 이해
Day 3-4: Causal language modeling
Day 5-7: Text generation (greedy, sampling, beam search)
프로젝트: GPT-style 모델로 시 생성
```

### Week 3: 현대 기법

```
Day 1-2: GQA (Grouped-Query Attention)
Day 3-4: RMSNorm, SwiGLU
Day 5-7: Llama 2 스타일 모델 구현
```

### Week 4: 프로젝트

```
프로젝트 선택:
- 챗봇 만들기
- 코드 생성 모델
- 요약 모델
- 번역 모델
```

---

## 💬 자주 묻는 질문

### Q: 수학을 꼭 알아야 하나요?

**A**: 기초만 있으면 됩니다!
- 필수: 행렬 곱셈, 미분 개념
- 선택: SVD, Eigenvalue (나중에 배워도 됨)

### Q: GPU가 없어요

**A**: 괜찮습니다!
- Phase 1-2는 CPU로 충분
- Google Colab 무료 GPU 사용 (제한적)
- Cloud GPU (AWS, GCP) 시간당 $1-3

### Q: 얼마나 걸리나요?

**A**: 목표에 따라 다릅니다
- 기본 이해: 1-2개월
- 실무 수준: 3-6개월
- 전문가: 12개월+

### Q: 어떤 프로젝트를 해야 하나요?

**A**: 관심사에 따라!
- 자연어: 챗봇, 요약, 번역
- 비전: 이미지 생성, 분류
- 코드: Code generation
- 음악/오디오: 음악 생성

---

## 🎯 당신의 첫 단계

**지금 바로 시작하세요!**

```bash
# 1. 이 repo를 clone
git clone https://github.com/your-repo/ai-model-development-roadmap

# 2. 첫 번째 코드 실행
cd ai-model-development-roadmap
python examples/attention_first.py

# 3. 첫 번째 문서 읽기
open phase1-transformer/01-attention-mechanism.md
```

**Remember**:
- 완벽보다 **진행**이 중요
- 매일 **조금씩**이 낫다
- **막히면** 쉬어가도 됨
- **즐기세요**! 🎉

---

**Good luck on your AI journey! 🚀**

**Questions?**
- GitHub Issues에 질문 남기기
- [NEXT-STEPS.md](NEXT-STEPS.md)에서 상세 로드맵 확인
- [FAQ.md](FAQ.md)에서 자주 묻는 질문 확인
