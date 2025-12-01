# PART 1: Transformer Core (필수 기본)

> AI 기업 면접 합격을 위한 Transformer 핵심 지식 완전 정복

---

## 목차
1. [Tokenizer (BPE/Unigram/SentencePiece)](#1-tokenizer)
2. [Embedding & Position Embedding](#2-embedding--position-embedding)
3. [Multi-Head Attention](#3-multi-head-attention)
4. [FFN/MLP](#4-ffnmlp)
5. [Residual, LayerNorm, Pre-LN vs Post-LN](#5-residual-layernorm-pre-ln-vs-post-ln)
6. [Transformer Block 전체 흐름](#6-transformer-block-전체-흐름)
7. [면접 예상 질문 및 답변](#7-면접-예상-질문-및-답변)

---

## 1. Tokenizer

### 1.1 직관적 이해

**왜 토크나이저가 필요한가?**

컴퓨터는 문자열을 직접 이해하지 못합니다. 신경망은 숫자(벡터)만 처리할 수 있으므로, 텍스트를 숫자 시퀀스로 변환해야 합니다.

```
"Hello world" → [15496, 995] → 임베딩 벡터들
```

**토크나이저의 딜레마:**
- **문자 단위**: 어휘 크기 작음 (26개), 시퀀스 매우 길어짐 → 긴 의존성 학습 어려움
- **단어 단위**: 어휘 크기 폭발 (수십만), OOV(Out-of-Vocabulary) 문제
- **서브워드 단위**: 적절한 타협점 (30K~50K 어휘)

### 1.2 BPE (Byte Pair Encoding)

**알고리즘 핵심:**
1. 모든 문자를 개별 토큰으로 시작
2. 가장 빈번한 연속 토큰 쌍을 새 토큰으로 병합
3. 원하는 어휘 크기까지 반복

**수학적 표현:**
```
병합 규칙: (a, b) → ab  where  (a, b) = argmax_{(x,y)} count(x,y)
```

**예시:**
```
초기: ['l', 'o', 'w', 'e', 'r', ' ', 'n', 'e', 'w', 'e', 's', 't']
1단계: 'e' + 's' → 'es'  (가장 빈번)
2단계: 'es' + 't' → 'est'
...
최종: ['low', 'er', ' ', 'new', 'est']
```

**장점:**
- 희귀 단어도 서브워드로 분해 가능
- 새로운 단어에 대한 일반화
- 형태소 수준의 의미 보존

### 1.3 Unigram Language Model

**BPE와의 차이:**
- BPE: Bottom-up (병합)
- Unigram: Top-down (제거)

**알고리즘:**
1. 매우 큰 초기 어휘로 시작
2. 각 토큰의 확률 계산: P(token)
3. 어휘에서 제거했을 때 전체 likelihood 감소가 가장 작은 토큰 제거
4. 원하는 크기까지 반복

**수학적 기반:**
```
텍스트 X에 대한 likelihood:
L(X) = Σ log P(x_i)  where x_i ∈ segmentation(X)

최적 분할:
x* = argmax_{x∈S(X)} Σ log P(x_i)
```

Viterbi 알고리즘으로 최적 분할을 O(n) 시간에 계산

### 1.4 SentencePiece

**핵심 특징:**
- 언어에 독립적 (Language-agnostic)
- 공백도 하나의 문자로 처리 (`▁` = 단어 시작)
- 원문 완벽 복원 가능 (Lossless)

```python
# 일반 토크나이저
"Hello World" → ["Hello", "World"]  # 공백 정보 손실

# SentencePiece
"Hello World" → ["▁Hello", "▁World"]  # 공백 보존
```

**구현 방식:**
- BPE 또는 Unigram 중 선택 가능
- 바이트 수준 폴백으로 모든 문자 처리

### 1.5 실제 코드에서의 토크나이저 (vLLM)

```python
# vllm/transformers_utils/tokenizer.py 참조
class TokenizerGroup:
    def encode(self, text: str) -> List[int]:
        """텍스트를 토큰 ID 시퀀스로 변환"""
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """토큰 ID 시퀀스를 텍스트로 복원"""
        return self.tokenizer.decode(token_ids)
```

### 1.6 파생 질문들

**Q1: 왜 GPT-4는 어휘 크기가 100K인가?**
- 다국어 지원을 위해 더 많은 토큰 필요
- 코드, 수학 기호 등 특수 도메인
- 토큰 효율성 vs 임베딩 테이블 크기 트레이드오프

**Q2: 토크나이저와 모델은 왜 함께 학습되지 않나?**
- 토크나이저는 discrete 연산 (미분 불가)
- 모델 학습 전에 고정된 어휘 필요
- 최근 연구: Byte-level 모델 (ByT5)

---

## 2. Embedding & Position Embedding

### 2.1 Token Embedding

**직관적 이해:**
각 토큰 ID를 고차원 벡터로 매핑. 이 벡터가 토큰의 "의미"를 표현.

**수학적 정의:**
```
E: ℝ^{V} → ℝ^{d}
E(x) = W_e[x]  where W_e ∈ ℝ^{V × d}

- V: 어휘 크기 (예: 50257)
- d: 임베딩 차원 (예: 768, 4096)
```

**메모리 계산:**
```
파라미터 수 = V × d
GPT-2: 50257 × 768 = 38.6M 파라미터
LLaMA-70B: 32000 × 8192 = 262M 파라미터
```

### 2.2 Position Embedding의 필요성

**왜 필요한가?**
Self-Attention은 순서 불변(permutation invariant):
```
Attention(Q, K, V) = Attention(π(Q), π(K), π(V))
```

"나는 밥을 먹었다"와 "밥을 나는 먹었다"가 동일하게 처리됨!

### 2.3 Sinusoidal Position Embedding (원본 Transformer)

**수학적 정의:**
```
PE(pos, 2i) = sin(pos / 10000^{2i/d})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d})
```

**핵심 특성:**
1. **상대 위치 표현 가능:**
   ```
   PE(pos+k) = Linear_transform(PE(pos))
   ```

2. **외삽 가능:** 학습하지 않은 긴 시퀀스도 처리 가능

**기하학적 해석:**
각 차원 쌍은 서로 다른 주파수의 회전. 저주파(큰 i)는 전체 위치, 고주파(작은 i)는 세밀한 위치.

### 2.4 Learned Position Embedding (GPT)

```python
self.wpe = nn.Embedding(max_position, d_model)

# 사용
position_ids = torch.arange(seq_len)
position_embeddings = self.wpe(position_ids)
```

**장점:** 더 유연한 표현 학습
**단점:** 학습된 위치 이상으로 외삽 불가

### 2.5 RoPE (Rotary Position Embedding) ⭐️ 중요

**현대 LLM의 표준** (LLaMA, Mistral, Qwen 등)

**핵심 아이디어:**
위치 정보를 벡터의 "회전"으로 인코딩

**수학적 정의:**
```
q̃_m = R_θ,m · q_m
k̃_n = R_θ,n · k_n

where R_θ,m = [cos(mθ)  -sin(mθ)]
              [sin(mθ)   cos(mθ)]
```

**왜 회전인가?**
내적이 상대 위치만의 함수가 됨:
```
q̃_m · k̃_n = q_m · R_{θ,n-m} · k_n = f(q, k, n-m)
```

**주파수 설정:**
```
θ_i = base^{-2i/d} = 10000^{-2i/d}
```

**vLLM에서의 RoPE 적용:**
```python
# vllm/model_executor/layers/rotary_embedding.py
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # cos, sin: [seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

### 2.6 ALiBi (Attention with Linear Biases)

**접근법:** 위치 임베딩 대신 attention score에 bias 추가
```
Attention_ij = softmax(q_i · k_j / √d - m · |i - j|)
```

**장점:** 학습 없이 긴 컨텍스트로 외삽 가능

### 2.7 임베딩 계층 요약

```python
# 최종 입력 임베딩
def get_input_embeddings(token_ids, position_ids):
    token_emb = self.token_embedding(token_ids)    # [B, T, d]
    pos_emb = self.position_embedding(position_ids) # [B, T, d]
    return token_emb + pos_emb  # [B, T, d]
```

---

## 3. Multi-Head Attention

### 3.1 Self-Attention 직관

**핵심 질문:** "이 토큰이 다른 토큰들 중 어디에 주목해야 할까?"

**비유:** 문장을 읽을 때 각 단어가 다른 단어들을 "참조"하여 의미 파악
```
"The animal didn't cross the street because it was too tired."
"it" → "animal"을 참조해야 의미 파악 가능
```

### 3.2 Scaled Dot-Product Attention

**수학적 정의:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

입력 차원:
- Q: [B, T, d_k]  (Query)
- K: [B, T, d_k]  (Key)
- V: [B, T, d_v]  (Value)
- 출력: [B, T, d_v]
```

**단계별 분석:**

1. **유사도 계산:** `S = QK^T`  → [B, T, T]
   - 각 쿼리와 모든 키의 내적
   - `S[i,j]` = i번째 토큰이 j번째 토큰과 얼마나 관련있는지

2. **스케일링:** `S = S / √d_k`
   - **왜?** 내적 값이 d_k가 커지면 분산도 커짐
   - softmax가 극단값에서 기울기 소실
   ```
   Var(q·k) = d_k · Var(q_i) · Var(k_i) ≈ d_k (정규화된 경우)
   ```

3. **Softmax:** `A = softmax(S)`  → [B, T, T]
   - 각 행이 확률 분포 (합=1)
   - `A[i,:]` = i번째 토큰의 attention 가중치

4. **가중 합:** `O = A · V`  → [B, T, d_v]
   - 각 토큰이 다른 토큰들의 value를 가중 합산

### 3.3 Multi-Head Attention

**왜 여러 개의 head?**
- 서로 다른 유형의 관계 학습 (구문, 의미, 지시어 등)
- 부분 공간에서 더 풍부한 표현

**수학적 정의:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

차원:
- W_i^Q: [d, d_k]  where d_k = d / h
- W_i^K: [d, d_k]
- W_i^V: [d, d_v]  where d_v = d / h
- W^O: [h·d_v, d]
```

**구현:**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        self.d_k = d_model // num_heads  # 64
        self.num_heads = num_heads

        # 하나의 큰 행렬로 모든 head 계산
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        # [B, T, d] → [B, T, h, d_k] → [B, h, T, d_k]
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Attention: [B, h, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        # [B, h, T, d_k] → [B, T, h, d_k] → [B, T, d]
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.W_o(out)
```

### 3.4 Causal Masking (Autoregressive)

**목적:** 미래 토큰 참조 방지 (LLM 학습 시)

```python
# Causal mask 생성
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

**시각화:**
```
      t0  t1  t2  t3
t0 [  ✓   ✗   ✗   ✗  ]  ← t0는 자신만 참조
t1 [  ✓   ✓   ✗   ✗  ]  ← t1은 t0, t1 참조
t2 [  ✓   ✓   ✓   ✗  ]  ← t2는 t0, t1, t2 참조
t3 [  ✓   ✓   ✓   ✓  ]  ← t3는 모두 참조
```

### 3.5 Grouped-Query Attention (GQA)

**문제:** Multi-Head Attention의 KV 캐시 메모리 부담

**해결:** Key, Value head 수 줄이기

```
MHA: Q heads = K heads = V heads = 32
MQA: Q heads = 32, K heads = V heads = 1  (너무 극단적)
GQA: Q heads = 32, K heads = V heads = 8  (적절한 타협)
```

**메모리 절약:**
```
KV 캐시 = 2 × num_kv_heads × seq_len × head_dim × dtype_size
GQA (8 heads) vs MHA (32 heads) → 4배 메모리 절약
```

### 3.6 Attention 복잡도

| 연산 | 시간 복잡도 | 공간 복잡도 |
|------|------------|------------|
| QK^T | O(T²d) | O(T²) |
| Softmax | O(T²) | O(T²) |
| AV | O(T²d) | O(Td) |
| **총계** | **O(T²d)** | **O(T² + Td)** |

**긴 시퀀스 문제:**
T = 100K 일 때, attention 행렬만 100K × 100K × 4bytes = 40GB!

---

## 4. FFN/MLP

### 4.1 직관적 이해

**역할:** Attention이 "어떤 정보를 모을지"를 결정하면, FFN은 "그 정보를 어떻게 변환할지" 결정

**비유:**
- Attention: 회의에서 누구의 의견을 들을지 선택
- FFN: 수집된 의견을 종합하여 결론 도출

### 4.2 기본 구조

**수학적 정의:**
```
FFN(x) = W_2 · σ(W_1 · x + b_1) + b_2

차원:
- x: [B, T, d]
- W_1: [d, 4d]  (확장)
- W_2: [4d, d]  (축소)
- 은닉층: 4d (보통 4배 확장)
```

**구현:**
```python
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))
```

### 4.3 활성화 함수

**ReLU → GELU → SiLU(Swish) 변천사:**

```
ReLU(x) = max(0, x)
- 문제: 음수 영역에서 완전히 0 (dying ReLU)

GELU(x) = x · Φ(x) ≈ x · σ(1.702x)
- 확률적 해석: x가 양수일 확률로 가중
- 부드러운 비선형성

SiLU(x) = x · σ(x) = x / (1 + e^{-x})
- GELU와 유사하지만 계산 더 간단
- LLaMA 등에서 사용
```

### 4.4 Gated FFN (GLU 변형)

**현대 LLM 표준** (LLaMA, Mistral)

```
FFN_gated(x) = (W_gate · x ⊙ σ(W_up · x)) · W_down

- W_gate: [d, 4d]
- W_up: [d, 4d]
- W_down: [4d, d]
- ⊙: element-wise multiplication
```

**vLLM 구현:**
```python
# vllm/model_executor/layers/activation.py
class SiluAndMul(nn.Module):
    def forward(self, x):
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]
```

**파라미터 증가:**
```
기본 FFN: 2 × d × 4d = 8d²
Gated FFN: 3 × d × 4d = 12d² (1.5배)
```
→ 실제로는 hidden dim을 8/3배로 조정하여 파라미터 수 맞춤

### 4.5 FFN의 역할 해석

**최근 연구 (Geva et al., 2020):**
- FFN의 첫 번째 층 = "키-값 메모리"
- 각 뉴런이 특정 패턴 감지 (키)
- 감지 시 특정 출력 생성 (값)

```
예: 뉴런 #1234
- 활성화 패턴: "수도는" 다음에 등장하는 토큰
- 출력: "서울", "파리", "도쿄" 등의 방향으로 이동
```

---

## 5. Residual, LayerNorm, Pre-LN vs Post-LN

### 5.1 Residual Connection (잔차 연결)

**핵심 아이디어:**
```
출력 = 입력 + 변환(입력)
y = x + F(x)
```

**왜 필요한가?**

1. **기울기 흐름:** 깊은 네트워크에서 기울기 소실 방지
   ```
   ∂L/∂x = ∂L/∂y · (1 + ∂F/∂x)

   잔차 없이: ∂L/∂x = ∂L/∂y · ∂F/∂x  (곱셈 → 소실)
   잔차 있음: 항상 1 이상 보장
   ```

2. **항등 함수 학습:** F(x)=0만 학습하면 됨 (쉬움)

3. **앙상블 효과:** 다양한 깊이의 경로 존재
   ```
   2개 블록 → 4가지 경로: x, F1(x), F2(x), F2(F1(x))
   n개 블록 → 2^n 경로
   ```

### 5.2 Layer Normalization

**수학적 정의:**
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β

μ = (1/d) Σ x_i        (평균)
σ² = (1/d) Σ (x_i - μ)²  (분산)
γ, β: 학습 가능한 파라미터
```

**Batch Norm과의 차이:**
```
Batch Norm: 배치 차원으로 정규화 [B] → 추론 시 running stats 필요
Layer Norm: 특징 차원으로 정규화 [d] → 배치 크기 1에서도 동작
```

**왜 LLM은 LayerNorm 사용?**
- 시퀀스 길이가 가변적
- 배치 크기 1로 추론하는 경우 많음
- 토큰 단위 독립적 정규화

### 5.3 RMSNorm (Root Mean Square Normalization)

**LLaMA 등 현대 LLM의 선택:**
```
RMSNorm(x) = γ · x / √((1/d) Σ x_i² + ε)
```

**LayerNorm 대비 장점:**
- 평균 빼기 연산 제거 → 계산 효율
- 실험적으로 성능 동등
- β 파라미터 불필요

**vLLM 구현:**
```python
# vllm/model_executor/layers/layernorm.py
class RMSNorm(nn.Module):
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
```

### 5.4 Pre-LN vs Post-LN

**Post-LN (원본 Transformer):**
```
x = x + Attention(LayerNorm(x))  ← 잘못됨 (원본)
x = LayerNorm(x + Attention(x))  ← Post-LN
```

**Pre-LN (현대 표준):**
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**비교:**

| 특성 | Post-LN | Pre-LN |
|-----|---------|--------|
| 학습 안정성 | 낮음 (warmup 필요) | 높음 |
| 최종 성능 | 약간 높음 | 약간 낮음 |
| 초기화 민감도 | 높음 | 낮음 |
| 현재 사용 | BERT | GPT, LLaMA |

**기울기 분석:**
```
Post-LN: 기울기가 층을 거치며 누적 → 초기 층 기울기 폭발
Pre-LN: 각 층에서 정규화 → 안정적 기울기 흐름
```

---

## 6. Transformer Block 전체 흐름

### 6.1 단일 블록 구조 (Pre-LN)

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = GatedFFN(d_model, d_ff)

    def forward(self, x, mask=None, kv_cache=None):
        # Self-Attention with residual
        h = self.ln1(x)
        attn_out, new_kv = self.attn(h, mask, kv_cache)
        x = x + attn_out

        # FFN with residual
        h = self.ln2(x)
        x = x + self.ffn(h)

        return x, new_kv
```

### 6.2 전체 모델 구조

```python
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len):
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids, kv_caches=None):
        # Token + Position Embedding
        x = self.tok_emb(token_ids)  # [B, T, d]
        # RoPE는 attention 내부에서 적용

        # Transformer Blocks
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches else None
            x, new_kv = block(x, kv_cache=kv)
            new_kv_caches.append(new_kv)

        # Final LayerNorm + LM Head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        return logits, new_kv_caches
```

### 6.3 데이터 흐름 시각화

```
입력: "The cat sat"
  ↓
[Tokenize] → [15, 2368, 3492]
  ↓
[Token Embedding] → [B, 3, 4096]
  ↓
┌─────────────────────────────────┐
│ Transformer Block × 32         │
│   ↓                            │
│ [RMSNorm] → [Self-Attention]   │
│      ↓         ↓               │
│      └────[Add]────┘           │
│            ↓                   │
│ [RMSNorm] → [Gated FFN]        │
│      ↓         ↓               │
│      └────[Add]────┘           │
└─────────────────────────────────┘
  ↓
[Final RMSNorm]
  ↓
[LM Head] → [B, 3, 50257]
  ↓
[Softmax] → 다음 토큰 확률
```

### 6.4 파라미터 수 계산

**LLaMA-7B 예시:**
```
설정: d=4096, h=32, n_layers=32, vocab=32000, d_ff=11008

토큰 임베딩: 32000 × 4096 = 131M
각 블록:
  - QKV proj: 3 × 4096 × 4096 = 50M
  - Output proj: 4096 × 4096 = 17M
  - FFN (gated): 3 × 4096 × 11008 = 135M
  - RMSNorm: 2 × 4096 = 8K
  - 블록 합계: ≈ 202M
32개 블록: 202M × 32 = 6.5B
LM Head: 4096 × 32000 = 131M (임베딩과 공유)
Final Norm: 4096

총계: ≈ 6.7B 파라미터
```

---

## 7. 면접 예상 질문 및 답변

### Q1: Self-Attention의 시간 복잡도가 O(n²)인 이유와 해결 방법은?

**답변:**
QK^T 연산에서 모든 토큰 쌍의 유사도를 계산하기 때문입니다.

해결 방법들:
1. **Sparse Attention** (Longformer): 지역적 + 전역적 패턴만 계산
2. **Linear Attention** (Performer): 커널 근사로 O(n)으로 감소
3. **FlashAttention**: 복잡도는 동일하나 메모리 I/O 최적화
4. **Sliding Window** (Mistral): 고정 윈도우 내에서만 attention

### Q2: 왜 d_k로 스케일링하는가?

**답변:**
Q와 K가 평균 0, 분산 1일 때, 내적 q·k의 분산은 d_k입니다.

```
Var(q·k) = Σ Var(q_i·k_i) = d_k (독립 가정)
```

√d_k로 나누면 분산이 1로 정규화되어 softmax가 안정적으로 동작합니다.
스케일링 없이는 softmax 출력이 one-hot에 가까워져 기울기 소실이 발생합니다.

### Q3: Pre-LN이 학습에 더 안정적인 이유는?

**답변:**
Pre-LN에서는 잔차 경로(identity path)를 통해 기울기가 직접 전파됩니다.

```
Post-LN: x → Attn → Add → LN
         기울기가 LN을 통과하며 스케일 변화

Pre-LN: x → LN → Attn ─┐
        x ──────────────┴→ Add
        기울기가 직접 x로 전파
```

### Q4: RoPE가 상대 위치를 표현하는 원리는?

**답변:**
회전의 성질을 이용합니다. 위치 m의 쿼리와 위치 n의 키를 각각 회전시키면:

```
q_m^T · k_n = (R_m · q)^T · (R_n · k)
            = q^T · R_m^T · R_n · k
            = q^T · R_{n-m} · k
```

결과가 상대 위치 (n-m)만의 함수가 됩니다.

### Q5: GQA의 장단점은?

**장점:**
- KV 캐시 메모리 절약 (4-8배)
- 추론 속도 향상 (메모리 대역폭 병목 해소)
- MQA보다 품질 저하 적음

**단점:**
- MHA 대비 약간의 성능 저하
- 구현 복잡도 증가

### Q6: FFN은 왜 4배로 확장했다가 축소하나?

**답변:**
1. **표현력 증가**: 더 높은 차원에서 비선형 변환 수행
2. **병목 구조**: 입력 정보를 압축하는 오토인코더 역할
3. **경험적 최적**: 연구 결과 4배가 효율적인 trade-off

Gated FFN(8/3배)은 파라미터 효율성을 유지하면서 표현력 향상

---

## 참고 자료

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (2020)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (2023)
- vLLM 소스코드: `vllm/model_executor/layers/`
