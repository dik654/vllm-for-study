# PART 6: Distributed Inference (대규모 배포)

> 대규모 LLM을 여러 GPU에서 효율적으로 실행하는 방법

---

## 목차
1. [Tensor Parallelism (TP)](#1-tensor-parallelism-tp)
2. [Pipeline Parallelism (PP)](#2-pipeline-parallelism-pp)
3. [Data Parallelism (DP)](#3-data-parallelism-dp)
4. [Continuous Batching과 TP/PP](#4-continuous-batching과-tppp)
5. [DeepSpeed-Inference vs vLLM](#5-deepspeed-inference-vs-vllm)
6. [Multi-GPU KV Cache Sharding](#6-multi-gpu-kv-cache-sharding)
7. [Speculative Decoding: Draft와 Main Model 상호작용](#7-speculative-decoding-draft와-main-model-상호작용)
8. [Speculative Decoding의 Latency 감소 원리](#8-speculative-decoding의-latency-감소-원리)
9. [면접 예상 질문 및 답변](#9-면접-예상-질문-및-답변)

---

## 1. Tensor Parallelism (TP)

### 1.1 기본 개념

**목적:** 단일 레이어를 여러 GPU에 분할하여 메모리와 연산 분산

```
단일 GPU:
┌─────────────────────────────────────────┐
│  Linear: [d_model, d_hidden]            │
│  전체 가중치가 한 GPU에                 │
└─────────────────────────────────────────┘

Tensor Parallel (TP=2):
┌───────────────────┐ ┌───────────────────┐
│  GPU 0            │ │  GPU 1            │
│  [d_model, d/2]   │ │  [d_model, d/2]   │
│  절반의 가중치    │ │  나머지 절반      │
└───────────────────┘ └───────────────────┘
```

### 1.2 Column Parallel Linear

```
Column Parallel (출력 차원 분할):

입력 X: [batch, seq, d_model]
가중치 W: [d_model, d_hidden]

┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   X ────┬────→ GPU 0: W[:, :d/2]  → Y0                          │
│         │                          ↓                             │
│         │                     [All-Gather]                       │
│         │                          ↓                             │
│         └────→ GPU 1: W[:, d/2:]  → Y1                          │
│                                    ↓                             │
│                                Y = concat(Y0, Y1)                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Y0 = X @ W[:, :d/2]    # [batch, seq, d/2] on GPU 0
Y1 = X @ W[:, d/2:]    # [batch, seq, d/2] on GPU 1
Y = concat(Y0, Y1)     # All-Gather로 결합
```

### 1.3 Row Parallel Linear

```
Row Parallel (입력 차원 분할):

입력 X: [batch, seq, d_hidden] (이미 분할됨)
가중치 W: [d_hidden, d_model]

┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   X0 (GPU 0) ──→ W[:d/2, :] ──→ Y0                              │
│                                  ↓                               │
│                              [All-Reduce]                        │
│                                  ↓                               │
│   X1 (GPU 1) ──→ W[d/2:, :] ──→ Y1                              │
│                                  ↓                               │
│                              Y = Y0 + Y1                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Y0 = X0 @ W[:d/2, :]   # [batch, seq, d_model] partial on GPU 0
Y1 = X1 @ W[d/2:, :]   # [batch, seq, d_model] partial on GPU 1
Y = AllReduce(Y0, Y1)  # 합산
```

### 1.4 Attention의 Tensor Parallelism

```
Multi-Head Attention에서 TP:

┌──────────────────────────────────────────────────────────────────┐
│  총 32 heads를 4 GPU에 분배 (TP=4)                              │
│                                                                  │
│  GPU 0: heads 0-7   (8 heads)                                   │
│  GPU 1: heads 8-15  (8 heads)                                   │
│  GPU 2: heads 16-23 (8 heads)                                   │
│  GPU 3: heads 24-31 (8 heads)                                   │
│                                                                  │
│  각 GPU에서 독립적으로 attention 계산                           │
│  Output projection에서 All-Reduce로 결합                        │
└──────────────────────────────────────────────────────────────────┘

통신 패턴:
1. QKV projection: Column Parallel (통신 없음)
2. Attention: 각 GPU 독립 계산
3. Output projection: Row Parallel (All-Reduce)
```

### 1.5 FFN의 Tensor Parallelism

```
FFN (d → 4d → d):

┌──────────────────────────────────────────────────────────────────┐
│  GPU 0                         │  GPU 1                         │
│                                │                                │
│  W1[:, :2d]  (Column Parallel) │  W1[:, 2d:]                   │
│      ↓                         │      ↓                         │
│  GELU                          │  GELU                          │
│      ↓                         │      ↓                         │
│  W2[:2d, :] (Row Parallel)     │  W2[2d:, :]                   │
│      ↓                         │      ↓                         │
│  All-Reduce ←──────────────────┼───────────→                   │
│      ↓                         │                                │
│  Output                        │                                │
└──────────────────────────────────────────────────────────────────┘

통신: 레이어당 1회 All-Reduce
```

### 1.6 통신 비용 분석

```
Tensor Parallelism 통신:

레이어당 통신량:
- Attention: 1 All-Reduce (d_model × batch × seq × 2)
- FFN: 1 All-Reduce (d_model × batch × seq × 2)

총 레이어당: 4 × d_model × batch × seq bytes

예시 (LLaMA-70B, TP=8, batch=32, seq=2048):
통신량 = 4 × 8192 × 32 × 2048 × 2 = 4.3 GB per layer
32 layers = 137 GB 통신 per forward

NVLink 대역폭 (A100): 600 GB/s
통신 시간: ~230 ms (이론적)

→ TP는 고속 interconnect (NVLink) 필수!
```

---

## 2. Pipeline Parallelism (PP)

### 2.1 기본 개념

**목적:** 모델의 레이어들을 순차적으로 여러 GPU에 분배

```
Pipeline Parallel (PP=4):

┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│  GPU 0  │→│  GPU 1  │→│  GPU 2  │→│  GPU 3  │
│ L0-L7   │ │ L8-L15  │ │ L16-L23 │ │ L24-L31 │
│ Stage 0 │ │ Stage 1 │ │ Stage 2 │ │ Stage 3 │
└─────────┘ └─────────┘ └─────────┘ └─────────┘

각 GPU가 연속된 레이어 블록 담당
activation만 전달 (가중치 통신 없음)
```

### 2.2 Pipeline Bubble 문제

```
Naive Pipeline:

Time →
GPU 0: [B0_S0][idle ][idle ][idle ]
GPU 1: [idle ][B0_S1][idle ][idle ]
GPU 2: [idle ][idle ][B0_S2][idle ]
GPU 3: [idle ][idle ][idle ][B0_S3]

문제: GPU 활용률 = 1/4 = 25%!
대부분의 시간 GPU가 놀고 있음 (bubble)
```

### 2.3 Microbatch Pipeline

```
Microbatch로 bubble 감소:

배치를 4개 microbatch로 분할: m0, m1, m2, m3

Time →
GPU 0: [m0_S0][m1_S0][m2_S0][m3_S0][idle][idle][idle]
GPU 1: [idle ][m0_S1][m1_S1][m2_S1][m3_S1][idle][idle]
GPU 2: [idle ][idle ][m0_S2][m1_S2][m2_S2][m3_S2][idle]
GPU 3: [idle ][idle ][idle ][m0_S3][m1_S3][m2_S3][m3_S3]

활용률 개선: 4/(4+3) ≈ 57%

일반화: 활용률 = M / (M + P - 1)
M = microbatch 수, P = pipeline stages
```

### 2.4 1F1B Schedule

```
1F1B (One Forward One Backward):
추론에서는 Forward만 있으므로 단순화

┌──────────────────────────────────────────────────────────────────┐
│  Steady State에서:                                              │
│                                                                  │
│  GPU 0: [F0][F1][F2][F3][F4][F5]...                             │
│  GPU 1:    [F0][F1][F2][F3][F4][F5]...                          │
│  GPU 2:       [F0][F1][F2][F3][F4][F5]...                       │
│  GPU 3:          [F0][F1][F2][F3][F4][F5]...                    │
│                                                                  │
│  Warmup과 Cooldown 제외하면 100% 활용 가능                     │
└──────────────────────────────────────────────────────────────────┘
```

### 2.5 PP의 장단점

```
장점:
- 낮은 통신 대역폭 요구 (activation만 전달)
- PCIe로도 가능 (TP보다 유연)
- 메모리 효율적 (각 GPU가 일부 레이어만)

단점:
- Pipeline bubble로 인한 효율 손실
- Latency 증가 (순차 처리)
- 작은 배치에서 비효율

추론에서:
- Prefill: 큰 배치로 bubble 최소화 가능
- Decode: 작은 배치(batch × 1 token)로 bubble 큼
```

### 2.6 TP vs PP 선택

```
┌──────────────────────────────────────────────────────────────────┐
│  상황              │  추천             │  이유                   │
├──────────────────────────────────────────────────────────────────┤
│  NVLink 있음       │  TP 우선          │  고속 통신 활용         │
│  PCIe만 있음       │  PP 우선          │  낮은 통신 요구         │
│  작은 배치         │  TP               │  PP bubble 심함         │
│  큰 배치           │  PP 가능          │  bubble 최소화          │
│  latency 중요      │  TP               │  PP는 순차 지연         │
│  throughput 중요   │  둘 다 OK         │  상황에 따라            │
└──────────────────────────────────────────────────────────────────┘

vLLM 기본: TP 우선, 필요시 PP 추가
```

---

## 3. Data Parallelism (DP)

### 3.1 기본 개념

```
Data Parallelism:
각 GPU가 전체 모델 복제본을 가지고, 다른 데이터 처리

┌─────────────────────────────────────────────────────────────────┐
│  GPU 0: 전체 모델 복제본, 배치 0-31                            │
│  GPU 1: 전체 모델 복제본, 배치 32-63                           │
│  GPU 2: 전체 모델 복제본, 배치 64-95                           │
│  GPU 3: 전체 모델 복제본, 배치 96-127                          │
│                                                                 │
│  각 GPU 독립적으로 추론, 결과 수집                             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 추론에서의 DP

```
추론 시 DP 특징:
- 학습과 달리 gradient 동기화 불필요
- 각 GPU가 완전히 독립적
- Load balancer가 요청 분배

장점:
- 구현 가장 단순
- 통신 거의 없음
- 선형 throughput 증가

단점:
- 각 GPU에 전체 모델 필요
- 큰 모델은 단일 GPU에 안 맞음
- KV cache도 각 GPU에 중복
```

### 3.3 Hybrid Parallelism

```
실제 대규모 배포: TP + PP + DP 조합

예시: 8개 노드, 노드당 8 GPU = 64 GPU

┌──────────────────────────────────────────────────────────────────┐
│  DP 그룹 0 (32 GPU):                                            │
│  ├── PP Stage 0 (8 GPU): TP=8                                   │
│  ├── PP Stage 1 (8 GPU): TP=8                                   │
│  ├── PP Stage 2 (8 GPU): TP=8                                   │
│  └── PP Stage 3 (8 GPU): TP=8                                   │
│                                                                  │
│  DP 그룹 1 (32 GPU): 동일 구조                                  │
└──────────────────────────────────────────────────────────────────┘

구성: DP=2 × PP=4 × TP=8 = 64 GPU

노드 내: TP (NVLink 활용)
노드 간: PP (낮은 통신)
복제: DP (독립 처리)
```

---

## 4. Continuous Batching과 TP/PP

### 4.1 TP에서의 Continuous Batching

```
Tensor Parallel에서 배치 재구성:

모든 TP rank가 동일한 배치 구성 필요!

┌──────────────────────────────────────────────────────────────────┐
│  Step 1:                                                        │
│  GPU 0,1,2,3: batch = [req1, req2, req3, req4]                 │
│               (모든 GPU 동일)                                   │
│                                                                  │
│  Step 2: req2 완료                                              │
│  Scheduler: 새로운 req5 추가 결정                               │
│  GPU 0,1,2,3: batch = [req1, req5, req3, req4]                 │
│               (모든 GPU가 동시에 업데이트)                      │
│                                                                  │
│  → Scheduler가 central하게 배치 결정                           │
│  → 모든 GPU에 브로드캐스트                                     │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 PP에서의 Continuous Batching

```
Pipeline Parallel에서:

문제: 각 stage가 다른 microbatch 처리 중

┌──────────────────────────────────────────────────────────────────┐
│  해결책 1: Stage별 독립 스케줄링                                │
│  - 각 stage가 완료된 요청 제거, 새 요청 추가                   │
│  - 복잡하고 동기화 어려움                                       │
│                                                                  │
│  해결책 2: Global 스케줄링 (vLLM 방식)                         │
│  - Stage 0 (첫 번째)에서 배치 결정                             │
│  - 결정된 배치가 pipeline 따라 전파                            │
│  - 구현 단순, 일관성 보장                                       │
└──────────────────────────────────────────────────────────────────┘

실제 동작:
Stage 0: [req1, req2, req3] 처리, req2 완료 → req5로 교체
         [req1, req5, req3]을 Stage 1으로 전달
Stage 1: 받은 대로 처리
...
```

### 4.3 KV Cache 일관성

```
TP에서 KV Cache:

┌──────────────────────────────────────────────────────────────────┐
│  각 GPU가 일부 head의 KV만 저장                                 │
│                                                                  │
│  GPU 0: heads 0-7의 KV cache                                   │
│  GPU 1: heads 8-15의 KV cache                                  │
│  GPU 2: heads 16-23의 KV cache                                 │
│  GPU 3: heads 24-31의 KV cache                                 │
│                                                                  │
│  요청 완료/추가 시 모든 GPU가 동시에 업데이트                  │
│  Block table은 공유 (어떤 블록이 어떤 요청에 할당됐는지)       │
└──────────────────────────────────────────────────────────────────┘

PP에서 KV Cache:

┌──────────────────────────────────────────────────────────────────┐
│  각 Stage가 자신의 레이어 KV만 저장                             │
│                                                                  │
│  Stage 0 (GPU 0): layers 0-7의 KV cache                        │
│  Stage 1 (GPU 1): layers 8-15의 KV cache                       │
│  Stage 2 (GPU 2): layers 16-23의 KV cache                      │
│  Stage 3 (GPU 3): layers 24-31의 KV cache                      │
│                                                                  │
│  각 stage 독립적으로 KV cache 관리                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. DeepSpeed-Inference vs vLLM

### 5.1 DeepSpeed-Inference 특징

```
DeepSpeed-Inference:
┌──────────────────────────────────────────────────────────────────┐
│  핵심 기능:                                                     │
│  1. Tensor Parallelism                                          │
│  2. Kernel fusion (transformer fusion)                          │
│  3. INT8/FP16 quantization                                      │
│  4. Dynamic batching                                            │
│                                                                  │
│  장점:                                                          │
│  - 학습과 추론 통합 프레임워크                                 │
│  - 다양한 최적화 기법 통합                                     │
│  - 큰 생태계 (Hugging Face 통합)                               │
│                                                                  │
│  단점:                                                          │
│  - Static batching (iteration-level 아님)                       │
│  - KV cache 관리 비효율                                        │
│  - Continuous batching 미지원                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 vLLM 특징

```
vLLM:
┌──────────────────────────────────────────────────────────────────┐
│  핵심 기능:                                                     │
│  1. PagedAttention (효율적 KV cache)                            │
│  2. Continuous batching                                         │
│  3. Tensor/Pipeline Parallelism                                 │
│  4. Speculative decoding                                        │
│                                                                  │
│  장점:                                                          │
│  - 최고 수준의 throughput                                      │
│  - 메모리 효율성 (PagedAttention)                              │
│  - 프로덕션 최적화                                             │
│                                                                  │
│  단점:                                                          │
│  - 추론 전용 (학습 불가)                                       │
│  - DeepSpeed보다 적은 quantization 옵션                        │
└──────────────────────────────────────────────────────────────────┘
```

### 5.3 성능 비교

```
벤치마크 (LLaMA-13B, A100):

┌──────────────────────────────────────────────────────────────────┐
│  메트릭                │  DeepSpeed  │  vLLM      │  차이       │
├──────────────────────────────────────────────────────────────────┤
│  Throughput (tok/s)   │  ~800       │  ~2400     │  3x ↑       │
│  Latency P50 (ms)     │  ~50        │  ~30       │  40% ↓      │
│  Memory utilization   │  ~40%       │  ~95%      │  2.4x ↑     │
│  Max batch size       │  32         │  256       │  8x ↑       │
└──────────────────────────────────────────────────────────────────┘

vLLM 우위 이유:
1. PagedAttention으로 메모리 효율 극대화
2. Continuous batching으로 GPU 활용률 향상
3. 추론에 특화된 최적화
```

### 5.4 사용 시나리오

```
DeepSpeed-Inference 선택:
- 학습과 추론 파이프라인 통합 필요
- 다양한 quantization 실험
- 기존 DeepSpeed 학습 코드와 연계

vLLM 선택:
- 프로덕션 서빙 (높은 throughput)
- 메모리 제한 환경
- 긴 컨텍스트 지원 필요
- Continuous batching 필수

TensorRT-LLM 선택:
- 최대 성능 필요 (NVIDIA 최적화)
- 배포 복잡도 감수 가능
- NVIDIA GPU 전용 환경
```

---

## 6. Multi-GPU KV Cache Sharding

### 6.1 KV Cache 분산 전략

```
TP에서의 KV Cache Sharding:

┌──────────────────────────────────────────────────────────────────┐
│  방식 1: Head-wise sharding (vLLM 기본)                         │
│                                                                  │
│  총 32 heads, TP=4:                                             │
│  GPU 0: heads 0-7 의 K, V                                       │
│  GPU 1: heads 8-15 의 K, V                                      │
│  GPU 2: heads 16-23 의 K, V                                     │
│  GPU 3: heads 24-31 의 K, V                                     │
│                                                                  │
│  장점: Attention 계산 시 통신 불필요                           │
│  단점: 요청 간 KV 공유(prefix caching) 복잡                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  방식 2: Sequence-wise sharding                                 │
│                                                                  │
│  총 시퀀스 2048 tokens, TP=4:                                   │
│  GPU 0: tokens 0-511의 모든 head KV                            │
│  GPU 1: tokens 512-1023의 모든 head KV                         │
│  ...                                                            │
│                                                                  │
│  장점: 긴 시퀀스 분산                                          │
│  단점: Attention 시 All-Gather 필요                            │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Block Table 동기화

```
PagedAttention + TP:

모든 GPU가 동일한 block table 유지

┌──────────────────────────────────────────────────────────────────┐
│  Request 1: block_table = [0, 3, 7, 12]                         │
│                                                                  │
│  GPU 0: block 0의 heads 0-7, block 3의 heads 0-7, ...          │
│  GPU 1: block 0의 heads 8-15, block 3의 heads 8-15, ...        │
│  GPU 2: block 0의 heads 16-23, ...                              │
│  GPU 3: block 0의 heads 24-31, ...                              │
│                                                                  │
│  Block table은 공유되지만,                                      │
│  각 GPU는 자신의 head 부분만 저장                              │
└──────────────────────────────────────────────────────────────────┘

Scheduler 결정 브로드캐스트:
1. Rank 0에서 스케줄링 결정
2. 모든 rank에 block 할당/해제 정보 전파
3. 각 rank가 로컬 KV cache 업데이트
```

### 6.3 Prefix Caching in Multi-GPU

```
도전 과제: 여러 GPU에서 prefix 공유

┌──────────────────────────────────────────────────────────────────┐
│  시나리오:                                                       │
│  Req1: "System prompt: You are helpful..."                      │
│  Req2: "System prompt: You are helpful..." (동일 prefix)        │
│                                                                  │
│  단일 GPU: prefix의 KV block 공유 간단                          │
│                                                                  │
│  Multi-GPU:                                                      │
│  - 각 GPU가 prefix의 일부(자기 heads)만 가짐                   │
│  - 모든 GPU에서 동시에 공유 설정 필요                          │
│  - Block hash가 모든 GPU에서 일치해야 함                        │
└──────────────────────────────────────────────────────────────────┘

해결책:
1. Global block hash 계산 (content 기반)
2. Rank 0에서 캐시 히트 판단
3. 결정을 모든 rank에 브로드캐스트
4. 각 rank가 로컬 ref_cnt 업데이트
```

---

## 7. Speculative Decoding: Draft와 Main Model 상호작용

### 7.1 Speculative Decoding 개념

```
기본 아이디어:
작은 draft model이 여러 토큰을 빠르게 예측
큰 main model이 한 번에 검증

┌──────────────────────────────────────────────────────────────────┐
│  기존 (Autoregressive):                                         │
│  Main: [t1] → [t2] → [t3] → [t4] → [t5]                        │
│  시간: 5 × latency_main                                         │
│                                                                  │
│  Speculative:                                                    │
│  Draft: [t1, t2, t3, t4, t5] (한 번에 5개 예측)                │
│  Main: [검증] → 3개 accept, 2개 reject                          │
│  시간: latency_draft + latency_main                             │
│  → 잠재적으로 5배 빠름!                                         │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 Draft Model 선택

```
Draft Model 옵션:

1. 작은 동일 아키텍처:
   Main: LLaMA-70B
   Draft: LLaMA-7B
   장점: 높은 acceptance rate
   단점: 별도 모델 로드

2. Early exit:
   Main의 앞 레이어만 사용
   장점: 추가 모델 불필요
   단점: 낮은 정확도

3. Medusa heads:
   Main model에 추가 head
   장점: 병렬 예측
   단점: fine-tuning 필요

4. EAGLE:
   Auto-regressive head 추가
   장점: 높은 acceptance
   단점: 학습 필요
```

### 7.3 상호작용 프로세스

```python
# Speculative Decoding 프로세스

def speculative_decode(prompt, main_model, draft_model, K=5):
    """
    K: draft model이 예측하는 토큰 수
    """
    tokens = prompt

    while not done:
        # 1. Draft model이 K개 토큰 예측
        draft_tokens = []
        draft_probs = []
        for _ in range(K):
            logits = draft_model(tokens + draft_tokens)
            token = sample(logits)
            draft_tokens.append(token)
            draft_probs.append(softmax(logits))

        # 2. Main model이 한 번에 검증
        # K+1개 position에서 logits 계산
        main_logits = main_model(tokens + draft_tokens)  # 배치 처리

        # 3. Rejection sampling으로 accept/reject 결정
        accepted = []
        for i, (draft_tok, draft_p) in enumerate(zip(draft_tokens, draft_probs)):
            main_p = softmax(main_logits[i])

            # Acceptance probability
            r = random.random()
            accept_prob = min(1, main_p[draft_tok] / draft_p[draft_tok])

            if r < accept_prob:
                accepted.append(draft_tok)
            else:
                # Reject 시 main model에서 재샘플링
                adjusted_p = normalize(max(0, main_p - draft_p))
                new_token = sample(adjusted_p)
                accepted.append(new_token)
                break  # 이후 draft 토큰은 무효

        tokens.extend(accepted)

    return tokens
```

### 7.4 vLLM에서의 Speculative Decoding

```python
# vllm/v1/spec_decode/ 구조

class SpeculativeDecoder:
    def __init__(self, main_model, draft_model, num_speculative_tokens):
        self.main = main_model
        self.draft = draft_model
        self.K = num_speculative_tokens

    def generate_step(self, requests):
        # 1. Draft 단계: 각 요청에 K개 토큰 예측
        draft_outputs = self.draft_generate(requests, self.K)

        # 2. Verification 단계: Main model이 검증
        # spec tokens를 포함한 입력으로 main model 실행
        verification_input = self.prepare_verification_input(
            requests, draft_outputs
        )
        main_outputs = self.main.forward(verification_input)

        # 3. Accept/Reject 결정
        accepted_tokens = self.verify_and_accept(
            draft_outputs, main_outputs
        )

        return accepted_tokens
```

---

## 8. Speculative Decoding의 Latency 감소 원리

### 8.1 수학적 분석

```
표기:
- T_main: Main model 1 step latency
- T_draft: Draft model 1 step latency
- K: 예측 토큰 수
- α: Average acceptance rate (0 < α ≤ 1)

기존 방식:
N 토큰 생성 시간 = N × T_main

Speculative Decoding:
각 iteration에서 평균 accepted 토큰 = K × α + 1
(최소 1개는 main에서 생성)

총 iterations = N / (K × α + 1)
총 시간 = N / (K × α + 1) × (K × T_draft + T_main)

Speedup = (N × T_main) / (N / (K × α + 1) × (K × T_draft + T_main))
        = (K × α + 1) × T_main / (K × T_draft + T_main)
```

### 8.2 조건 분석

```
Speedup > 1 조건:

(K × α + 1) × T_main > K × T_draft + T_main
K × α × T_main > K × T_draft
α > T_draft / T_main

예시:
T_main = 50ms (LLaMA-70B)
T_draft = 5ms (LLaMA-7B)
필요한 α > 5/50 = 10%

실제 acceptance rate: 60-80% (유사한 모델일 때)
→ 충분히 이득!
```

### 8.3 Speedup 계산 예시

```
설정:
- Main: 50ms/token
- Draft: 5ms/token
- K = 4 (4개 토큰 예측)
- α = 0.7 (70% acceptance)

계산:
평균 accepted per iteration = 4 × 0.7 + 1 = 3.8 tokens
Time per iteration = 4 × 5 + 50 = 70ms

기존: 3.8 tokens × 50ms = 190ms
Speculative: 70ms

Speedup = 190 / 70 = 2.7x

실제 100 tokens 생성:
기존: 100 × 50ms = 5000ms
Speculative: ceil(100 / 3.8) × 70ms ≈ 1842ms
Speedup = 2.7x
```

### 8.4 최적 K 선택

```
K가 작으면:
- Draft overhead 작음
- 한 번에 accept할 수 있는 토큰 적음
- → 낮은 speedup

K가 크면:
- Draft overhead 큼 (K × T_draft)
- Accept rate가 낮아질 수 있음 (누적 확률)
- → 어느 시점부터 speedup 감소

최적 K:
d(Speedup)/dK = 0을 푸는 것
실험적으로 K = 3~5가 최적인 경우 많음
```

### 8.5 실제 고려사항

```
┌──────────────────────────────────────────────────────────────────┐
│  1. Memory overhead:                                            │
│     - Draft model 가중치: 추가 메모리                           │
│     - 두 모델의 KV cache 필요할 수 있음                        │
│                                                                  │
│  2. Batching 영향:                                              │
│     - 배치 내 각 요청의 accepted 수가 다름                     │
│     - Padding overhead 발생 가능                                │
│                                                                  │
│  3. Task 의존성:                                                │
│     - 코드 생성: 높은 acceptance (예측 가능)                   │
│     - 창의적 글쓰기: 낮은 acceptance (불확실)                  │
│                                                                  │
│  4. Throughput vs Latency:                                      │
│     - Latency 감소에 효과적                                    │
│     - Throughput은 오히려 감소할 수 있음 (draft 오버헤드)      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. 면접 예상 질문 및 답변

### Q1: TP와 PP의 차이점과 각각 언제 사용하나요?

**답변:**

**Tensor Parallelism (TP):**
- 단일 레이어를 여러 GPU에 분할
- 레이어 내에서 All-Reduce 통신 필요
- **고속 interconnect (NVLink) 필수**
- Latency 영향 적음

**Pipeline Parallelism (PP):**
- 레이어들을 순차적으로 분배
- Activation만 전달 (낮은 대역폭)
- Pipeline bubble로 효율 손실
- **PCIe로도 가능**

```
선택 기준:
- NVLink 있으면 → TP 우선
- 큰 배치 + PCIe → PP 가능
- 작은 배치 + latency 중요 → TP
```

### Q2: Continuous Batching이 분산 환경에서 어떻게 동작하나요?

**답변:**
TP에서는 **모든 GPU가 동일한 배치 구성**을 가져야 합니다.

1. **Central Scheduler**: Rank 0에서 배치 구성 결정
2. **Broadcast**: 결정 사항을 모든 rank에 전파
3. **Synchronized Update**: 모든 GPU가 동시에 배치 업데이트

```python
# 스케줄러가 결정 후
schedule_output = scheduler.schedule()
# 모든 rank에 브로드캐스트
dist.broadcast_object(schedule_output, src=0)
# 각 rank가 동일하게 실행
```

PP에서는 **Stage 0에서 결정**하고 pipeline을 따라 전파합니다.

### Q3: Speculative Decoding이 효과적인 조건은?

**답변:**
Speedup > 1 조건: **α > T_draft / T_main**

```
필요한 acceptance rate = Draft latency / Main latency

예: Main 50ms, Draft 5ms
필요 acceptance rate > 10%
실제: 60-80% (유사 모델)
→ 2-3배 speedup 가능
```

**효과적인 경우:**
- Draft가 Main보다 훨씬 빠름 (10배+)
- 높은 acceptance rate (예측 가능한 task)
- Latency가 중요한 사용 케이스

**비효과적인 경우:**
- 창의적/불확실한 생성 (낮은 acceptance)
- Throughput이 중요한 경우 (draft overhead)

### Q4: Multi-GPU에서 KV Cache는 어떻게 관리되나요?

**답변:**
**TP의 경우:**
```
각 GPU가 자신의 head 부분만 저장
GPU 0: heads 0-7의 KV
GPU 1: heads 8-15의 KV
...

Block table은 공유되지만 실제 KV는 분산
Scheduler가 모든 GPU의 할당을 동기화
```

**PP의 경우:**
```
각 Stage가 자신의 레이어 KV만 저장
독립적인 메모리 관리
Activation 전달 시에만 통신
```

### Q5: DeepSpeed-Inference 대비 vLLM의 장점은?

**답변:**

| 측면 | DeepSpeed | vLLM |
|------|-----------|------|
| Batching | Static | **Continuous** |
| KV Cache | 단순 | **PagedAttention** |
| Memory 효율 | ~40% | **~95%** |
| Throughput | 1x | **2-3x** |

vLLM 장점:
1. **PagedAttention**: 메모리 파편화 제거
2. **Continuous batching**: GPU 유휴 시간 최소화
3. **프로덕션 최적화**: 실제 서빙에 특화

DeepSpeed 장점:
- 학습-추론 통합
- 다양한 quantization

### Q6: Speculative Decoding에서 Main Model이 여러 토큰을 한 번에 처리하는 이유는?

**답변:**
**핵심 인사이트:** Main model forward pass의 비용은 토큰 수에 크게 의존하지 않습니다.

```
Main model latency:
1 token: ~50ms
5 tokens: ~52ms (거의 동일!)

이유:
- Memory-bound 연산 (KV cache 읽기)
- 배치 처리 효율
- GPU의 병렬성 활용
```

따라서 **1번의 main forward로 K개 draft 토큰을 검증**하면:
- K번 forward 비용: K × 50ms = 250ms
- 1번 forward 비용: ~52ms
- 4.8배 절약!

---

## 참고 자료

- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://arxiv.org/abs/2207.00032)
- vLLM 소스코드:
  - `vllm/distributed/` - 분산 처리 관련
  - `vllm/v1/spec_decode/` - Speculative decoding
