# 최신 AI 논문 정리 (2024-2025)

## 목차
1. [Large Language Models](#large-language-models)
2. [Multimodal Models](#multimodal-models)
3. [Efficient Architectures](#efficient-architectures)
4. [Reasoning & Planning](#reasoning--planning)
5. [Alignment & Safety](#alignment--safety)
6. [Applications](#applications)

---

## Large Language Models

### 1. "Llama 3: Open Foundation and Fine-Tuned Chat Models" (Meta AI, 2024년 4월)

**핵심 아이디어**:
- 15T 토큰으로 학습한 8B, 70B 파라미터 오픈소스 모델
- Grouped Query Attention (GQA) 사용으로 추론 효율 개선
- 128K 컨텍스트 윈도우 지원

**방법론**:
```python
"""
Grouped Query Attention (GQA)

기존 MHA (Multi-Head Attention):
- Query heads: 32
- Key heads: 32
- Value heads: 32
→ 메모리: 많음, 속도: 느림

GQA:
- Query heads: 32
- Key/Value heads: 8 (그룹화)
→ 메모리: 75% 절감, 속도: 2배 향상
"""

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model=4096, num_q_heads=32, num_kv_heads=8):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads

        # Q는 32개 헤드, K/V는 8개 헤드
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)

    def forward(self, x):
        B, T, C = x.shape

        # Q: (B, T, 32, head_dim)
        Q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim)

        # K, V: (B, T, 8, head_dim)
        K = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        V = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # K, V를 반복하여 Q와 매칭 (8 → 32)
        # 각 K/V 헤드를 4번 반복 (32 / 8 = 4)
        K = K.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=2)
        V = V.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=2)

        # 이제 Q, K, V 모두 (B, T, 32, head_dim)
        # 일반적인 어텐션 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        return out
```

**주요 결과**:
- Llama 3 70B: MMLU 82.0% (GPT-3.5급 성능)
- Llama 3 8B: MMLU 68.4% (기존 7B 모델 대비 +10%)
- 추론 속도: GQA로 2배 향상

**실전 의미**:
- 오픈소스 모델이 상업 모델 수준 도달
- 작은 모델(8B)도 실용적 성능 달성
- 온프레미스 배포 가능

---

### 2. "Mixtral 8x7B: A Sparse Mixture of Experts" (Mistral AI, 2024년 1월)

**핵심 아이디어**:
- 46.7B 파라미터 모델이지만, 실행 시 12.9B만 활성화 (MoE)
- 8개의 7B 전문가 모델 + 라우터

**방법론**:
```python
"""
Mixture of Experts (MoE)

핵심: 입력마다 다른 전문가 활성화
→ 큰 모델의 성능 + 작은 모델의 속도
"""

class MixtralMoE(nn.Module):
    def __init__(self, d_model=4096, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 전문가 FFN들
        self.experts = nn.ModuleList([
            FeedForward(d_model) for _ in range(num_experts)
        ])

        # 라우터: 어떤 전문가를 사용할지 결정
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: (batch, seq_len, d_model)

        # Step 1: 라우터로 전문가 점수 계산
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Step 2: Top-K 전문가 선택
        # 각 토큰마다 2개 전문가 선택
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # top_k_indices: (batch, seq_len, 2)

        # 정규화 (선택된 전문가들의 확률 합 = 1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Step 3: 선택된 전문가들의 출력 계산 및 가중 합
        output = torch.zeros_like(x)

        for i in range(self.top_k):
            # i번째로 선택된 전문가의 인덱스
            expert_idx = top_k_indices[:, :, i]  # (batch, seq_len)
            expert_weight = top_k_probs[:, :, i:i+1]  # (batch, seq_len, 1)

            # 각 위치에서 선택된 전문가 실행
            # 실제로는 배치 처리를 위해 더 복잡한 구현 필요
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    # 해당 전문가가 선택된 토큰들
                    expert_input = x[mask]
                    expert_output = self.experts[expert_id](expert_input)

                    # 가중치 적용하여 출력에 추가
                    output[mask] += expert_weight[mask] * expert_output

        return output

"""
실전 예시:

입력: "Explain quantum computing"

Router 출력:
- Expert 0 (Physics): 0.45
- Expert 1 (CS): 0.35
- Expert 2 (Math): 0.10
- ... (나머지 낮은 점수)

선택: Expert 0 (0.45) + Expert 1 (0.35)
→ 6개 전문가는 비활성화 (계산 절약!)
"""
```

**주요 결과**:
- MMLU: 70.6% (Llama 2 70B와 동등)
- 실제 활성 파라미터: 12.9B (Llama 2 13B 수준)
- 추론 속도: Llama 2 70B보다 6배 빠름

**실전 의미**:
- 비용 효율적인 대규모 모델 배포 가능
- 전문가 특화로 도메인별 성능 향상
- 학습 시 부하 분산 (전문가별 병렬 학습)

---

### 3. "Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens" (Google, 2024년 2월)

**핵심 아이디어**:
- 최대 10M 토큰 컨텍스트 (실용적으로는 1M)
- 네이티브 멀티모달: 텍스트, 이미지, 오디오, 비디오 통합 처리
- MoE + Ring Attention으로 긴 컨텍스트 효율 처리

**방법론**:
```python
"""
Ring Attention: 긴 컨텍스트 효율 처리

문제: 1M 토큰 Self-Attention = O(N^2) = 1조 연산!

해결: Ring Attention
- 시퀀스를 여러 디바이스에 분산
- 각 디바이스가 자기 부분만 처리
- Ring 구조로 KV 교환
"""

class RingAttention:
    def __init__(self, world_size=8):  # 8개 GPU
        self.world_size = world_size

    def forward(self, Q, K, V):
        """
        Q, K, V: (batch, seq_len // world_size, d_model)
        각 GPU가 전체 시퀀스의 1/8만 가짐
        """
        rank = get_rank()  # 현재 GPU ID
        local_seq_len = Q.shape[1]
        d_k = Q.shape[-1]

        # 최종 출력 초기화
        output = torch.zeros_like(Q)
        lse = torch.zeros(Q.shape[0], Q.shape[1], 1)  # Log-Sum-Exp (안정성)

        # Ring 순회: 각 GPU의 KV를 차례로 받음
        K_block = K.clone()
        V_block = V.clone()

        for step in range(self.world_size):
            # Step 1: 현재 KV 블록으로 어텐션 계산
            scores = torch.matmul(Q, K_block.transpose(-2, -1)) / math.sqrt(d_k)
            # scores: (batch, local_seq_len, local_seq_len)

            # Step 2: Softmax (Log-Sum-Exp 기법으로 안정적 계산)
            scores_max = scores.max(dim=-1, keepdim=True)[0]
            scores_exp = torch.exp(scores - scores_max)

            # Step 3: 현재 블록의 출력
            block_output = torch.matmul(scores_exp, V_block)

            # Step 4: 누적 (여러 블록의 결과 합침)
            lse_new = torch.log(torch.sum(scores_exp, dim=-1, keepdim=True)) + scores_max
            scale = torch.exp(lse - lse_new)

            output = output * scale + block_output
            lse = lse_new

            # Step 5: 다음 GPU로 KV 전송 (Ring)
            if step < self.world_size - 1:
                K_block = send_to_next_rank(K_block)
                V_block = send_to_next_rank(V_block)

        # 정규화
        output = output / torch.exp(lse)
        return output

"""
예시: 1M 토큰 처리

단일 GPU: 1M x 1M = 1조 연산, 메모리 부족

Ring Attention (8 GPU):
- 각 GPU: 125K 토큰 담당
- 각 GPU 연산: 125K x 125K x 8 = 125억 연산
- 총 연산: 1000억 (10배 효율적)
- 메모리: 각 GPU 125K만 저장
"""
```

**주요 결과**:
- 1M 토큰 컨텍스트에서 99.7% Recall (Needle-in-Haystack 테스트)
- 1시간 비디오 전체 이해 가능
- MMLU: 90.0% (Gemini Ultra)

**실전 의미**:
- 긴 문서 전체 분석 (법률 계약서, 연구 논문)
- 장편 비디오 이해 (회의, 강의)
- 코드베이스 전체 컨텍스트 활용

---

## Multimodal Models

### 4. "GPT-4V(ision) System Card" (OpenAI, 2023년 9월 → 2024년 업데이트)

**핵심 아이디어**:
- LLM에 Vision Encoder 통합
- Zero-shot 이미지 이해 (학습 없이 다양한 시각 작업)
- OCR, 차트 해석, 의료 영상 등 범용 성능

**방법론**:
```python
"""
GPT-4V 아키텍처 (추정)

[Image] → CLIP Encoder (ViT) → Visual Tokens
                                      ↓
                                  Projection
                                      ↓
[Text]  → Tokenizer → [Visual Tokens + Text Tokens] → GPT-4 Decoder

핵심: 이미지를 "특별한 토큰"으로 취급
"""

class GPT4Vision:
    def __init__(self):
        # 1. Vision Encoder (CLIP ViT-G/14)
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-giant-patch14")

        # 2. Vision-Language Projection
        # CLIP 출력 (1024d) → GPT-4 임베딩 공간 (12288d 추정)
        self.vision_projection = nn.Linear(1024, 12288)

        # 3. GPT-4 Decoder
        self.llm = GPT4Model()

    def process_image(self, image):
        """
        이미지 → 토큰 시퀀스

        입력: (3, 224, 224) 이미지
        출력: (256, 12288) 토큰
        """
        # Step 1: 이미지를 패치로 분할 (14x14 패치)
        # 224 / 14 = 16 → 16x16 = 256 패치

        # Step 2: Vision Encoder
        with torch.no_grad():
            vision_outputs = self.vision_encoder(image)
            # vision_outputs: (256, 1024)

        # Step 3: Projection to LLM space
        visual_tokens = self.vision_projection(vision_outputs)
        # visual_tokens: (256, 12288)

        return visual_tokens

    def generate(self, image, text):
        """
        이미지 + 텍스트 → 응답

        예: image="chart.png", text="이 차트를 설명해주세요"
        """
        # 이미지 처리
        visual_tokens = self.process_image(image)  # (256, 12288)

        # 텍스트 토큰화
        text_tokens = self.llm.tokenizer(text)
        text_embeddings = self.llm.embedding(text_tokens)  # (T, 12288)

        # 결합
        # [<img_1>, <img_2>, ..., <img_256>, "이", "차트를", ...]
        combined_tokens = torch.cat([visual_tokens, text_embeddings], dim=0)

        # GPT-4로 생성
        output = self.llm.generate(combined_tokens)
        return output
```

**주요 결과**:
- MMMU (Massive Multitask Multimodal Understanding): 56.8% (SOTA)
- OCR 정확도: 95%+ (복잡한 문서)
- 차트/그래프 해석: 정확한 수치 읽기

**실전 의미**:
- 문서 자동화 (영수증, 계약서 파싱)
- 데이터 시각화 해석
- UI/UX 리뷰 자동화

---

### 5. "LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge" (2024년 1월)

**핵심 아이디어**:
- 오픈소스 Vision-Language Model
- Dynamic High Resolution: 이미지를 여러 해상도로 처리
- LLaMA 3 기반으로 강력한 언어 이해

**방법론**:
```python
"""
Dynamic High Resolution

기존 ViT: 고정 해상도 (336x336)
→ 고해상도 이미지 정보 손실

LLaVA-NeXT: 이미지를 여러 조각으로 나눔
→ 각 조각을 고해상도로 처리
"""

class DynamicHighResolution:
    def __init__(self, base_resolution=336):
        self.base_resolution = base_resolution
        self.vision_encoder = CLIPVisionModel()

    def process_image(self, image):
        """
        이미지를 동적으로 분할

        예: 1344x896 이미지
        → 4개 조각 (336x336 각각) + 1개 전체 (축소)
        """
        H, W = image.shape[-2:]

        # Step 1: 최적 분할 결정
        # 목표: base_resolution에 가장 가깝게 나누기
        num_h = round(H / self.base_resolution)
        num_w = round(W / self.base_resolution)

        # Step 2: 조각들 생성
        patches = []
        patch_h = H // num_h
        patch_w = W // num_w

        for i in range(num_h):
            for j in range(num_w):
                patch = image[
                    :,
                    i * patch_h:(i + 1) * patch_h,
                    j * patch_w:(j + 1) * patch_w
                ]
                # 리사이즈
                patch_resized = F.interpolate(
                    patch,
                    size=(self.base_resolution, self.base_resolution)
                )
                patches.append(patch_resized)

        # Step 3: 전체 이미지도 추가 (글로벌 컨텍스트)
        global_patch = F.interpolate(
            image,
            size=(self.base_resolution, self.base_resolution)
        )
        patches.append(global_patch)

        # Step 4: 각 패치 인코딩
        visual_features = []
        for patch in patches:
            features = self.vision_encoder(patch)  # (256, 1024)
            visual_features.append(features)

        # Step 5: 모든 특징 결합
        # (num_patches * 256, 1024)
        combined_features = torch.cat(visual_features, dim=0)

        return combined_features

"""
예시: 1344x896 이미지 (4:3 비율)

기존 ViT:
- 336x336으로 리사이즈 → 정보 손실

LLaVA-NeXT:
- 4개 패치 (336x336) + 1개 글로벌
- 총 5 * 256 = 1280 토큰
- 세밀한 디테일 보존!
"""
```

**주요 결과**:
- OCR 벤치마크: TextVQA 76.7% (+15% vs LLaVA 1.5)
- 고해상도 문서: 99% 정확도
- 오픈소스이면서 GPT-4V 80% 성능

**실전 의미**:
- 자체 호스팅 가능 (비용 절감)
- Fine-tuning 가능 (도메인 특화)
- 고해상도 문서 처리

---

## Efficient Architectures

### 6. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023년 12월)

**핵심 아이디어**:
- Transformer의 대안: Self-Attention 없이 O(N) 복잡도
- Selective SSM: 중요한 정보만 기억
- 긴 시퀀스(100K+ 토큰)에서 Transformer보다 빠름

**방법론**:
```python
"""
Mamba: Selective State Space Model

Transformer: O(N^2) - 모든 토큰 쌍 비교
Mamba: O(N) - 선택적 상태 업데이트
"""

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 파라미터 프로젝션
        self.x_proj = nn.Linear(d_model, d_model)
        self.delta_proj = nn.Linear(d_model, d_model)

        # SSM 파라미터 (입력에 따라 변함 - "Selective")
        self.A = nn.Parameter(torch.randn(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)

        핵심: 각 토큰마다 다른 "중요도" 부여
        """
        batch, seq_len, d = x.shape

        # Step 1: Selective parameters 계산
        # delta: 이 토큰을 얼마나 "기억"할지
        delta = F.softplus(self.delta_proj(x))  # (batch, seq_len, d_model)

        # Step 2: State-Space 계산
        # 상태 초기화
        h = torch.zeros(batch, self.d_state, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)
            delta_t = delta[:, t, :]  # (batch, d_model)

            # Selective State Update
            # 중요한 토큰(delta 큰 값) → 상태에 큰 영향
            # 덜 중요한 토큰(delta 작은 값) → 상태에 작은 영향

            # A_bar = exp(delta * A)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * self.A)  # (batch, d_model, d_state)

            # B_bar = delta * B
            B_bar = delta_t.unsqueeze(-1) * self.B  # (batch, d_model, d_state)

            # State update (의도: 선택적 기억)
            # h_new = A_bar * h + B_bar * x
            h = A_bar * h.unsqueeze(1) + B_bar * x_t.unsqueeze(-1)
            h = h.squeeze(1)  # (batch, d_state)

            # Output 계산
            # y = C * h + D * x
            y = torch.matmul(h, self.C.T) + self.D * x_t
            outputs.append(y)

        # (batch, seq_len, d_model)
        output = torch.stack(outputs, dim=1)
        return output

"""
Selective 메커니즘 예시:

토큰 시퀀스: "The cat sat on the mat"

delta 값 (중요도):
- "cat": 0.9 (주어 - 중요!)
- "sat": 0.8 (동사 - 중요!)
- "the": 0.1 (관사 - 덜 중요)
- "on": 0.1 (전치사 - 덜 중요)

→ 중요한 토큰만 상태에 강하게 반영
→ 메모리 효율적!
"""

# 성능 비교
def compare_complexity(seq_len=10000, d_model=4096):
    # Transformer
    transformer_ops = seq_len ** 2 * d_model
    # 10000^2 * 4096 = 409조 연산!

    # Mamba
    mamba_ops = seq_len * d_model * 16  # d_state=16
    # 10000 * 4096 * 16 = 6.5억 연산

    print(f"Transformer: {transformer_ops:,} ops")
    print(f"Mamba: {mamba_ops:,} ops")
    print(f"속도 향상: {transformer_ops / mamba_ops:.1f}x")
    # 속도 향상: 625배!
```

**주요 결과**:
- 100K 토큰 처리: Transformer보다 5배 빠름
- 언어 모델링: Transformer와 동등한 Perplexity
- 메모리: 1/10 수준

**실전 의미**:
- 긴 문서 처리 (법률, 의료 기록)
- 실시간 스트리밍 (온라인 학습)
- 엣지 디바이스 배포 (메모리 제약)

---

### 7. "RWKV: Reinventing RNNs for the Transformer Era" (2024년 2월)

**핵심 아이디어**:
- RNN의 O(N) 복잡도 + Transformer의 병렬 학습
- Linear Attention 기반
- 무한 컨텍스트 (이론적으로)

**방법론**:
```python
"""
RWKV: Receptance Weighted Key Value

핵심: Attention을 선형으로 근사
→ 학습은 병렬, 추론은 순차 (RNN처럼)
"""

class RWKVAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Time-mixing parameters
        self.time_decay = nn.Parameter(torch.randn(d_model))
        self.time_first = nn.Parameter(torch.randn(d_model))

        # Projections
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.receptance = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x, state=None):
        """
        x: (batch, seq_len, d_model)
        state: 이전 상태 (RNN처럼)

        학습 시: state=None, 전체 시퀀스 병렬 처리
        추론 시: state 유지, 토큰 하나씩 처리
        """
        batch, seq_len, d = x.shape

        # Projections
        k = self.key(x)  # (batch, seq_len, d_model)
        v = self.value(x)
        r = self.receptance(x)

        # Time-mixing with exponential decay
        if state is None:
            # 학습 모드: 병렬 계산
            wkv = torch.zeros_like(v)

            for t in range(seq_len):
                # t 시점까지의 누적 weighted sum
                # 과거로 갈수록 exponential decay
                wkv[:, t, :] = sum(
                    torch.exp(-(t - i) * self.time_decay) * k[:, i, :] * v[:, i, :]
                    for i in range(t + 1)
                )
        else:
            # 추론 모드: 순차 계산 (상태 유지)
            wkv = []
            current_state = state

            for t in range(seq_len):
                # 현재 토큰의 기여
                current_wkv = k[:, t, :] * v[:, t, :]

                # 이전 상태 (decay 적용)
                current_state = torch.exp(-self.time_decay) * current_state + current_wkv

                wkv.append(current_state)

            wkv = torch.stack(wkv, dim=1)

        # Receptance (gating)
        rwkv = r * wkv

        # Output
        out = self.output(rwkv)
        return out, current_state

"""
장점:

1. 학습: 병렬 처리 가능 (Transformer처럼)
2. 추론: O(1) 메모리 (RNN처럼)
3. 무한 컨텍스트 (이론적)

예시: 추론 시

Step 1: "The" → state_1
Step 2: "cat" + state_1 → state_2
Step 3: "sat" + state_2 → state_3
...

각 단계에서 state만 업데이트 (O(1))!
"""
```

**주요 결과**:
- RWKV-7B: Pythia-7B (Transformer)와 동등 성능
- 추론 속도: Transformer보다 10배 빠름 (긴 시퀀스)
- 메모리: O(1) (컨텍스트 길이 무관)

**실전 의미**:
- 무한 대화 (챗봇, 어시스턴트)
- 스트리밍 처리 (실시간 번역)
- 리소스 제약 환경

---

## Reasoning & Planning

### 8. "Let's Verify Step by Step" (OpenAI, 2024년 6월 - 추정)

**핵심 아이디어**:
- Process Reward Model (PRM): 각 추론 단계마다 검증
- 기존 Outcome Reward Model (ORM): 최종 답만 검증
- 수학 문제 정확도 큰 폭 향상

**방법론**:
```python
"""
Process Reward Model

기존 ORM:
- 문제 → 풀이 → 답
- 답만 맞으면 reward (중간 과정 무시)

PRM:
- 각 단계마다 검증
- 오류 조기 발견
"""

class ProcessRewardModel:
    def __init__(self):
        # PRM: 각 단계의 정확도 예측
        self.prm = RewardModel()

        # Generator: 답 생성
        self.generator = ChatOpenAI(model="gpt-4-turbo")

    def solve_with_verification(self, problem: str):
        """
        단계별 검증하며 문제 풀이
        """
        prompt = f"""
        문제: {problem}

        단계별로 풀이하세요. 각 단계를 명확히 표시하세요.

        Step 1: ...
        Step 2: ...
        """

        # 초기 풀이 생성
        response = self.generator.invoke(prompt)
        solution = response.content

        # 단계 분리
        steps = self.parse_steps(solution)

        # 각 단계 검증
        verified_steps = []
        cumulative_solution = ""

        for i, step in enumerate(steps):
            cumulative_solution += f"\nStep {i+1}: {step}"

            # PRM으로 현재까지의 풀이 검증
            score = self.prm.score(problem, cumulative_solution)

            print(f"Step {i+1} 검증 점수: {score:.2f}")

            if score < 0.5:  # 낮은 점수 → 오류 가능성
                print(f"⚠️ Step {i+1}에 오류 가능성!")

                # 재시도
                retry_prompt = f"""
                문제: {problem}

                현재까지 풀이:
                {cumulative_solution}

                이 단계에 오류가 있을 수 있습니다. 검토하고 수정하세요.
                """

                retry_response = self.generator.invoke(retry_prompt)
                corrected_step = retry_response.content

                # 수정된 단계로 교체
                cumulative_solution = cumulative_solution.rsplit('\n', 1)[0]
                cumulative_solution += f"\nStep {i+1} (수정): {corrected_step}"

            verified_steps.append(step)

        return cumulative_solution

    def parse_steps(self, solution: str):
        """단계 파싱"""
        import re
        steps = re.findall(r'Step \d+: (.+)', solution)
        return steps

"""
실전 예시:

문제: "x^2 + 5x + 6 = 0을 풀어라"

Step 1: (x + 2)(x + 3) = 0으로 인수분해
→ PRM 점수: 0.95 ✓

Step 2: x = -2 또는 x = 3
→ PRM 점수: 0.45 ⚠️ (오류!)
(정답: x = -2 또는 x = -3)

Step 2 (수정): x = -2 또는 x = -3
→ PRM 점수: 0.98 ✓
"""

# PRM 학습 데이터 예시
prm_training_data = [
    {
        "problem": "2 + 2 = ?",
        "step": "Step 1: 2와 2를 더한다",
        "is_correct": True,
        "label": 1.0
    },
    {
        "problem": "2 + 2 = ?",
        "step": "Step 1: 2에서 2를 뺀다",
        "is_correct": False,
        "label": 0.0
    }
]
```

**주요 결과**:
- MATH 벤치마크: 78.2% → 84.1% (+5.9%)
- 오류 조기 발견: 82%
- GSM8K: 97.3% (거의 완벽)

**실전 의미**:
- 신뢰성 높은 추론 시스템
- 교육 (학생 풀이 과정 검토)
- 코드 생성 (각 라인 검증)

---

### 9. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023년 5월 → 2024년 확장)

**핵심 아이디어**:
- 여러 사고 경로를 탐색 (BFS/DFS)
- 각 경로 평가 후 최선 선택
- 복잡한 계획/추론 문제에 효과적

**방법론**:
```python
"""
Tree of Thoughts (ToT)

CoT: 선형 사고 (A → B → C → 답)
ToT: 트리 탐색 (여러 가능성 동시 고려)
"""

class TreeOfThoughts:
    def __init__(self, model="gpt-4-turbo"):
        self.llm = ChatOpenAI(model=model, temperature=0.7)

    def generate_thoughts(self, state, k=3):
        """
        현재 상태에서 k개의 다음 사고 생성

        예: "24를 만들기 위한 수식"
        → ["4 * 6", "8 * 3", "12 * 2"]
        """
        prompt = f"""
        현재 상태: {state}

        다음 단계로 가능한 {k}개의 사고를 생성하세요.
        각 사고는 독립적이고 창의적이어야 합니다.

        JSON 형식:
        {{
            "thoughts": ["사고1", "사고2", "사고3"]
        }}
        """

        response = self.llm.invoke(prompt)
        thoughts = json.loads(response.content)['thoughts']
        return thoughts

    def evaluate_thought(self, thought, goal):
        """
        사고의 품질 평가 (0-1 점수)

        기준:
        - 목표에 얼마나 가까운가?
        - 실현 가능한가?
        - 창의적인가?
        """
        prompt = f"""
        목표: {goal}
        사고: {thought}

        이 사고가 목표 달성에 얼마나 유용한지 평가하세요.
        점수: 0.0 (전혀 유용하지 않음) ~ 1.0 (매우 유용함)

        JSON: {{"score": 0.85, "reasoning": "..."}}
        """

        response = self.llm.invoke(prompt)
        evaluation = json.loads(response.content)
        return evaluation['score']

    def solve_bfs(self, problem, depth=3, breadth=3):
        """
        BFS 탐색: 넓게 탐색

        depth: 탐색 깊이
        breadth: 각 노드에서 확장할 사고 수
        """
        # 초기 노드
        root = {
            'state': problem,
            'path': [],
            'value': 0.0
        }

        current_level = [root]

        for level in range(depth):
            print(f"\n=== 레벨 {level + 1} ===")
            next_level = []

            for node in current_level:
                # 사고 생성
                thoughts = self.generate_thoughts(node['state'], k=breadth)

                for thought in thoughts:
                    # 평가
                    value = self.evaluate_thought(thought, problem)

                    # 새 노드
                    new_node = {
                        'state': thought,
                        'path': node['path'] + [thought],
                        'value': value
                    }

                    next_level.append(new_node)
                    print(f"  사고: {thought[:50]}... (점수: {value:.2f})")

            # Top-K 선택 (pruning)
            next_level.sort(key=lambda x: x['value'], reverse=True)
            current_level = next_level[:breadth]

            print(f"→ 상위 {breadth}개 사고 선택")

        # 최종 최선 경로
        best = max(current_level, key=lambda x: x['value'])
        return best

    def solve_dfs(self, problem, max_depth=5, threshold=0.7):
        """
        DFS 탐색: 깊게 탐색

        threshold: 이 점수 이상만 계속 탐색
        """
        def dfs_helper(state, path, depth):
            if depth >= max_depth:
                return {'state': state, 'path': path, 'value': 0}

            # 사고 생성
            thoughts = self.generate_thoughts(state, k=3)

            best_result = {'value': 0}

            for thought in thoughts:
                # 평가
                value = self.evaluate_thought(thought, problem)

                if value >= threshold:  # 유망한 사고만
                    # 재귀적 탐색
                    result = dfs_helper(
                        thought,
                        path + [thought],
                        depth + 1
                    )

                    if result['value'] > best_result['value']:
                        best_result = result

            return best_result

        return dfs_helper(problem, [], 0)

"""
실전 예시: "Game of 24"

문제: 4, 6, 8, 10을 사용하여 24 만들기

레벨 1:
- 사고 1: "4 * 6 = 24" (점수: 0.95) ✓
- 사고 2: "8 + 10 + 6 = 24" (점수: 0.90) ✓
- 사고 3: "10 - 6 = 4" (점수: 0.30)

레벨 2 (사고 1 확장):
- "확인: 4 * 6 = 24 맞음!" (점수: 1.0) ✓
...

최선 경로: ["4 * 6 = 24", "확인: 맞음"]
"""
```

**주요 결과**:
- Game of 24: 74% → 89% 성공률
- Creative Writing: 품질 점수 +40%
- Crossword Puzzles: 20% → 60%

**실전 의미**:
- 복잡한 계획 (여행 일정, 프로젝트 관리)
- 창의적 작업 (스토리 작성, 디자인)
- 게임 AI (체스, 바둑 전략)

---

## Alignment & Safety

### 10. "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2023년 → 2024년 업데이트)

**핵심 아이디어**:
- RLHF 대신 RLAIF (AI Feedback)
- "헌법" (규칙 집합) 정의 → AI가 스스로 개선
- 인간 라벨링 비용 대폭 절감

**방법론**:
```python
"""
Constitutional AI (CAI)

단계:
1. Supervised Learning with Critiques and Revisions
2. Reinforcement Learning from AI Feedback (RLAIF)
"""

class ConstitutionalAI:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4-turbo")

        # Constitution: 원칙들
        self.constitution = [
            "절대 폭력을 조장하지 마세요.",
            "개인 정보를 요구하거나 공유하지 마세요.",
            "편견이나 차별적 내용을 포함하지 마세요.",
            "불법 활동을 도와주지 마세요.",
            "정직하고 정확한 정보를 제공하세요."
        ]

    def critique(self, prompt, response):
        """
        응답을 헌법에 따라 비판

        입력: 원본 응답
        출력: 문제점 리스트
        """
        critique_prompt = f"""
        원본 질문: {prompt}
        AI 응답: {response}

        다음 원칙들을 기준으로 이 응답을 평가하세요:

        {chr(10).join(f"{i+1}. {rule}" for i, rule in enumerate(self.constitution))}

        위반 사항이 있다면 구체적으로 설명하세요.

        JSON 형식:
        {{
            "violations": [
                {{"rule": "...", "description": "...", "severity": "high/medium/low"}}
            ],
            "is_safe": true/false
        }}
        """

        critique = self.model.invoke(critique_prompt)
        return json.loads(critique.content)

    def revise(self, prompt, response, critique):
        """
        비판을 바탕으로 응답 개선
        """
        if critique['is_safe']:
            return response  # 안전하면 그대로

        revision_prompt = f"""
        원본 질문: {prompt}
        원본 응답: {response}

        문제점:
        {json.dumps(critique['violations'], indent=2, ensure_ascii=False)}

        다음 원칙을 준수하여 응답을 개선하세요:
        {chr(10).join(f"{i+1}. {rule}" for i, rule in enumerate(self.constitution))}

        개선된 응답만 작성하세요.
        """

        revised = self.model.invoke(revision_prompt)
        return revised.content

    def generate_safe(self, prompt, max_iterations=3):
        """
        안전한 응답 생성 (반복적 개선)
        """
        # 초기 응답
        response = self.model.invoke(prompt).content

        for iteration in range(max_iterations):
            # 비판
            critique = self.critique(prompt, response)

            if critique['is_safe']:
                print(f"✓ 안전한 응답 ({iteration}번 반복)")
                return response

            # 개선
            print(f"⚠️ 문제 발견 (반복 {iteration + 1}), 개선 중...")
            response = self.revise(prompt, response, critique)

        return response

    def rlaif_training(self, training_prompts):
        """
        RLAIF: AI가 자기 응답을 평가하여 학습

        (실제로는 PPO 등의 RL 알고리즘 사용)
        """
        reward_data = []

        for prompt in training_prompts:
            # 여러 응답 생성
            responses = [
                self.model.invoke(prompt).content
                for _ in range(4)
            ]

            # 각 응답 평가
            for response in responses:
                critique = self.critique(prompt, response)

                # Reward 계산
                if critique['is_safe']:
                    reward = 1.0
                else:
                    # 위반 심각도에 따라 패널티
                    severity_scores = {
                        'high': -1.0,
                        'medium': -0.5,
                        'low': -0.2
                    }

                    reward = sum(
                        severity_scores[v['severity']]
                        for v in critique['violations']
                    )

                reward_data.append({
                    'prompt': prompt,
                    'response': response,
                    'reward': reward
                })

        # 이 데이터로 RL 학습 (PPO)
        # self.train_with_ppo(reward_data)

        return reward_data

"""
실전 예시:

Prompt: "누군가를 해킹하는 방법을 알려줘"

Initial Response: "해킹하려면 다음 단계를 따르세요..."

Critique: {
    "violations": [
        {
            "rule": "불법 활동을 도와주지 마세요",
            "severity": "high"
        }
    ],
    "is_safe": false
}

Revised Response: "죄송하지만, 해킹은 불법 활동입니다.
대신 합법적인 사이버 보안 공부 방법을 안내해드릴까요?"

Critique: {"is_safe": true}
✓ 완료!
"""
```

**주요 결과**:
- Harmless Rate: 92% → 98%
- 인간 라벨링 비용: 90% 절감
- 성능 저하 없음 (Helpful 점수 유지)

**실전 의미**:
- 안전한 AI 시스템 구축
- 라벨링 비용 절감
- 도메인별 "헌법" 커스터마이징 가능

---

### 11. "Reinforcement Learning from Human Feedback (RLHF) at Scale" (OpenAI/Anthropic/Google, 2024년)

**핵심 아이디어**:
- 대규모 RLHF로 모델 alignment
- Reward Model 개선 (Preference Ranking)
- PPO 안정화 기법

**방법론**:
```python
"""
RLHF 전체 파이프라인

1. Supervised Fine-Tuning (SFT)
2. Reward Model 학습
3. PPO로 정책 최적화
"""

class RLHFPipeline:
    def __init__(self):
        self.sft_model = None
        self.reward_model = None
        self.policy_model = None

    # ========== Step 1: SFT ==========
    def supervised_fine_tuning(self, demonstrations):
        """
        인간이 작성한 고품질 응답으로 파인튜닝

        demonstrations: [
            {"prompt": "...", "response": "..."},
            ...
        ]
        """
        from transformers import Trainer, TrainingArguments

        # 베이스 모델 로드
        model = AutoModelForCausalLM.from_pretrained("gpt-2")
        tokenizer = AutoTokenizer.from_pretrained("gpt-2")

        # 데이터 준비
        def tokenize(example):
            text = example['prompt'] + example['response']
            return tokenizer(text, truncation=True, max_length=512)

        dataset = Dataset.from_list(demonstrations)
        tokenized = dataset.map(tokenize)

        # 학습
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="./sft",
                num_train_epochs=3,
                per_device_train_batch_size=4,
                learning_rate=5e-5
            ),
            train_dataset=tokenized
        )

        trainer.train()
        self.sft_model = model

    # ========== Step 2: Reward Model ==========
    def train_reward_model(self, comparisons):
        """
        인간 선호도 데이터로 Reward Model 학습

        comparisons: [
            {
                "prompt": "...",
                "chosen": "좋은 응답",
                "rejected": "나쁜 응답"
            },
            ...
        ]
        """
        class RewardModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model
                self.value_head = nn.Linear(base_model.config.hidden_size, 1)

            def forward(self, input_ids):
                # 마지막 토큰의 hidden state
                outputs = self.base(input_ids, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1][:, -1, :]

                # Scalar reward
                reward = self.value_head(last_hidden)
                return reward

        rm = RewardModel(self.sft_model)

        # Loss: Bradley-Terry model
        def compute_loss(chosen_rewards, rejected_rewards):
            # P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
            return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        optimizer = torch.optim.Adam(rm.parameters(), lr=1e-5)

        for epoch in range(3):
            for batch in comparisons:
                # 인코딩
                chosen_tokens = tokenizer(batch['prompt'] + batch['chosen'], return_tensors='pt')
                rejected_tokens = tokenizer(batch['prompt'] + batch['rejected'], return_tensors='pt')

                # Reward 계산
                r_chosen = rm(chosen_tokens['input_ids'])
                r_rejected = rm(rejected_tokens['input_ids'])

                # Loss
                loss = compute_loss(r_chosen, r_rejected)

                # 업데이트
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.reward_model = rm

    # ========== Step 3: PPO ==========
    def ppo_training(self, prompts, epochs=10):
        """
        PPO로 정책 최적화

        목표: Reward Model 점수 최대화
        """
        from trl import PPOTrainer, PPOConfig

        # PPO 설정
        config = PPOConfig(
            model_name="gpt-2",
            learning_rate=1e-5,
            batch_size=16,
            mini_batch_size=4,
            ppo_epochs=4,
            target_kl=0.01  # KL divergence 제약 (SFT 모델과 너무 멀어지지 않도록)
        )

        ppo_trainer = PPOTrainer(
            config=config,
            model=self.sft_model,
            ref_model=copy.deepcopy(self.sft_model),  # Reference (SFT)
            tokenizer=tokenizer,
            dataset=prompts
        )

        for epoch in range(epochs):
            for batch in ppo_trainer.dataloader:
                # 응답 생성
                query_tensors = batch['input_ids']
                response_tensors = ppo_trainer.generate(query_tensors)

                # Reward 계산
                texts = [tokenizer.decode(r) for r in response_tensors]
                rewards = [self.reward_model(r).item() for r in response_tensors]

                # PPO 업데이트
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

                print(f"Epoch {epoch}, Mean Reward: {np.mean(rewards):.2f}")

        self.policy_model = ppo_trainer.model

"""
전체 흐름:

1. SFT: 인간 데모로 기본 학습
   "좋은 응답이 어떤 건지 보여줌"

2. Reward Model: 선호도 학습
   "응답 A가 B보다 나은지 점수화"

3. PPO: Reward 최대화
   "Reward가 높은 응답을 더 자주 생성하도록 학습"

결과: 인간 선호와 align된 모델!
"""

# 실전 사용
rlhf = RLHFPipeline()

# Step 1: SFT
rlhf.supervised_fine_tuning([
    {"prompt": "파이썬이 뭐야?", "response": "파이썬은 프로그래밍 언어입니다..."},
    # ... 10K 데모
])

# Step 2: Reward Model
rlhf.train_reward_model([
    {
        "prompt": "저를 도와주세요",
        "chosen": "무엇을 도와드릴까요?",
        "rejected": "싫어요"
    },
    # ... 100K 비교
])

# Step 3: PPO
rlhf.ppo_training(["안녕하세요", "파이썬 설명해줘", ...])
```

**주요 결과**:
- Win Rate vs SFT: 72%
- Helpfulness: +45%
- Harmlessness: +30%

**실전 의미**:
- ChatGPT, Claude 등의 핵심 기술
- 도메인 특화 Alignment (의료, 법률)
- 지속적 개선 (새로운 피드백 반영)

---

## Applications

### 12. "Toolformer: Language Models Can Teach Themselves to Use Tools" (Meta AI, 2023년)

**핵심 아이디어**:
- LLM이 스스로 도구 사용법 학습
- API 호출 시점과 파라미터를 자동으로 결정
- 인간 라벨링 없이 도구 통합

**방법론**:
```python
"""
Toolformer

핵심: LLM이 자기 생성 데이터로 도구 사용 학습
"""

class Toolformer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo")

        # 사용 가능한 도구들
        self.tools = {
            'Calculator': self.calculator,
            'QA': self.qa_system,
            'Search': self.search,
            'Calendar': self.calendar
        }

    # ========== Step 1: 도구 호출 후보 생성 ==========
    def generate_tool_calls(self, text):
        """
        텍스트에 도구 호출 삽입 (후보)

        "The population of Tokyo is"
        → "The population of Tokyo is [QA(What is the population of Tokyo?)]"
        """
        prompt = f"""
        다음 텍스트에서 도구를 호출하면 유용한 위치를 찾고,
        적절한 도구 호출을 삽입하세요.

        사용 가능한 도구:
        - Calculator(expression): 계산
        - QA(question): 질문 답변
        - Search(query): 웹 검색
        - Calendar(date_query): 날짜 계산

        텍스트: {text}

        도구 호출을 [Tool(args)] 형식으로 삽입하세요.
        """

        response = self.llm.invoke(prompt)
        return response.content

    # ========== Step 2: 자기 평가 (Filtering) ==========
    def filter_useful_calls(self, original, with_tools):
        """
        도구 호출이 실제로 유용한지 평가

        방법: 도구 호출 전후의 perplexity 비교
        - PPL 감소 → 유용!
        - PPL 증가 → 불필요
        """
        # 원본 텍스트 perplexity
        ppl_original = self.compute_perplexity(original)

        # 도구 호출 실행 후 perplexity
        executed = self.execute_tools(with_tools)
        ppl_with_tools = self.compute_perplexity(executed)

        # 개선되었는지 확인
        is_useful = ppl_with_tools < ppl_original

        return is_useful, executed

    def execute_tools(self, text):
        """
        텍스트 내 도구 호출 실행

        "[Calculator(2+2)]" → "4"
        """
        import re

        # 도구 호출 패턴 찾기
        pattern = r'\[(\w+)\((.*?)\)\]'
        matches = re.findall(pattern, text)

        for tool_name, args in matches:
            if tool_name in self.tools:
                # 도구 실행
                result = self.tools[tool_name](args)

                # 결과로 대체
                text = text.replace(
                    f'[{tool_name}({args})]',
                    str(result)
                )

        return text

    # ========== Step 3: 파인튜닝 ==========
    def fine_tune(self, corpus):
        """
        필터링된 도구 호출 데이터로 파인튜닝

        학습 데이터:
        "The population of Tokyo is [QA(population of Tokyo)] 13.9M"
        """
        training_data = []

        for text in corpus:
            # 도구 호출 후보 생성
            with_tools = self.generate_tool_calls(text)

            # 유용성 평가
            is_useful, executed = self.filter_useful_calls(text, with_tools)

            if is_useful:
                training_data.append({
                    'input': text,
                    'output': executed
                })

        # 파인튜닝 (실제로는 Transformer 학습)
        # fine_tune_model(training_data)

        return training_data

    # ========== 도구 구현 ==========
    def calculator(self, expression):
        try:
            return eval(expression)
        except:
            return "Error"

    def qa_system(self, question):
        # Wikipedia API 등 사용
        return "13.9 million"  # 예시

    def search(self, query):
        # 웹 검색 API
        return "Search results..."

    def calendar(self, date_query):
        from datetime import datetime, timedelta
        # 날짜 계산
        return datetime.now() + timedelta(days=7)

"""
실전 예시:

원본: "3년 후 2027년입니다"
→ PPL: 높음 (2027년이 맞는지 불확실)

도구 삽입: "3년 후 [Calendar(3 years from now)]입니다"
→ 실행: "3년 후 2027입니다"
→ PPL: 낮음 (정확함!)

결과: 도구 호출 유지!
"""
```

**주요 결과**:
- LAMA (fact recall): 60% → 84%
- Math problems: 34% → 72%
- 추가 라벨링 없이 도구 통합

**실전 의미**:
- 자동 도구 통합 (API, 데이터베이스)
- 지속적 도구 추가 (새 API 출시 시)
- Domain-specific tools (의료, 금융)

---

### 13. "ReAct: Synergizing Reasoning and Acting in Language Models" (Google/Princeton, 2022년 → 2024년 확장)

**핵심 아이디어**:
- Reasoning (추론) + Acting (행동) 결합
- Thought → Action → Observation 루프
- Chain-of-Thought보다 실제 작업에 효과적

**방법론**:
```python
"""
ReAct: Reason + Act

CoT: Thought만
ReAct: Thought + Action + Observation
"""

class ReActAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

        # 사용 가능한 액션들
        self.actions = {
            'Search': self.search,
            'Lookup': self.lookup,
            'Finish': self.finish
        }

        self.observations = []

    def run(self, question, max_steps=10):
        """
        ReAct 루프

        Thought: 다음에 무엇을 할지
        Action: 도구 선택
        Observation: 결과 관찰
        """
        context = f"Question: {question}\n"

        for step in range(max_steps):
            # ===== Thought =====
            thought_prompt = f"""
            {context}

            다음 단계를 생각하세요. 사용 가능한 액션:
            - Search[query]: 위키피디아 검색
            - Lookup[keyword]: 현재 문서에서 키워드 찾기
            - Finish[answer]: 최종 답변

            Thought: (다음에 무엇을 할지 설명)
            Action: ActionName[argument]
            """

            response = self.llm.invoke(thought_prompt).content

            # Thought 추출
            thought = self.extract_thought(response)
            print(f"\nThought {step + 1}: {thought}")

            # Action 추출
            action_match = re.search(r'Action:\s*(\w+)\[(.*?)\]', response)
            if not action_match:
                print("액션을 찾을 수 없습니다.")
                break

            action_name = action_match.group(1)
            action_arg = action_match.group(2)

            print(f"Action {step + 1}: {action_name}[{action_arg}]")

            # ===== Action 실행 =====
            if action_name not in self.actions:
                observation = f"Error: Unknown action {action_name}"
            else:
                observation = self.actions[action_name](action_arg)

            print(f"Observation {step + 1}: {observation[:100]}...")

            # ===== Finish 확인 =====
            if action_name == 'Finish':
                return observation

            # 컨텍스트 업데이트
            context += f"\nThought {step + 1}: {thought}\n"
            context += f"Action {step + 1}: {action_name}[{action_arg}]\n"
            context += f"Observation {step + 1}: {observation}\n"

        return "Max steps reached"

    def extract_thought(self, response):
        match = re.search(r'Thought:\s*(.+)', response)
        return match.group(1) if match else ""

    def search(self, query):
        """Wikipedia 검색"""
        import wikipedia
        try:
            page = wikipedia.page(query)
            self.current_page = page.content
            return page.summary[:500]
        except:
            return "No results found"

    def lookup(self, keyword):
        """현재 페이지에서 키워드 찾기"""
        if not hasattr(self, 'current_page'):
            return "No page loaded"

        # 키워드 주변 문맥 반환
        idx = self.current_page.find(keyword)
        if idx == -1:
            return f"Keyword '{keyword}' not found"

        start = max(0, idx - 100)
        end = min(len(self.current_page), idx + 100)
        return self.current_page[start:end]

    def finish(self, answer):
        """최종 답변"""
        return answer

"""
실전 예시:

Question: "피에르 페로의 출생지는 어디인가?"

Thought 1: 먼저 피에르 페로가 누구인지 검색해야겠다
Action 1: Search[Pierre Perrot]
Observation 1: Pierre Perrot는 프랑스 언어학자... 파리에서 태어남...

Thought 2: 파리라는 정보를 찾았다. 확인해보자
Action 2: Lookup[출생]
Observation 2: ...1932년 파리에서 태어남...

Thought 3: 답을 찾았다
Action 3: Finish[파리]
Observation 3: 파리

답: 파리
"""
```

**주요 결과**:
- HotpotQA: 29% → 45%
- Fever (fact verification): 62% → 75%
- WebShop (온라인 쇼핑): 36% → 51%

**실전 의미**:
- 실제 도구 사용하는 Agent
- 복잡한 다단계 작업
- 디버깅 용이 (추론 과정 추적)

---

## 벡터 유사도 & 임베딩

### 14. "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (OpenAI, 2024년)

**핵심 아이디어**:
- text-embedding-3 모델: 차원 축소 가능 (Matryoshka)
- Contrastive Learning으로 의미적 유사도 학습
- MTEB 벤치마크에서 SOTA

**방법론**:
```python
"""
Contrastive Learning for Embeddings

핵심: 유사한 텍스트는 가깝게, 다른 텍스트는 멀게
"""

class ContrastiveEmbedding:
    def __init__(self, model_name="text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model_name

    def embed(self, text, dimensions=None):
        """
        텍스트 임베딩

        dimensions: 차원 축소 (1536 → 256)
        → 성능 약간 저하, 속도/비용 대폭 절감
        """
        params = {
            "input": text,
            "model": self.model
        }

        if dimensions:
            params["dimensions"] = dimensions

        response = self.client.embeddings.create(**params)
        return response.data[0].embedding

    def cosine_similarity(self, emb1, emb2):
        """코사인 유사도"""
        import numpy as np
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

"""
Matryoshka Embeddings:

Full: 1536 dimensions
      ↓ 축소
Large: 512 dimensions (99% 성능, 3배 빠름)
      ↓
Small: 256 dimensions (95% 성능, 6배 빠름)

사용 사례별 선택:
- 정밀 검색: 1536
- 일반 검색: 512
- 실시간 추천: 256
"""

# 실전 사용
embedder = ContrastiveEmbedding()

# 예시: 유사한 문장 찾기
sentences = [
    "강아지가 공을 쫓고 있다",
    "개가 뛰어노는 중이다",
    "고양이가 자고 있다",
    "파이썬 코드를 작성한다"
]

embeddings = [embedder.embed(s, dimensions=256) for s in sentences]

# 첫 문장과의 유사도
base = embeddings[0]
for i, emb in enumerate(embeddings[1:], 1):
    sim = embedder.cosine_similarity(base, emb)
    print(f"{sentences[0]} vs {sentences[i]}: {sim:.3f}")

# 출력:
# 강아지가 공을 쫓고 있다 vs 개가 뛰어노는 중이다: 0.892 (높음!)
# 강아지가 공을 쫓고 있다 vs 고양이가 자고 있다: 0.623
# 강아지가 공을 쫓고 있다 vs 파이썬 코드를 작성한다: 0.234 (낮음!)
```

**주요 결과**:
- MTEB (Massive Text Embedding Benchmark): 64.6% (SOTA)
- 256 차원: 1536 차원 대비 95% 성능, 6배 빠름
- 다국어 지원 (100+ 언어)

**실전 의미**:
- RAG 시스템 (문서 검색)
- 추천 시스템
- 중복 탐지

---

## 핵심 요약

### LLMs
- **Llama 3**: GQA로 추론 효율 2배, 오픈소스 SOTA
- **Mixtral 8x7B**: MoE로 46.7B인데 12.9B처럼 빠름
- **Gemini 1.5**: 1M 토큰 컨텍스트, Ring Attention

### Multimodal
- **GPT-4V**: 범용 vision-language, 56.8% MMMU
- **LLaVA-NeXT**: 오픈소스, Dynamic High Resolution

### Efficient
- **Mamba**: O(N) 복잡도, Selective SSM
- **RWKV**: RNN + Transformer 장점 결합

### Reasoning
- **Process Reward Model**: 단계별 검증으로 +5.9%
- **Tree of Thoughts**: 트리 탐색으로 +40% 품질

### Alignment
- **Constitutional AI**: RLAIF로 90% 비용 절감
- **RLHF at Scale**: PPO로 alignment

### Applications
- **Toolformer**: 자동 도구 통합
- **ReAct**: Reasoning + Acting

---

**작성일**: 2024-11-18
**업데이트**: Phase 6 - Current Trends
