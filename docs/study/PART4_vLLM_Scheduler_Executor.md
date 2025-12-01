# PART 4: vLLM Scheduler & Executor

> 면접에서 vLLM을 아는지 테스트할 때 핵심이 되는 부분

---

## 목차
1. [vLLM Scheduler 구조](#1-vllm-scheduler-구조)
2. [Continuous Batching](#2-continuous-batching)
3. [Scheduler의 Prefill/Decode 구분](#3-scheduler의-prefilldecode-구분)
4. [토큰 생성 속도 영향 요인](#4-토큰-생성-속도-영향-요인)
5. [ExecuteModel 내부 로직](#5-executemodel-내부-로직)
6. [CUDA Graph Padding Dispatch](#6-cuda-graph-padding-dispatch)
7. [그래프 캐싱과 Latency 감소](#7-그래프-캐싱과-latency-감소)
8. [Multi-Request와 Multi-Tenancy](#8-multi-request와-multi-tenancy)
9. [면접 예상 질문 및 답변](#9-면접-예상-질문-및-답변)

---

## 1. vLLM Scheduler 구조

### 1.1 Scheduler의 역할

```
┌─────────────────────────────────────────────────────────────────┐
│                      vLLM Scheduler                             │
│                                                                 │
│  입력: 새로운 요청들, 실행 중인 요청들의 상태                   │
│  출력: 이번 스텝에서 처리할 요청들과 토큰 수                    │
│                                                                 │
│  핵심 결정:                                                     │
│  1. 어떤 요청을 실행할 것인가?                                  │
│  2. 각 요청에 몇 개의 토큰을 할당할 것인가?                     │
│  3. 메모리 부족 시 어떤 요청을 중단할 것인가?                   │
│  4. KV 블록을 어떻게 할당/해제할 것인가?                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 큐 구조

```python
# vllm/v1/core/sched/scheduler.py:46-131 기반

class Scheduler:
    def __init__(self, vllm_config, kv_cache_config, ...):
        # 제약 조건
        self.max_num_running_reqs = scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = scheduler_config.max_num_batched_tokens
        self.max_model_len = scheduler_config.max_model_len

        # 요청 저장소
        self.requests: dict[str, Request] = {}  # req_id → Request

        # 스케줄링 큐
        self.waiting = create_request_queue(policy)  # 대기 큐
        self.running: list[Request] = []             # 실행 큐

        # 완료된 요청 ID
        self.finished_req_ids: set[str] = set()

        # KV 캐시 매니저
        self.kv_cache_manager = KVCacheManager(...)
```

### 1.3 요청 상태 전이

```
┌─────────┐    add_request()    ┌─────────┐    schedule()    ┌─────────┐
│  NEW    │ ─────────────────→  │ WAITING │ ───────────────→ │ RUNNING │
└─────────┘                     └─────────┘                  └─────────┘
                                     ↑                            │
                                     │      preemption            │
                                     └────────────────────────────┤
                                                                  │
                                                                  ↓
                                                            ┌─────────┐
                                                            │FINISHED │
                                                            └─────────┘

상태 정의:
- NEW: 방금 도착한 요청
- WAITING: 대기 큐에서 실행 대기
- RUNNING: 현재 토큰 생성 중
- PREEMPTED: 메모리 부족으로 중단됨
- FINISHED: 완료 (EOS 또는 max_len 도달)
```

### 1.4 Scheduling 알고리즘

```python
# vllm/v1/core/sched/scheduler.py:183-653 기반

def schedule(self) -> SchedulerOutput:
    """매 스텝마다 실행할 요청들과 토큰 수 결정"""

    # 1. RUNNING 요청들 먼저 스케줄
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]

        # 이 요청에 할당할 토큰 수 계산
        num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)

        # KV 블록 할당 시도
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens
        )

        if new_blocks is None:
            # 메모리 부족! 낮은 우선순위 요청 preempt
            preempted_req = self.running.pop()
            self.kv_cache_manager.free(preempted_req)
            preempted_req.status = RequestStatus.PREEMPTED
            self.waiting.prepend_request(preempted_req)
        else:
            # 스케줄 성공
            scheduled_running_reqs.append(request)
            token_budget -= num_new_tokens
            req_index += 1

    # 2. WAITING 요청들 스케줄 (메모리 여유 있을 때)
    while self.waiting and token_budget > 0:
        if len(self.running) == self.max_num_running_reqs:
            break

        request = self.waiting.pop_request()

        # Prefix caching으로 이미 계산된 토큰 확인
        num_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)

        num_new_tokens = request.num_tokens - num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)

        new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)

        if new_blocks is None:
            break  # 메모리 부족

        # 실행 큐에 추가
        self.running.append(request)
        scheduled_new_reqs.append(request)
        token_budget -= num_new_tokens

    return SchedulerOutput(...)
```

### 1.5 스케줄링 정책

```python
# vllm/v1/core/sched/request_queue.py 기반

class SchedulingPolicy(Enum):
    FCFS = "fcfs"      # First-Come-First-Served
    PRIORITY = "priority"  # 우선순위 기반

def create_request_queue(policy: SchedulingPolicy):
    if policy == SchedulingPolicy.FCFS:
        return FIFORequestQueue()
    elif policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
```

**FCFS (기본):**
- 먼저 도착한 요청 먼저 처리
- 공정성 보장
- 단순하고 예측 가능

**Priority:**
- 우선순위 높은 요청 먼저 처리
- 중요 요청의 latency 최소화
- Preemption 시 낮은 우선순위부터

---

## 2. Continuous Batching

### 2.1 Static Batching의 문제

```
Static Batching:
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Batch = [Req1, Req2, Req3, Req4]                       │
│         모든 요청이 완료될 때까지 대기                          │
│                                                                 │
│ Timeline:                                                       │
│ Req1: ████████████████████ (20 tokens)                         │
│ Req2: ████████░░░░░░░░░░░░ (8 tokens, 12 idle)                │
│ Req3: ██████████████░░░░░░ (14 tokens, 6 idle)                │
│ Req4: ████████████████████████████ (28 tokens) ← 결정자        │
│                                                                 │
│ 문제: Req2 완료 후에도 새 요청 추가 불가                        │
│ 낭비: 12 + 6 = 18 토큰 슬롯 낭비                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Continuous Batching의 해결

```
Continuous Batching:
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Batch = [Req1, Req2, Req3, Req4]                       │
│ Step 8: Req2 완료 → Batch = [Req1, Req3, Req4, Req5(새로운)]   │
│ Step 14: Req3 완료 → Batch = [Req1, Req4, Req5, Req6(새로운)]  │
│ ...                                                             │
│                                                                 │
│ Timeline:                                                       │
│ Req1: ████████████████████                                     │
│ Req2: ████████ → Req5: ████████████████                        │
│ Req3: ██████████████ → Req6: ████████████                      │
│ Req4: ████████████████████████████                             │
│                                                                 │
│ 효과: GPU 항상 최대 활용!                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Iteration-Level Batching

```python
# 매 iteration마다 배치 재구성

def continuous_batching_loop():
    while True:
        # 1. 완료된 요청 제거
        finished = [r for r in running if r.is_finished()]
        for req in finished:
            running.remove(req)
            send_response(req)

        # 2. 새 요청 추가 (여유 있으면)
        while len(running) < max_batch_size and waiting:
            new_req = waiting.pop()
            running.append(new_req)

        # 3. 현재 배치로 1 step 실행
        outputs = model.generate_step(running)

        # 4. 결과 업데이트
        for req, output in zip(running, outputs):
            req.append_token(output)
```

### 2.4 Throughput 극대화 원리

```
Static Batching 분석:
- 배치 완료 시간: max(각 요청의 길이)
- 평균 GPU 활용률: mean(길이) / max(길이) ≈ 60-70%

Continuous Batching 분석:
- 빈 슬롯 즉시 채움
- GPU 활용률: ~95%+

수치 예시:
배치 크기: 32
평균 생성 길이: 100 tokens
길이 분산: ±50 tokens

Static: 150 tokens 동안 32 슬롯 유지 → 100/150 = 67% 효율
Continuous: 항상 32 슬롯 채움 → 95%+ 효율

→ Throughput 1.4배 향상!
```

### 2.5 vLLM의 Continuous Batching 구현

```python
# vllm/v1/core/sched/scheduler.py 의 schedule() 메서드

def schedule(self) -> SchedulerOutput:
    """
    Continuous batching의 핵심:
    매 스텝마다 배치를 동적으로 재구성
    """

    # RUNNING 요청들의 decode 토큰 할당
    for request in self.running:
        # 1 토큰씩 decode
        num_scheduled_tokens[request.id] = 1

    # 여유 토큰 예산으로 새 요청 추가
    remaining_budget = self.max_num_scheduled_tokens - len(self.running)

    while remaining_budget > 0 and self.waiting:
        new_request = self.waiting.pop()
        # Prefill 토큰 할당 (청크 가능)
        prefill_tokens = min(new_request.prompt_len, remaining_budget)
        num_scheduled_tokens[new_request.id] = prefill_tokens
        remaining_budget -= prefill_tokens

    return SchedulerOutput(num_scheduled_tokens=num_scheduled_tokens)
```

---

## 3. Scheduler의 Prefill/Decode 구분

### 3.1 Prefill과 Decode의 특성 차이

```
┌─────────────────────────────────────────────────────────────────┐
│              Prefill                 │         Decode           │
├─────────────────────────────────────────────────────────────────┤
│  다수 토큰 처리 (프롬프트 전체)      │  1 토큰 처리            │
│  Compute-bound                       │  Memory-bound           │
│  높은 GPU 활용률                     │  낮은 GPU 활용률        │
│  QKV 모두 새로 계산                  │  Q만 새로, KV는 캐시    │
│  Attention: O(T²)                   │  Attention: O(T)        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Scheduler의 구분 로직

```python
# vllm/v1/core/sched/scheduler.py:183-194 기반

def schedule(self) -> SchedulerOutput:
    # NOTE: 스케줄러에는 "디코딩" 또는 "프리필" 단계가 없습니다.
    # 각 요청은 num_computed_tokens와 num_tokens_with_spec만 있습니다.
    #
    # 매 스텝에서 스케줄러는 각 요청의 num_computed_tokens가
    # num_tokens_with_spec를 따라잡도록 토큰을 할당합니다.
    #
    # 이는 청크 프리필, 프리픽스 캐싱, 추론적 디코딩,
    # 그리고 미래의 "점프 디코딩" 최적화를 일반화합니다.

    for request in self.running:
        # Prefill인지 Decode인지는 암묵적으로 결정됨
        num_new_tokens = (
            request.num_tokens_with_spec - request.num_computed_tokens
        )

        # num_new_tokens > 1이면 Prefill 특성
        # num_new_tokens == 1이면 Decode 특성
```

### 3.3 Chunked Prefill

```
문제: 긴 프롬프트의 Prefill이 다른 요청을 블로킹

해결: Prefill을 청크로 나누어 처리
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  기존: Req1 Prefill (4096 tokens) → 다른 요청 대기             │
│  ─────────────────────────────────────────────────────→        │
│                                                                 │
│  Chunked: Req1 chunk1 + Req2 decode + Req3 decode              │
│           Req1 chunk2 + Req2 decode + Req3 decode              │
│           Req1 chunk3 + Req2 decode + Req3 decode              │
│           ...                                                   │
│                                                                 │
│  효과: 긴 프리필 중에도 디코드 요청 처리 가능                   │
└─────────────────────────────────────────────────────────────────┘
```

```python
# 청크 프리필 로직
def schedule_chunked_prefill(request, token_budget, chunk_size=512):
    remaining_prefill = request.prompt_len - request.num_computed_tokens

    if remaining_prefill > 0:
        # Prefill 청크 할당
        chunk_tokens = min(remaining_prefill, chunk_size, token_budget)
        return chunk_tokens

    else:
        # Decode 단계
        return 1
```

### 3.4 Prefill-Decode 분리 전략

```
전략 1: Mixed Batching (vLLM 기본)
┌─────────────────────────────────────────────────────────────────┐
│ Batch: [Prefill_req1(512), Decode_req2(1), Decode_req3(1), ...] │
│                                                                 │
│ 장점: 단순, GPU 활용 최대화                                     │
│ 단점: Prefill이 Decode latency에 영향                          │
└─────────────────────────────────────────────────────────────────┘

전략 2: Separate Batching (SplitFuse)
┌─────────────────────────────────────────────────────────────────┐
│ Prefill Batch: [Prefill_req1, Prefill_req2, ...]               │
│ Decode Batch: [Decode_req3, Decode_req4, ...]                  │
│                                                                 │
│ 장점: 각 특성에 맞는 최적화 가능                                │
│ 단점: 스케줄링 복잡, 잠재적 GPU 낭비                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 토큰 생성 속도 영향 요인

### 4.1 주요 영향 요인

```
┌─────────────────────────────────────────────────────────────────┐
│                 토큰 생성 속도 결정 요인                        │
├─────────────────────────────────────────────────────────────────┤
│  1. 배치 크기 (Batch Size)                                     │
│     - 크면: Throughput ↑, Latency ↑                            │
│     - 작으면: Throughput ↓, Latency ↓                          │
│                                                                 │
│  2. 시퀀스 길이 (Sequence Length)                              │
│     - 길면: KV 캐시 읽기 ↑, Attention 비용 ↑                   │
│     - 짧으면: 효율적이지만 context 제한                         │
│                                                                 │
│  3. 모델 크기                                                   │
│     - 크면: 파라미터 많음, 연산/메모리 ↑                        │
│     - 작으면: 빠르지만 품질 ↓                                   │
│                                                                 │
│  4. GPU 메모리 대역폭                                           │
│     - Decode는 Memory-bound                                     │
│     - 대역폭이 병목                                             │
│                                                                 │
│  5. KV 캐시 효율                                                │
│     - GQA: KV heads 감소 → 메모리 접근 ↓                        │
│     - Prefix caching: 중복 계산 제거                            │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Batch Size와 Throughput/Latency 관계

```
                    Throughput (tokens/sec)
                         ▲
                         │              ┌─────────────
                         │           ──┘
                         │        ──┘
                         │     ──┘
                         │  ──┘
                         │─┘
                         └────────────────────────────→ Batch Size

                    Latency (ms/token)
                         ▲
                         │                     ────────
                         │                ────┘
                         │            ───┘
                         │        ───┘
                         │    ──┘
                         │───┘
                         └────────────────────────────→ Batch Size

최적점: Throughput이 포화되기 시작하는 지점
      보통 메모리 대역폭이 포화되는 지점
```

### 4.3 수치 분석

```python
# 토큰 생성 속도 분석

def analyze_token_generation_speed():
    # A100 80GB, LLaMA-7B

    # Decode 단계 분석
    model_params = 7e9  # 7B
    param_bytes = model_params * 2  # FP16 = 14GB

    # 메모리 대역폭 제한 계산
    memory_bandwidth = 2e12  # 2TB/s (A100)

    # 모델 가중치 읽기 시간 (배치 공유)
    weight_read_time = param_bytes / memory_bandwidth  # 7ms

    # KV 캐시 읽기 (배치 × 시퀀스 × KV크기)
    batch_size = 32
    seq_len = 2048
    kv_per_token = 0.5e6  # 0.5MB (LLaMA-7B)

    kv_read = batch_size * seq_len * kv_per_token  # 32GB
    kv_time = kv_read / memory_bandwidth  # 16ms

    # 총 시간
    total_time = weight_read_time + kv_time  # 23ms
    tokens_per_sec = batch_size * 1000 / total_time  # ~1400 tokens/sec

    return tokens_per_sec  # 배치 32에서 ~1400 tokens/sec
```

### 4.4 최적화 전략

```
┌─────────────────────────────────────────────────────────────────┐
│  최적화 전략                       │  효과                      │
├─────────────────────────────────────────────────────────────────┤
│  1. Continuous Batching           │  GPU 활용률 1.5-2x ↑      │
│  2. PagedAttention                │  배치 크기 2-4x ↑         │
│  3. GQA/MQA                       │  KV 메모리 4-8x ↓         │
│  4. FlashAttention                │  Prefill 속도 2-3x ↑      │
│  5. CUDA Graph                    │  커널 오버헤드 제거        │
│  6. Tensor Parallelism            │  모델 크기 분산            │
│  7. Speculative Decoding          │  유효 토큰/스텝 ↑         │
│  8. Quantization (INT8/FP8)       │  메모리/연산 2x ↓         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. ExecuteModel 내부 로직

### 5.1 ModelRunner의 역할

```
┌─────────────────────────────────────────────────────────────────┐
│                      ExecuteModel Flow                          │
│                                                                 │
│  SchedulerOutput                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                               │
│  │ Prepare     │  입력 텐서 준비, 패딩, 배치 구성              │
│  │ Inputs      │                                               │
│  └──────┬──────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ Model       │  실제 신경망 forward pass                     │
│  │ Forward     │  (CUDA Graph 또는 eager mode)                 │
│  └──────┬──────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ Sample      │  logits → token_id 샘플링                     │
│  │ Tokens      │                                               │
│  └──────┬──────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ModelRunnerOutput                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 입력 준비 단계

```python
# 개념적 코드

def prepare_inputs(scheduler_output: SchedulerOutput):
    """
    스케줄러 출력을 GPU 텐서로 변환
    """
    # 토큰 ID 수집
    input_ids = []
    positions = []
    slot_mapping = []

    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        request = requests[req_id]

        # 이번 스텝에서 처리할 토큰들
        start = request.num_computed_tokens
        end = start + num_tokens
        tokens = request.all_token_ids[start:end]

        input_ids.extend(tokens)
        positions.extend(range(start, end))

        # KV 캐시 슬롯 매핑
        for pos in range(start, end):
            block_idx = pos // block_size
            block_offset = pos % block_size
            physical_block = request.block_table[block_idx]
            slot = physical_block * block_size + block_offset
            slot_mapping.append(slot)

    return {
        "input_ids": torch.tensor(input_ids, device="cuda"),
        "positions": torch.tensor(positions, device="cuda"),
        "slot_mapping": torch.tensor(slot_mapping, device="cuda"),
        "block_tables": build_block_tables(scheduler_output),
    }
```

### 5.3 Model Forward

```python
def execute_model(prepared_inputs):
    """
    모델 forward pass 실행
    """
    input_ids = prepared_inputs["input_ids"]
    positions = prepared_inputs["positions"]
    kv_caches = prepared_inputs["kv_caches"]

    # 임베딩
    hidden_states = self.embed_tokens(input_ids)

    # Transformer 레이어들
    for layer_idx, layer in enumerate(self.layers):
        hidden_states = layer(
            hidden_states,
            positions,
            kv_caches[layer_idx],
            attn_metadata=prepared_inputs["attn_metadata"]
        )

    # LM Head
    hidden_states = self.norm(hidden_states)
    logits = self.lm_head(hidden_states)

    return logits
```

### 5.4 샘플링 단계

```python
def sample_tokens(logits, sampling_params):
    """
    Logits에서 다음 토큰 샘플링
    """
    # Temperature 적용
    if sampling_params.temperature > 0:
        logits = logits / sampling_params.temperature

    # Top-p (nucleus) 샘플링
    if sampling_params.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > sampling_params.top_p
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    # Top-k 샘플링
    if sampling_params.top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, sampling_params.top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(1, top_k_indices, top_k_logits)

    # 확률 분포에서 샘플링
    probs = F.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)

    return next_tokens
```

---

## 6. CUDA Graph Padding Dispatch

### 6.1 CUDA Graph란?

```
일반 실행:
┌─────────────────────────────────────────────────────────────────┐
│  CPU: [커널1 런치] [커널2 런치] [커널3 런치] ...                │
│        ↓           ↓           ↓                               │
│  GPU: [커널1실행]  [커널2실행]  [커널3실행]                     │
│                                                                 │
│  문제: 각 커널 런치마다 CPU-GPU 동기화 오버헤드                │
│        Decode의 작은 커널에서 오버헤드가 지배적                 │
└─────────────────────────────────────────────────────────────────┘

CUDA Graph:
┌─────────────────────────────────────────────────────────────────┐
│  캡처 단계:                                                     │
│  CPU: [커널1, 커널2, 커널3, ...] → Graph 캡처                  │
│                                                                 │
│  실행 단계:                                                     │
│  CPU: [Graph 런치] ─────────────────────────────────→          │
│  GPU: [커널1][커널2][커널3][...] 연속 실행                     │
│                                                                 │
│  효과: 커널 런치 오버헤드 제거, ~20-30% 속도 향상              │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 CUDA Graph의 제약

**입력 크기 고정 필요:**
```
CUDA Graph 제약:
- 그래프 캡처 시 텐서 크기가 고정됨
- 다른 크기 입력에 같은 그래프 사용 불가

문제:
- Continuous batching에서 배치 크기가 매 스텝 변함
- 해결: 패딩으로 고정 크기 맞추기
```

### 6.3 Padding Dispatch 전략

```python
# vllm/compilation/cuda_graph.py 기반 개념

class CUDAGraphDispatcher:
    def __init__(self, max_batch_sizes=[1, 2, 4, 8, 16, 32]):
        self.graphs = {}

        # 각 배치 크기에 대해 그래프 사전 캡처
        for batch_size in max_batch_sizes:
            self.graphs[batch_size] = self._capture_graph(batch_size)

    def dispatch(self, actual_batch_size):
        """실제 배치 크기에 맞는 그래프 선택"""

        # 가장 작은 충분한 그래프 크기 찾기
        for graph_size in sorted(self.graphs.keys()):
            if graph_size >= actual_batch_size:
                # 패딩 추가
                padding_needed = graph_size - actual_batch_size
                return self.graphs[graph_size], padding_needed

        # 너무 큰 배치는 eager 모드로 폴백
        return None, 0
```

### 6.4 패딩 오버헤드 분석

```
배치 크기 그래프: [1, 2, 4, 8, 16, 32, 64, 128]

실제 배치가 10일 때:
→ 그래프 16 사용, 패딩 6
→ 오버헤드: 6/16 = 37.5%

실제 배치가 30일 때:
→ 그래프 32 사용, 패딩 2
→ 오버헤드: 2/32 = 6.25%

평균 오버헤드:
그래프 간격이 2배면 평균 ~25% 패딩
그래프 간격이 1.5배면 평균 ~17% 패딩

Trade-off: 그래프 수 ↑ → 메모리 사용 ↑, 패딩 ↓
```

### 6.5 vLLM의 CUDA Graph 관리

```python
# vllm/v1/cudagraph_dispatcher.py 참조

class CUDAGraphDispatcher:
    """
    가변 배치 크기에 대한 CUDA Graph 디스패치
    """

    def __init__(self, cudagraph_batch_sizes: list[int]):
        # 배치 크기별 그래프 저장소
        self.batch_size_to_graph: dict[int, CUDAGraph] = {}
        self.cudagraph_batch_sizes = sorted(cudagraph_batch_sizes)

    def get_graph_for_batch(self, num_tokens: int) -> tuple[CUDAGraph, int]:
        """
        주어진 토큰 수에 적합한 CUDA Graph 반환
        """
        for graph_batch_size in self.cudagraph_batch_sizes:
            if graph_batch_size >= num_tokens:
                graph = self.batch_size_to_graph[graph_batch_size]
                padding = graph_batch_size - num_tokens
                return graph, padding

        # 폴백: eager execution
        return None, 0
```

---

## 7. 그래프 캐싱과 Latency 감소

### 7.1 그래프 캐싱의 효과

```
캡처 비용:
- 첫 실행 시 그래프 캡처: ~100ms
- 메모리 할당, 커널 컴파일 등

실행 비용:
- 캡처된 그래프 실행: ~5ms
- 일반 실행: ~7ms

절감:
- 커널 런치 오버헤드: ~30%
- 특히 작은 배치에서 효과적
```

### 7.2 Latency 성분 분석

```
LLM Decode Step의 Latency 성분:

┌─────────────────────────────────────────────────────────────────┐
│  1. 스케줄링 (CPU)          │  0.1ms  │  무시 가능           │
│  2. 입력 준비 (CPU→GPU)     │  0.2ms  │  데이터 전송         │
│  3. 커널 런치 (CPU)         │  0.5ms  │  ← CUDA Graph로 제거 │
│  4. 모델 연산 (GPU)         │  4.0ms  │  주요 비용           │
│  5. 샘플링 (GPU)            │  0.2ms  │  작은 비용           │
│  6. 결과 전송 (GPU→CPU)     │  0.1ms  │  무시 가능           │
├─────────────────────────────────────────────────────────────────┤
│  총계 (CUDA Graph 없음)     │  5.1ms  │                      │
│  총계 (CUDA Graph 있음)     │  4.6ms  │  10% 개선            │
└─────────────────────────────────────────────────────────────────┘

작은 배치(1-4)에서:
- 모델 연산: 1.0ms
- 커널 런치: 0.5ms
- 개선 효과: 33%!
```

### 7.3 Warmup과 캐시 빌드

```python
def warmup_cuda_graphs(model, batch_sizes, seq_lens):
    """
    서비스 시작 전 CUDA Graph 사전 캡처
    """
    graphs = {}

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            # 더미 입력 생성
            dummy_input = create_dummy_input(batch_size, seq_len)

            # 그래프 캡처
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = model(dummy_input)

            graphs[(batch_size, seq_len)] = graph

    return graphs

# 서비스 시작 시
print("Warming up CUDA graphs...")
graphs = warmup_cuda_graphs(
    model,
    batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
    seq_lens=[1]  # Decode는 항상 1
)
print(f"Captured {len(graphs)} CUDA graphs")
```

### 7.4 그래프 캐시 메모리 관리

```
CUDA Graph 메모리 사용:

각 그래프는 중간 텐서 공간을 보유:
- Activations: ~배치크기 × 히든크기 × 레이어수
- Workspace: cuBLAS/cuDNN 임시 공간

예시 (LLaMA-7B):
배치 128 그래프: ~2GB
배치 64 그래프: ~1GB
배치 32 그래프: ~0.5GB
...

총 그래프 메모리: 4-8GB (배치 크기 조합에 따라)

Trade-off:
- 많은 그래프: 낮은 패딩, 높은 메모리
- 적은 그래프: 높은 패딩, 낮은 메모리
```

---

## 8. Multi-Request와 Multi-Tenancy

### 8.1 Multi-Request 처리

```
단일 사용자, 다중 요청:
┌─────────────────────────────────────────────────────────────────┐
│  User A가 동시에 3개 요청 전송:                                 │
│  - Req1: "Translate to French: Hello"                          │
│  - Req2: "Summarize: [long text]"                              │
│  - Req3: "Code: fizzbuzz in Python"                            │
│                                                                 │
│  Scheduler가 배치로 묶어 처리:                                  │
│  Batch = [Req1, Req2, Req3]                                    │
│                                                                 │
│  각 요청 독립적으로 완료 후 응답                                │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Multi-Tenancy 문제

```
다중 사용자 환경의 문제:
┌─────────────────────────────────────────────────────────────────┐
│  1. 공정성 (Fairness)                                          │
│     - User A: 100 요청/분 vs User B: 1 요청/분                 │
│     - User B가 항상 뒤로 밀리면 안됨                           │
│                                                                 │
│  2. 격리 (Isolation)                                           │
│     - User A의 대량 요청이 User B의 latency에 영향 주면 안됨   │
│                                                                 │
│  3. 우선순위 (Priority)                                        │
│     - 유료 사용자 vs 무료 사용자                               │
│     - SLA 보장 필요                                            │
│                                                                 │
│  4. 리소스 제한 (Rate Limiting)                                │
│     - 사용자별 최대 동시 요청 수                               │
│     - 토큰/분 제한                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 vLLM의 해결 방안

**1. Priority Queue:**
```python
# vllm/v1/core/sched/request_queue.py

class PriorityRequestQueue:
    """우선순위 기반 요청 큐"""

    def add_request(self, request: Request):
        # 우선순위와 도착 시간으로 정렬
        heapq.heappush(
            self.queue,
            (request.priority, request.arrival_time, request)
        )

    def pop_request(self) -> Request:
        _, _, request = heapq.heappop(self.queue)
        return request
```

**2. Preemption Policy:**
```python
# 메모리 부족 시 낮은 우선순위 먼저 preempt
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(
        self.running,
        key=lambda r: (r.priority, r.arrival_time)  # 높은 priority = 낮은 우선순위
    )
```

**3. Token Budget 분배:**
```python
def allocate_token_budget(users, total_budget):
    """사용자별 공정한 토큰 예산 분배"""

    # Weighted fair queuing
    total_weight = sum(user.weight for user in users)

    for user in users:
        user.budget = total_budget * (user.weight / total_weight)
```

### 8.4 공정성 보장 전략

```
┌─────────────────────────────────────────────────────────────────┐
│  전략 1: Round-Robin                                           │
│  - 각 사용자에게 순차적으로 토큰 슬롯 할당                     │
│  - 단순하지만 throughput 최적화 어려움                         │
│                                                                 │
│  전략 2: Weighted Fair Queuing                                 │
│  - 사용자별 가중치에 따라 리소스 분배                          │
│  - SLA 티어별 차등 서비스 가능                                 │
│                                                                 │
│  전략 3: Deficit Round-Robin                                   │
│  - 이전에 못 받은 만큼 다음에 보상                             │
│  - 장기적 공정성 보장                                          │
│                                                                 │
│  전략 4: Virtual Clock                                         │
│  - 가상 시간 기반 스케줄링                                     │
│  - 예측 가능한 latency 보장                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 8.5 실제 구현 고려사항

```python
# Multi-tenancy 구현 예시

class MultiTenantScheduler:
    def __init__(self):
        self.tenant_queues: dict[str, RequestQueue] = {}
        self.tenant_configs: dict[str, TenantConfig] = {}

    def schedule(self) -> SchedulerOutput:
        # 테넌트별 가중치에 따라 토큰 예산 분배
        total_budget = self.max_num_scheduled_tokens
        tenant_budgets = self._compute_fair_budgets(total_budget)

        scheduled = []

        # 각 테넌트에서 예산만큼 스케줄
        for tenant_id, budget in tenant_budgets.items():
            tenant_reqs = self._schedule_tenant(tenant_id, budget)
            scheduled.extend(tenant_reqs)

        return SchedulerOutput(scheduled)

    def _compute_fair_budgets(self, total_budget):
        """Weighted fair queuing으로 예산 분배"""
        active_tenants = [
            t for t in self.tenant_queues
            if self.tenant_queues[t].has_requests()
        ]

        total_weight = sum(
            self.tenant_configs[t].weight
            for t in active_tenants
        )

        return {
            t: total_budget * (self.tenant_configs[t].weight / total_weight)
            for t in active_tenants
        }
```

---

## 9. 면접 예상 질문 및 답변

### Q1: Continuous Batching이 Static Batching보다 좋은 이유는?

**답변:**
Static Batching은 배치 내 가장 긴 요청이 완료될 때까지 다른 요청도 대기해야 합니다.

```
Static: 배치 완료 시간 = max(각 요청 길이)
→ 짧은 요청도 긴 요청 때문에 대기
→ GPU 슬롯 낭비 (평균 30-40%)
```

Continuous Batching은 **iteration 단위**로 배치를 재구성합니다.

```
Continuous: 완료된 요청 즉시 제거, 새 요청 추가
→ GPU 항상 최대 활용
→ 평균 throughput 1.5-2배 향상
```

### Q2: vLLM Scheduler가 Prefill과 Decode를 구분하는 방법은?

**답변:**
명시적 구분 없이 **num_computed_tokens**와 **num_tokens_with_spec**의 차이로 암묵적 결정됩니다.

```python
num_new_tokens = num_tokens_with_spec - num_computed_tokens

if num_new_tokens > 1:
    # Prefill 특성 (많은 토큰 처리)
else:
    # Decode 특성 (1 토큰)
```

이 접근법은 **Chunked Prefill**, **Prefix Caching**, **Speculative Decoding**을 자연스럽게 통합합니다.

### Q3: CUDA Graph가 LLM 추론을 빠르게 하는 원리는?

**답변:**
LLM Decode는 작은 연산이 많아 **커널 런치 오버헤드**가 지배적입니다.

```
일반 실행:
각 커널마다 CPU→GPU 명령 전송, 동기화
→ 커널당 ~10-50μs 오버헤드
→ 100개 커널 = 1-5ms 오버헤드

CUDA Graph:
모든 커널을 하나의 그래프로 캡처
→ 단일 런치로 모든 커널 실행
→ 오버헤드 ~100μs로 감소
```

작은 배치에서 **20-30%** 속도 향상을 얻을 수 있습니다.

### Q4: Preemption이 발생하는 상황과 처리 방법은?

**답변:**
**상황:** GPU 메모리 부족으로 새 요청의 KV 블록 할당 불가

**처리:**
1. 낮은 우선순위(또는 나중 도착) 요청 선택
2. KV 캐시 해제 (또는 CPU로 swap)
3. 요청을 WAITING 큐 앞쪽에 재삽입
4. 메모리 확보 후 재개

```python
# vLLM의 Preemption
preempted_req = self.running.pop()  # FCFS: 마지막 요청
self.kv_cache_manager.free(preempted_req)
preempted_req.status = RequestStatus.PREEMPTED
self.waiting.prepend_request(preempted_req)  # 앞에 재삽입
```

### Q5: Token Budget이란 무엇이고 어떻게 관리되나요?

**답변:**
**Token Budget:** 한 스케줄링 스텝에서 처리할 수 있는 최대 토큰 수

```python
max_num_scheduled_tokens = 4096  # 예시

# 예산 분배
running_reqs: 32 × 1 = 32 tokens (decode)
new_prefill: 1 × 500 = 500 tokens (chunked)
remaining: 4096 - 532 = 3564 tokens

# 추가 요청에 할당 가능
```

이를 통해 GPU 메모리와 연산 자원을 효율적으로 관리합니다.

### Q6: Multi-Tenancy 환경에서 공정성을 보장하는 방법은?

**답변:**
vLLM은 여러 전략을 지원합니다:

1. **Priority Queue:** 우선순위 기반 스케줄링
2. **Weighted Fair Queuing:** 테넌트별 가중치에 따른 리소스 분배
3. **Preemption Policy:** 메모리 부족 시 낮은 우선순위 먼저 중단

```
실제 구현:
- 테넌트별 요청 큐 분리
- 토큰 예산의 가중치 기반 분배
- SLA 티어에 따른 우선순위 차등
```

---

## 참고 자료

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf)
- vLLM 소스코드:
  - `vllm/v1/core/sched/scheduler.py` - 스케줄러 핵심 로직
  - `vllm/v1/core/sched/request_queue.py` - 요청 큐 구현
  - `vllm/compilation/cuda_graph.py` - CUDA Graph 관리
  - `vllm/v1/cudagraph_dispatcher.py` - 그래프 디스패치
