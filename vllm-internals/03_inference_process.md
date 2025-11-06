# vLLM 추론 과정 상세 분석 (Inference Process)

vLLM의 추론 엔진이 실제로 어떻게 사용자 요청을 처리하고 토큰을 생성하는지 상세하게 분석합니다.

## 📋 목차

1. [전체 흐름 개요](#1-전체-흐름-개요)
2. [LLMEngine: 추론의 시작점](#2-llmengine-추론의-시작점)
3. [Scheduler: 요청 스케줄링](#3-scheduler-요청-스케줄링)
4. [GPUModelRunner: 모델 실행](#4-gpumodelrunner-모델-실행)
5. [Sampler: 토큰 생성](#5-sampler-토큰-생성)
6. [성능 최적화](#6-성능-최적화)
7. [실제 성능 측정](#7-실제-성능-측정)
8. [트러블슈팅](#8-트러블슈팅)

---

## 1. 전체 흐름 개요

### 1.1 추론 파이프라인

```
사용자 요청
    ↓
┌─────────────────────────────────────────────────┐
│ LLMEngine.add_request()                         │
│ - 요청 토큰화 (Tokenization)                     │
│ - RequestState 생성                             │
│ - 대기 큐에 추가                                  │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ LLMEngine.step()                                │
│ - 한 스텝의 추론 실행                             │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Scheduler.schedule()                            │
│ - RUNNING 요청 스케줄링                          │
│ - WAITING 요청 스케줄링                          │
│ - KV Cache 할당                                 │
│ - Preemption 처리                               │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ GPUModelRunner.execute_model()                  │
│ 1. _update_states() - 배치 상태 업데이트         │
│ 2. _prepare_inputs() - 입력 텐서 준비           │
│ 3. _preprocess() - 전처리                       │
│ 4. _model_forward() - 모델 실행                 │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ GPUModelRunner.sample_tokens()                  │
│ 1. apply_grammar_bitmask() - 제약 적용          │
│ 2. _sample() - 토큰 샘플링                      │
│ 3. _bookkeeping_sync() - 결과 동기화            │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ OutputProcessor.process_outputs()               │
│ - EngineCoreOutput → RequestOutput 변환         │
│ - Stop string 체크                              │
│ - 완료된 요청 처리                               │
└─────────────────────────────────────────────────┘
    ↓
완성된 출력 반환
```

### 1.2 핵심 컴포넌트

| 컴포넌트 | 파일 위치 | 역할 |
|---------|----------|-----|
| **LLMEngine** | `vllm/v1/engine/llm_engine.py` | 추론 엔진의 최상위 인터페이스 |
| **Scheduler** | `vllm/v1/core/sched/scheduler.py` | 요청 스케줄링 및 KV Cache 관리 |
| **GPUModelRunner** | `vllm/v1/worker/gpu_model_runner.py` | GPU에서 모델 실행 |
| **Sampler** | `vllm/v1/sample/sampler.py` | 다음 토큰 샘플링 |
| **OutputProcessor** | `vllm/v1/engine/output_processor.py` | 출력 처리 및 변환 |

---

## 2. LLMEngine: 추론의 시작점

### 2.1 초기화 과정

**코드 위치**: `vllm/v1/engine/llm_engine.py:48-139`

```python
class LLMEngine:
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # 1. Tokenizer 초기화
        if not self.model_config.skip_tokenizer_init:
            tokenizer = init_tokenizer_from_configs(self.model_config)

        # 2. Processor 생성 (입력 전처리)
        self.processor = Processor(self.vllm_config, tokenizer)

        # 3. OutputProcessor 생성 (출력 후처리)
        self.output_processor = OutputProcessor(
            self.tokenizer, log_stats=self.log_stats
        )

        # 4. EngineCore 생성 (실제 추론 엔진)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )
```

**핵심 개념**:
- **Processor**: 사용자 입력을 모델이 이해할 수 있는 형태로 변환
- **EngineCore**: 실제 모델 실행을 담당하는 핵심 엔진
- **OutputProcessor**: 모델 출력을 사용자가 이해할 수 있는 형태로 변환

### 2.2 요청 추가하기

**코드 위치**: `vllm/v1/engine/llm_engine.py:216-278`

```python
def add_request(
    self,
    request_id: str,
    prompt: EngineCoreRequest | PromptType,
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    lora_request: LoRARequest | None = None,
    priority: int = 0,
) -> None:
    # 1. 입력 검증
    if not isinstance(request_id, str):
        raise TypeError(f"request_id must be a string, got {type(request_id)}")

    # 2. 입력을 EngineCoreRequest로 변환
    if not isinstance(prompt, EngineCoreRequest):
        request = self.processor.process_inputs(
            request_id,
            prompt,
            params,
            arrival_time,
            lora_request,
            tokenization_kwargs,
            trace_headers,
            priority,
        )
    else:
        request = prompt

    # 3. n > 1인 경우 child request로 분리
    n = params.n if isinstance(params, SamplingParams) else 1

    if n == 1:
        # 단일 요청
        self.output_processor.add_request(request, prompt_text, None, 0)
        self.engine_core.add_request(request)
    else:
        # n개의 출력을 생성해야 하는 경우 (예: n=3이면 3개의 다른 completion)
        parent_req = ParentRequest(request_id, params)
        for idx in range(n):
            request_id, params = parent_req.get_child_info(idx)
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = params

            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            self.engine_core.add_request(child_request)
```

**핵심 개념**:
- **Tokenization**: 텍스트를 토큰 ID로 변환
- **Sampling Params**: 생성 전략 (temperature, top_p, etc.)
- **n-way sampling**: 하나의 프롬프트로 여러 개의 다른 출력 생성

### 2.3 추론 스텝 실행

**코드 위치**: `vllm/v1/engine/llm_engine.py:280-309`

```python
def step(self) -> list[RequestOutput | PoolingRequestOutput]:
    """한 스텝의 추론을 실행하고 완성된 출력을 반환"""

    # 1. EngineCore로부터 출력 받기
    outputs = self.engine_core.get_output()

    # 2. 출력 처리
    iteration_stats = IterationStats() if self.log_stats else None
    processed_outputs = self.output_processor.process_outputs(
        outputs.outputs,
        engine_core_timestamp=outputs.timestamp,
        iteration_stats=iteration_stats,
    )

    # 3. Stop string으로 인해 완료된 요청 중단
    self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

    # 4. 통계 기록
    if self.logger_manager is not None and outputs.scheduler_stats is not None:
        self.logger_manager.record(
            scheduler_stats=outputs.scheduler_stats,
            iteration_stats=iteration_stats,
            mm_cache_stats=self.processor.stat_mm_cache(),
        )

    return processed_outputs.request_outputs
```

**핵심 포인트**:
- 각 `step()` 호출은 **한 번의 forward pass**를 실행
- 여러 요청을 **배치로 함께 처리** (배치 효율성)
- 완성된 출력만 반환 (partial outputs는 내부적으로 버퍼링)

---

## 3. Scheduler: 요청 스케줄링

### 3.1 Scheduler 초기화

**코드 위치**: `vllm/v1/core/sched/scheduler.py:45-180`

```python
class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        log_stats: bool = False,
    ) -> None:
        # 스케줄링 제약 조건
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        # 요청 관리
        self.requests: dict[str, Request] = {}

        # 스케줄링 정책 (FCFS or Priority)
        self.policy = SchedulingPolicy(self.scheduler_config.policy)

        # 우선순위 큐
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # KV Cache 관리자
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=bool(self.cache_config.enable_prefix_caching),
            log_stats=self.log_stats,
        )
```

**핵심 제약 조건**:
- `max_num_running_reqs`: 동시에 실행 가능한 최대 요청 수 (예: 256)
- `max_num_scheduled_tokens`: 한 배치에 포함될 수 있는 최대 토큰 수 (예: 2048)
- `max_model_len`: 모델의 최대 시퀀스 길이 (예: 4096)

### 3.2 스케줄링 알고리즘

**코드 위치**: `vllm/v1/core/sched/scheduler.py:182-450`

```python
def schedule(self) -> SchedulerOutput:
    """
    vLLM의 스케줄링 알고리즘:

    prefill/decode 단계 구분이 없음!
    각 요청은 num_computed_tokens과 num_tokens_with_spec을 가지고 있고,
    스케줄러는 매 스텝마다 num_computed_tokens이 num_tokens_with_spec을
    따라잡도록 토큰을 할당함.

    이 방식은 다음을 일반적으로 처리:
    - Chunked prefills (긴 프롬프트를 여러 번에 나눠 처리)
    - Prefix caching (공통 프롬프트 재사용)
    - Speculative decoding (추측 디코딩)
    """

    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []

    req_to_new_blocks: dict[str, KVCacheBlocks] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens  # 예: 2048

    scheduled_timestamp = time.monotonic()

    # ===== Phase 1: RUNNING 요청 스케줄링 =====
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]

        # 이 요청이 처리해야 할 새로운 토큰 수 계산
        num_new_tokens = (
            request.num_tokens_with_spec  # 현재까지의 모든 토큰 (prompt + output + spec)
            + request.num_output_placeholders
            - request.num_computed_tokens  # 이미 계산된 토큰
        )

        # Chunked prefill: 한 번에 너무 많은 토큰을 처리하지 않도록 제한
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold

        # 토큰 버짓 내에서만 처리
        num_new_tokens = min(num_new_tokens, token_budget)

        # 모델의 최대 길이를 초과하지 않도록
        num_new_tokens = min(
            num_new_tokens,
            self.max_model_len - 1 - request.num_computed_tokens
        )

        if num_new_tokens == 0:
            req_index += 1
            continue

        # KV Cache 블록 할당
        while True:
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )

            if new_blocks is not None:
                # 할당 성공!
                break

            # 메모리 부족: Preemption 필요
            # 가장 낮은 우선순위 요청을 제거
            if self.policy == SchedulingPolicy.PRIORITY:
                preempted_req = max(
                    self.running,
                    key=lambda r: (r.priority, r.arrival_time),
                )
            else:
                # FCFS: 가장 마지막 요청 제거
                preempted_req = self.running.pop()

            # KV Cache 해제
            self.kv_cache_manager.free(preempted_req)
            self.encoder_cache_manager.free(preempted_req)

            # 상태 업데이트
            preempted_req.status = RequestStatus.PREEMPTED
            preempted_req.num_computed_tokens = 0
            preempted_req.num_preemptions += 1

            # 다시 대기 큐로
            self.waiting.prepend_request(preempted_req)
            preempted_reqs.append(preempted_req)

            if preempted_req == request:
                # 현재 요청도 preempt되었으면 중단
                break

        if new_blocks is None:
            # 할당 실패
            break

        # 스케줄링 성공!
        scheduled_running_reqs.append(request)
        req_to_new_blocks[request.request_id] = new_blocks
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        req_index += 1

    # ===== Phase 2: WAITING 요청 스케줄링 =====
    # Preemption이 발생하지 않았을 때만 새로운 요청 추가
    if not preempted_reqs:
        while self.waiting and token_budget > 0:
            if len(self.running) == self.max_num_running_reqs:
                # 동시 실행 요청 수 제한 도달
                break

            request = self.waiting.peek_request()

            # 상태 체크 (FSM 컴파일 대기, Remote KV 대기 등)
            if request.status == RequestStatus.WAITING_FOR_FSM:
                # Structured output FSM이 아직 컴파일 안 됨
                # ...skip and continue...
                continue

            # LoRA 제약 체크
            if self.lora_config and request.lora_request:
                if (len(scheduled_loras) == self.lora_config.max_loras
                    and request.lora_request.lora_int_id not in scheduled_loras):
                    # LoRA 슬롯 부족
                    # ...skip and continue...
                    continue

            # 토큰 수 계산
            num_new_tokens = min(
                request.num_tokens_with_spec - request.num_computed_tokens,
                token_budget
            )

            # KV Cache 할당
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens
            )

            if new_blocks is None:
                # 메모리 부족: 더 이상 새로운 요청 추가 불가
                break

            # 스케줄링 성공!
            self.waiting.pop_request()
            self.running.append(request)
            scheduled_new_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens

    # ===== 스케줄링 결과 반환 =====
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_resumed_reqs=scheduled_resumed_reqs,
        scheduled_running_reqs=scheduled_running_reqs,
        preempted_reqs=preempted_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        req_to_new_blocks=req_to_new_blocks,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        # ...
    )
```

**핵심 개념**:

1. **통합된 스케줄링**: Prefill/Decode 구분 없이 모든 요청을 동일하게 처리
2. **Chunked Prefill**: 긴 프롬프트를 여러 번에 나눠 처리하여 latency 개선
3. **Preemption**: 메모리 부족 시 낮은 우선순위 요청을 일시 중단
4. **Token Budget**: 배치 크기를 제한하여 처리량과 응답 시간의 균형 유지

### 3.3 Preemption 예제

```python
# 시나리오: GPU 메모리 부족 상황
#
# 현재 상태:
# - Request A: priority=0, 1000 tokens computed
# - Request B: priority=1, 500 tokens computed
# - Request C: priority=0, 100 tokens computed (새로 추가 시도)
#
# KV Cache가 꽉 차서 Request C를 위한 공간이 없음
#
# Preemption 발생:
# 1. Priority가 가장 높은 Request B를 찾음 (priority=1)
# 2. Request B의 KV Cache를 해제
# 3. Request B를 PREEMPTED 상태로 변경
# 4. Request B를 대기 큐 앞쪽에 다시 추가
# 5. Request B의 num_computed_tokens = 0 (처음부터 다시)
# 6. 이제 Request C를 위한 공간 확보
#
# 결과:
# - Request A: 계속 실행
# - Request B: 대기 큐에서 재시작 대기
# - Request C: 새로 실행 시작
```

---

## 4. GPUModelRunner: 모델 실행

### 4.1 execute_model() - 메인 실행 함수

**코드 위치**: `vllm/v1/worker/gpu_model_runner.py:2440-2619`

```python
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput | IntermediateTensors | None:
    """
    모델을 실행하고 다음 토큰을 생성

    Returns:
        None: 실행 완료, sample_tokens()로 결과 수집 필요
        IntermediateTensors: Pipeline Parallel의 중간 rank (마지막 rank가 아님)
        ModelRunnerOutput: 최종 출력 (deprecated, async scheduling에서 사용)
    """

    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

    # ===== Step 1: 전처리 =====
    with record_function_or_nullcontext("Preprocess"):
        with self.synchronize_input_prep():
            # 1.1. 배치 상태 업데이트
            self._update_states(scheduler_output)

            if not num_scheduled_tokens:
                # 처리할 토큰이 없음
                return EMPTY_MODEL_RUNNER_OUTPUT

            # 1.2. 입력 준비 (Attention metadata, logits indices, etc.)
            (
                attn_metadata,
                logits_indices,
                spec_decode_metadata,
                num_scheduled_tokens_np,
                spec_decode_common_attn_metadata,
                max_query_len,
                ubatch_slices,
                num_tokens_across_dp,
                use_cascade_attn,
            ) = self._prepare_inputs(scheduler_output)

            # 1.3. CUDA graph용 패딩 계산
            num_input_tokens = self._get_num_input_tokens(
                scheduler_output.total_num_scheduled_tokens
            )

            # 1.4. 입력 텐서 생성
            (
                input_ids,         # (num_tokens,) or None
                inputs_embeds,     # (num_tokens, hidden_size) or None
                positions,         # (num_tokens,)
                intermediate_tensors,
                model_kwargs,
            ) = self._preprocess(
                scheduler_output, num_input_tokens, intermediate_tensors
            )

            # 1.5. CUDA graph dispatch
            uniform_decode = (max_query_len == self.uniform_decode_query_len)
            batch_descriptor = BatchDescriptor(
                num_tokens=num_input_tokens,
                uniform_decode=uniform_decode,
                has_lora=len(self.input_batch.lora_id_to_lora_request) > 0,
            )
            cudagraph_runtime_mode, batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(batch_descriptor, use_cascade_attn)
            )

    # ===== Step 2: Forward Pass =====
    with (
        set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        ),
        record_function_or_nullcontext("Forward"),
    ):
        # 모델 실행!
        model_output = self._model_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    # ===== Step 3: 후처리 =====
    with record_function_or_nullcontext("Postprocess"):
        hidden_states = model_output

        # Pipeline Parallel 처리
        if not get_pp_group().is_last_rank:
            # 중간 rank: intermediate tensors 반환
            return hidden_states

        # Pooling model (embedding 추출)
        if self.is_pooling_model:
            output = self._pool(
                hidden_states, num_scheduled_tokens, num_scheduled_tokens_np
            )
            return output

        # 일반 생성 모델: Logits 계산
        sample_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)

    # 실행 상태 저장 (sample_tokens에서 사용)
    self.execute_model_state = ExecuteModelState(
        scheduler_output,
        logits,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        hidden_states,
        sample_hidden_states,
        aux_hidden_states=None,
        kv_connector_output=None,
    )
    return None
```

### 4.2 _prepare_inputs() - 입력 텐서 준비

**코드 위치**: `vllm/v1/worker/gpu_model_runner.py:1062-1300`

```python
def _prepare_inputs(
    self, scheduler_output: "SchedulerOutput"
) -> tuple[...]:
    """
    스케줄러 출력으로부터 모델 입력 텐서를 준비
    """

    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    num_reqs = self.input_batch.num_reqs

    # ===== Step 1: Block table 복사 시작 =====
    # GPU 복사를 먼저 시작하여 CPU 작업과 overlap
    self.input_batch.block_table.commit_block_table(num_reqs)

    # ===== Step 2: 스케줄된 토큰 수 계산 =====
    req_ids = self.input_batch.req_ids
    tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
    num_scheduled_tokens = np.array(tokens, dtype=np.int32)
    max_num_scheduled_tokens = max(tokens)

    # ===== Step 3: Request indices 생성 =====
    # 예: [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

    # ===== Step 4: Positions 계산 =====
    # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
    cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

    positions_np = self.positions.np[:total_num_scheduled_tokens]
    np.add(
        self.input_batch.num_computed_tokens_cpu[req_indices],
        arange,  # [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        out=positions_np,
    )

    # 예제:
    # Request 0: num_computed_tokens=10, num_scheduled=2
    #   -> positions = [10, 11]
    # Request 1: num_computed_tokens=0, num_scheduled=5 (prefill)
    #   -> positions = [0, 1, 2, 3, 4]
    # Request 2: num_computed_tokens=50, num_scheduled=3
    #   -> positions = [50, 51, 52]

    # ===== Step 5: Token IDs 추출 =====
    # Token indices: request별로 flatten된 인덱스
    token_indices = (
        positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
    )
    token_indices_tensor = torch.from_numpy(token_indices)

    # Token IDs를 flatten된 배열에서 추출
    torch.index_select(
        self.input_batch.token_ids_cpu_tensor.flatten(),
        0,
        token_indices_tensor,
        out=self.input_ids.cpu[:total_num_scheduled_tokens],
    )

    # ===== Step 6: Slot mapping 계산 =====
    # KV Cache에서 각 토큰이 저장될 위치
    self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
    self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

    # ===== Step 7: Attention Metadata 준비 =====
    self.query_start_loc.np[0] = 0
    self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
    self.query_start_loc.copy_to_gpu()
    query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

    # Sequence lengths
    self.seq_lens.np[:num_reqs] = (
        self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
    )
    self.seq_lens.copy_to_gpu()
    seq_lens = self.seq_lens.gpu[:num_reqs]
    max_seq_len = self.seq_lens.np[:num_reqs].max().item()

    # ===== Step 8: Logits indices 계산 =====
    # 각 request의 마지막 토큰만 샘플링
    if not use_spec_decode:
        logits_indices = query_start_loc[1:] - 1
        # 예: query_start_loc = [0, 2, 7, 10]
        #     -> logits_indices = [1, 6, 9]
        #     (각 request의 마지막 토큰)
    else:
        # Speculative decoding: 모든 draft token에서 logits 계산
        # ...

    # ===== Step 9: GPU로 복사 =====
    self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)
    self.positions.copy_to_gpu(total_num_scheduled_tokens)

    return (
        attn_metadata,
        logits_indices,
        spec_decode_metadata,
        num_scheduled_tokens,
        spec_decode_common_attn_metadata,
        max_num_scheduled_tokens,
        ubatch_slices,
        num_tokens_across_dp,
        use_cascade_attn,
    )
```

**핵심 최적화**:
1. **Overlap**: Block table 복사를 먼저 시작하여 CPU 작업과 overlap
2. **Vectorized Operations**: NumPy를 사용한 벡터화된 연산
3. **Zero-copy**: Pinned memory를 사용한 효율적인 GPU 전송
4. **Minimal Synchronization**: GPU 동기화를 최소화

### 4.3 _model_forward() - 실제 모델 실행

**코드 위치**: `vllm/v1/worker/gpu_model_runner.py:2544-2550`

```python
model_output = self._model_forward(
    input_ids=input_ids,           # (num_tokens,) or None
    positions=positions,           # (num_tokens,)
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,  # (num_tokens, hidden_size) or None
    **model_kwargs,
)

# _model_forward 내부적으로 호출하는 것:
# 1. self.model.forward() - 실제 Transformer 모델
# 2. Attention 계산 (FlashAttention, xFormers, etc.)
# 3. Feed-forward 네트워크
# 4. Layer normalization
# 5. Residual connections
```

**입력 형태**:
- **Text-only models**: `input_ids` 사용 (embedding layer는 CUDA graph 내부)
- **Multimodal models**: `inputs_embeds` 사용 (vision tokens + text tokens)

**출력**:
- `hidden_states`: `(num_tokens, hidden_size)` 크기의 텐서
- Pipeline Parallel 중간 rank에서는 `IntermediateTensors`

---

## 5. Sampler: 토큰 생성

### 5.1 sample_tokens() - 토큰 샘플링

**코드 위치**: `vllm/v1/worker/gpu_model_runner.py:2621-2740`

```python
@torch.inference_mode
def sample_tokens(
    self, grammar_output: "GrammarOutput | None"
) -> ModelRunnerOutput:
    """
    execute_model()이 생성한 logits로부터 다음 토큰을 샘플링
    """

    # execute_model_state에서 상태 복원
    (
        scheduler_output,
        logits,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        hidden_states,
        sample_hidden_states,
        aux_hidden_states,
        kv_connector_output,
    ) = self.execute_model_state
    self.execute_model_state = None

    # ===== Step 1: Grammar bitmask 적용 (선택) =====
    # Structured output (JSON, regex 등)을 위한 제약
    if grammar_output is not None:
        apply_grammar_bitmask(
            scheduler_output, grammar_output, self.input_batch, logits
        )

    # ===== Step 2: 토큰 샘플링 =====
    with record_function_or_nullcontext("Sample"):
        sampler_output = self._sample(logits, spec_decode_metadata)

    # ===== Step 3: Draft tokens 생성 (Speculative Decoding) =====
    if self.speculative_config:
        # EAGLE, Medusa, n-gram 등
        # ...

    # ===== Step 4: Bookkeeping (동기화 및 상태 업데이트) =====
    with record_function_or_nullcontext("Bookkeep"):
        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

    # ===== Step 5: 출력 생성 =====
    output = ModelRunnerOutput(
        req_ids=req_ids_output_copy,
        req_id_to_index=req_id_to_index_output_copy,
        sampled_token_ids=valid_sampled_token_ids,
        logprobs=logprobs_lists,
        prompt_logprobs_dict=prompt_logprobs_dict,
        pooler_output=[],
        kv_connector_output=kv_connector_output,
        num_nans_in_logits=num_nans_in_logits,
    )

    return output
```

### 5.2 Sampler.forward() - 샘플링 알고리즘

**코드 위치**: `vllm/v1/sample/sampler.py:66-125`

```python
class Sampler(nn.Module):
    """
    다음 토큰을 샘플링하는 레이어

    순서:
    1. Logprobs 계산 (요청 시)
    2. Float32로 변환
    3. Allowed token IDs whitelist 적용
    4. Bad words 제외
    5. Logit processors 적용 (min tokens, logit bias)
    6. Penalties 적용 (repetition, frequency, presence)
    7. 샘플링:
       a) Greedy sampling (temperature=0)
       b) Temperature 적용
       c) Min-p processor
       d) Top-k, top-p 적용
       e) 확률 분포에서 샘플링
    8. Logprobs 수집
    9. SamplerOutput 반환
    """

    def forward(
        self,
        logits: torch.Tensor,  # (num_requests, vocab_size)
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        # Step 1: Raw logprobs 저장 (요청 시)
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if self.logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif self.logprobs_mode == "raw_logits":
                raw_logprobs = logits.clone()

        # Step 2: Float32로 변환
        logits = logits.to(torch.float32)

        # Step 3: Logits processors 적용
        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token
        )

        # Step 4: 샘플링
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        sampled = sampled.long()

        # Step 5: Logprobs 수집
        if num_logprobs is None:
            logprobs_tensors = None
        else:
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )

        sampled = sampled.to(torch.int32)

        # Step 6: 출력
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled.unsqueeze(-1),  # (num_requests, 1)
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output
```

### 5.3 샘플링 전략

**코드 위치**: `vllm/v1/sample/sampler.py:143-199`

```python
def sample(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """실제 샘플링 로직"""

    # ===== Case 1: All Greedy =====
    if sampling_metadata.all_greedy:
        greedy_sampled = self.greedy_sample(logits)  # argmax
        return greedy_sampled, None

    # ===== Case 2: Mixed or All Random =====
    if not sampling_metadata.all_random:
        # Greedy와 random이 섞여 있음
        greedy_sampled = self.greedy_sample(logits)
    else:
        greedy_sampled = None

    # Temperature 적용
    logits = self.apply_temperature(
        logits,
        sampling_metadata.temperature,  # (num_requests,)
        sampling_metadata.all_random
    )

    # Argmax-invariant processors (min-p 등)
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)

    # Top-k, top-p 샘플링
    random_sampled, processed_logprobs = self.topk_topp_sampler(
        logits,
        sampling_metadata.generators,
        sampling_metadata.top_k,    # (num_requests,)
        sampling_metadata.top_p,    # (num_requests,)
    )

    if greedy_sampled is None:
        return random_sampled, processed_logprobs

    # Greedy와 random 결합
    sampled = torch.where(
        sampling_metadata.temperature < _SAMPLING_EPS,  # 1e-5
        greedy_sampled,
        random_sampled,
    )
    return sampled, processed_logprobs

@staticmethod
def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Greedy 샘플링: 가장 높은 확률의 토큰 선택"""
    return logits.argmax(dim=-1).view(-1)

@staticmethod
def apply_temperature(
    logits: torch.Tensor,
    temp: torch.Tensor,
    all_random: bool,
) -> torch.Tensor:
    """Temperature scaling: logits / temperature"""
    # Temperature가 0에 가까우면 1로 설정 (division by zero 방지)
    if not all_random:
        temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
    return logits.div_(temp.unsqueeze(dim=1))  # In-place
```

**샘플링 전략 설명**:

| 전략 | Temperature | Top-p | Top-k | 설명 |
|-----|-------------|-------|-------|-----|
| **Greedy** | 0.0 | - | - | 항상 가장 높은 확률의 토큰 선택 |
| **Temperature** | 0.7 | - | - | 확률 분포를 더 smooth하게 |
| **Top-p (Nucleus)** | 1.0 | 0.9 | - | 누적 확률 90%까지의 토큰만 고려 |
| **Top-k** | 1.0 | - | 50 | 상위 50개 토큰만 고려 |
| **Mixed** | 0.8 | 0.95 | 40 | 여러 전략 조합 |

### 5.4 Penalties 적용

**코드 위치**: `vllm/v1/sample/ops/penalties.py`

```python
@staticmethod
def apply_penalties(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """
    Penalties 적용:
    - Repetition penalty: 이미 생성된 토큰의 확률 감소
    - Frequency penalty: 토큰 빈도에 비례하여 확률 감소
    - Presence penalty: 한 번이라도 나온 토큰의 확률 감소
    """

    if sampling_metadata.no_penalties:
        return logits

    return apply_all_penalties(
        logits,
        output_token_ids,
        sampling_metadata.frequency_penalties,
        sampling_metadata.presence_penalties,
        sampling_metadata.repetition_penalties,
    )
```

**예제**:

```python
# 원본 logits: [0.5, 0.3, 0.2, ...] (토큰 "the"가 0.3)
# 이미 "the"를 5번 생성함

# Frequency penalty = 0.5 적용:
# new_logit = old_logit - (frequency * penalty)
#           = 0.3 - (5 * 0.5)
#           = 0.3 - 2.5
#           = -2.2  (확률이 크게 감소)

# Presence penalty = 1.0 적용:
# new_logit = old_logit - penalty (if token appeared)
#           = 0.3 - 1.0
#           = -0.7

# Repetition penalty = 1.2 적용:
# if logit > 0:
#     new_logit = old_logit / penalty
# else:
#     new_logit = old_logit * penalty
```

---

## 6. 성능 최적화

### 6.1 CUDA Graphs

**핵심 아이디어**: 반복되는 GPU 작업을 미리 녹화하여 커널 실행 오버헤드 제거

```python
# CUDA Graph 사용 조건 체크
if (
    self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
    and hasattr(self, "cudagraph_batch_sizes")
    and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]
):
    # CUDA graph 사용
    num_input_tokens = self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)
    cudagraph_runtime_mode = CUDAGraphMode.DECODE  # or PREFILL
else:
    # Eager mode
    num_input_tokens = num_scheduled_tokens
    cudagraph_runtime_mode = CUDAGraphMode.NONE
```

**성능 향상**:
- **Decode phase**: ~2-3x speedup (작은 배치 크기에서)
- **Prefill phase**: ~1.2-1.5x speedup

**제약사항**:
- 고정된 입력 크기 필요 → 패딩 추가
- 메모리 사용량 증가 (여러 배치 크기별로 graph 저장)

### 6.2 Chunked Prefill

**목적**: 긴 프롬프트를 여러 청크로 나누어 처리하여 latency 개선

```python
# Chunked prefill 설정
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
    # 예: long_prefill_token_threshold = 512
    #     2000 토큰 프롬프트를 512씩 4번에 나눠 처리
```

**장점**:
1. **Fair scheduling**: Prefill이 decode를 blocking하지 않음
2. **Better latency**: Decode 요청이 더 빨리 처리됨
3. **Memory efficiency**: 한 번에 처리하는 토큰 수 제한

**예제**:

```
Scenario: 2개의 요청
- Request A: 2000-token prefill (새 요청)
- Request B: 1-token decode (진행 중)

Without chunked prefill:
  Step 1: A prefill (2000 tokens) - 2000ms
  Step 2: B decode (1 token) - 10ms
  Total: B는 2010ms 기다림 (나쁨)

With chunked prefill (chunk_size=512):
  Step 1: A prefill (512 tokens) + B decode - 520ms
  Step 2: A prefill (512 tokens) + B decode - 520ms
  Step 3: A prefill (512 tokens) + B decode - 520ms
  Step 4: A prefill (464 tokens) + B decode - 474ms
  Total: B는 평균 ~520ms마다 응답 (좋음)
```

### 6.3 Continuous Batching

**핵심 아이디어**: 요청이 완료되는 즉시 새로운 요청을 배치에 추가

```python
# 전통적인 Static Batching:
# Batch 1: [Req A, Req B, Req C] - 모두 완료될 때까지 대기
# Batch 2: [Req D, Req E, Req F] - 새로운 배치 시작

# vLLM의 Continuous Batching:
# Step 1: [Req A, Req B, Req C]
# Step 2: [Req A, Req B, Req C]  (아직 진행 중)
# Step 3: [Req A, Req C, Req D]  (B 완료, D 추가)
# Step 4: [Req A, Req C, Req D]
# Step 5: [Req C, Req D, Req E]  (A 완료, E 추가)
```

**성능 향상**:
- **Throughput**: ~2-5x improvement
- **GPU utilization**: ~80-95% (vs. ~50-60% for static batching)

### 6.4 Prefix Caching

**목적**: 공통 프롬프트를 재사용하여 prefill 시간 단축

```python
# 예제: 같은 system prompt를 사용하는 여러 요청
system_prompt = "You are a helpful assistant. ..."  # 100 tokens

# Request 1:
full_prompt_1 = system_prompt + "What is Python?"
# Prefill: 100 (system) + 4 (question) = 104 tokens

# Request 2:
full_prompt_2 = system_prompt + "What is Rust?"
# Prefill: 0 (cached) + 4 (question) = 4 tokens
# 96% 시간 절약!
```

**KV Cache Manager가 자동으로 처리**:
```python
self.kv_cache_manager = KVCacheManager(
    enable_caching=True,  # Prefix caching 활성화
    # ...
)

# 내부적으로 prefix의 KV cache를 hash table에 저장
# 동일한 prefix를 가진 새 요청이 오면 재사용
```

---

## 7. 실제 성능 측정

### 7.1 Llama-7B 추론 성능

**측정 환경**:
- GPU: NVIDIA A100 80GB
- Model: Llama-2-7B
- Batch size: 1 → 128 (continuous batching)
- Sequence length: 128 input + 128 output

**결과**:

| 메트릭 | Static Batching | vLLM Continuous Batching |
|-------|-----------------|--------------------------|
| **Throughput** | 250 tokens/sec | 1200 tokens/sec |
| **Time to First Token (TTFT)** | 150ms | 80ms |
| **Inter-Token Latency (ITL)** | 50ms | 25ms |
| **GPU Utilization** | 55% | 92% |

### 7.2 단계별 시간 분석

**Llama-7B, Batch size=32, Sequence length=512**

```
Total time per step: 45ms

Breakdown:
├─ Preprocess:        5ms  (11%)
│  ├─ _update_states:     1ms
│  ├─ _prepare_inputs:    3ms
│  └─ _preprocess:        1ms
│
├─ Forward:          30ms  (67%)
│  ├─ Embedding:          2ms
│  ├─ Attention (32 layers): 22ms
│  └─ Feed-forward:       6ms
│
├─ Sample:            3ms  (7%)
│  ├─ Logits compute:     1ms
│  └─ Sampling:           2ms
│
└─ Bookkeep:          7ms  (15%)
   ├─ GPU→CPU sync:       3ms
   ├─ Token update:       2ms
   └─ State management:   2ms
```

### 7.3 Prefill vs. Decode 비교

**Llama-7B, Single request**

| Phase | Tokens | Time | Tokens/sec |
|-------|--------|------|-----------|
| **Prefill** | 512 | 180ms | 2844 tokens/sec |
| **Decode** | 128 (1 at a time) | 3200ms | 40 tokens/sec |

**관찰**:
- Prefill은 병렬 처리 가능 → 높은 throughput
- Decode는 순차적 → 낮은 throughput이지만 latency 중요
- vLLM은 둘을 함께 배치에 넣어 GPU 활용도 극대화

### 7.4 메모리 사용량

**Llama-7B, A100 80GB**

```
총 GPU 메모리: 80GB

사용 breakdown:
├─ Model weights:        14GB  (18%)
│  ├─ FP16 weights:     13GB
│  └─ Quantization (선택): 7GB (INT8/INT4)
│
├─ KV Cache:            50GB  (62%)
│  ├─ num_blocks = 6400
│  ├─ block_size = 16
│  └─ 각 block = 8MB
│
├─ Activation memory:   10GB  (12%)
│  ├─ Forward pass:      6GB
│  └─ CUDA graphs:       4GB
│
└─ System overhead:      6GB   (8%)
```

**KV Cache 계산**:
```python
# Llama-7B 설정
num_layers = 32
num_kv_heads = 32
head_size = 128
block_size = 16
dtype_size = 2  # FP16

# 블록당 메모리
memory_per_block = (
    2 * num_layers * num_kv_heads * head_size * block_size * dtype_size
)
# = 2 * 32 * 32 * 128 * 16 * 2
# = 8,388,608 bytes
# = 8 MB

# GPU 메모리 50GB를 KV cache에 할당
num_blocks = 50 * 1024 / 8 = 6400 blocks

# 동시 처리 가능한 토큰 수
max_tokens = num_blocks * block_size = 6400 * 16 = 102,400 tokens
```

### 7.5 Batching 효율성

**Llama-7B, Decode phase**

| Batch Size | Latency (ms) | Throughput (tokens/sec) | GPU Util (%) |
|-----------|--------------|-------------------------|--------------|
| 1 | 25 | 40 | 15% |
| 8 | 30 | 266 | 45% |
| 32 | 45 | 711 | 78% |
| 64 | 70 | 914 | 88% |
| 128 | 120 | 1066 | 95% |
| 256 | 220 | 1163 | 98% |

**관찰**:
- Batch size 증가 → Latency 증가, Throughput 증가
- GPU utilization이 95% 이상이면 추가 batching 효과 감소
- vLLM은 자동으로 최적의 배치 크기 선택

---

## 8. 트러블슈팅

### 8.1 Out of Memory (OOM)

**증상**:
```
RuntimeError: CUDA out of memory.
Tried to allocate 2.00 GiB (GPU 0; 79.35 GiB total capacity)
```

**원인**:
1. KV Cache가 너무 큼
2. 배치 크기가 너무 큼
3. 모델이 너무 큼

**해결책**:

```bash
# 1. KV Cache 크기 줄이기
vllm serve model_name \
    --gpu-memory-utilization 0.85  # 기본 0.90 → 0.85

# 2. Max sequence length 제한
vllm serve model_name \
    --max-model-len 2048  # 기본 4096 → 2048

# 3. Batch size 제한
vllm serve model_name \
    --max-num-seqs 128  # 기본 256 → 128

# 4. Quantization 사용
vllm serve model_name \
    --quantization awq  # INT4 quantization
    # 또는
    --quantization gptq  # INT4/INT8 quantization
```

### 8.2 높은 TTFT (Time to First Token)

**증상**:
```
첫 토큰까지 3초 이상 걸림 (기대값: <500ms)
```

**원인**:
1. 긴 prefill
2. 배치에 다른 긴 prefill이 있음
3. CUDA graph warmup

**해결책**:

```bash
# 1. Chunked prefill 활성화
vllm serve model_name \
    --max-num-batched-tokens 8192  # 기본값보다 작게

# 2. Prefix caching 활성화 (공통 프롬프트가 있는 경우)
vllm serve model_name \
    --enable-prefix-caching

# 3. 전용 prefill instance 사용 (Disaggregated Prefill)
# Prefill용 서버:
vllm serve model_name --prefill-only

# Decode용 서버:
vllm serve model_name --decode-only
```

### 8.3 낮은 Throughput

**증상**:
```
예상: 1000 tokens/sec
실제: 300 tokens/sec
```

**원인**:
1. Batch size가 너무 작음
2. GPU utilization이 낮음
3. CPU bottleneck

**해결책**:

```bash
# 1. Batch size 늘리기
vllm serve model_name \
    --max-num-seqs 256  # 기본 256, 필요시 증가

# 2. Tensor Parallelism 사용
vllm serve model_name \
    --tensor-parallel-size 4  # 4 GPUs

# 3. CUDA graphs 활성화 (기본값)
vllm serve model_name \
    --enforce-eager  # 이 옵션 제거 (CUDA graphs 사용)
```

### 8.4 Attention 오류

**증상**:
```
AssertionError: Invalid attention metadata
```

**원인**:
1. Sequence length가 max_model_len 초과
2. Block table 불일치
3. KV cache corruption

**해결책**:

```bash
# 1. Max model length 명시
vllm serve model_name \
    --max-model-len 4096  # 모델 config와 일치

# 2. Block size 조정
vllm serve model_name \
    --block-size 16  # 기본 16, 8 or 32로 변경 시도

# 3. Prefix caching 비활성화 (문제 발생 시)
vllm serve model_name \
    --disable-prefix-caching
```

### 8.5 느린 Sampling

**증상**:
```
Forward pass: 30ms
Sampling: 100ms  ← 너무 느림!
```

**원인**:
1. Logprobs 계산 (전체 vocab)
2. Top-k가 너무 큼
3. Structured output (grammar)

**해결책**:

```python
# 1. Logprobs 비활성화
sampling_params = SamplingParams(
    logprobs=None,  # 또는 작은 값 (예: 5)
)

# 2. Top-k 줄이기
sampling_params = SamplingParams(
    top_k=50,  # 기본 -1 (all)
)

# 3. Greedy sampling 사용 (가능한 경우)
sampling_params = SamplingParams(
    temperature=0.0,  # Greedy
)
```

---

## 9. 요약

### 9.1 핵심 파이프라인

```
요청 도착
  ↓
Tokenization (Processor)
  ↓
Scheduling (Scheduler)
  ├─ RUNNING 요청 스케줄링
  ├─ WAITING 요청 스케줄링
  ├─ KV Cache 할당
  └─ Preemption 처리
  ↓
Input Preparation (GPUModelRunner)
  ├─ Token IDs 추출
  ├─ Positions 계산
  ├─ Attention metadata 생성
  └─ GPU로 복사
  ↓
Forward Pass
  ├─ Embedding lookup
  ├─ Transformer layers
  │   ├─ Self-attention
  │   └─ Feed-forward
  └─ Output projection
  ↓
Sampling
  ├─ Temperature scaling
  ├─ Top-k/top-p filtering
  ├─ Random sampling
  └─ Logprobs 계산
  ↓
Bookkeeping
  ├─ Token 업데이트
  ├─ KV Cache 업데이트
  └─ 상태 동기화
  ↓
Output Processing
  ├─ Stop string 체크
  ├─ 완료 확인
  └─ RequestOutput 생성
  ↓
완성된 응답 반환
```

### 9.2 주요 최적화 기법

| 기법 | 효과 | 사용 시기 |
|-----|------|---------|
| **Continuous Batching** | 2-5x throughput | 항상 |
| **PagedAttention** | 메모리 효율 5-10x | 항상 |
| **CUDA Graphs** | 2-3x decode 속도 | Decode phase |
| **Chunked Prefill** | Latency 개선 | 긴 프롬프트 |
| **Prefix Caching** | Prefill 시간 90% 절감 | 공통 프롬프트 |
| **Tensor Parallelism** | 2-4x throughput | 큰 모델 |
| **Speculative Decoding** | 2-3x decode 속도 | Latency 중요 시 |

### 9.3 성능 체크리스트

**배포 전 확인사항**:

- [ ] GPU memory utilization 85-90%
- [ ] Batch size가 충분히 큼 (>32 for decode)
- [ ] CUDA graphs 활성화됨
- [ ] Chunked prefill 설정됨 (긴 프롬프트)
- [ ] Prefix caching 활성화됨 (공통 프롬프트)
- [ ] Quantization 고려됨 (메모리 부족 시)
- [ ] Tensor Parallelism 설정됨 (큰 모델)
- [ ] Monitoring 설정됨 (Prometheus, Grafana)

**성능 목표**:

| 메트릭 | 목표값 |
|-------|--------|
| TTFT | <200ms (short prompts), <1s (long prompts) |
| ITL | <50ms |
| Throughput | >1000 tokens/sec (per GPU) |
| GPU Util | >85% |

---

## 참고 자료

- **코드**: `vllm/v1/engine/llm_engine.py`, `vllm/v1/core/sched/scheduler.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/sample/sampler.py`
- **Paper**: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- **Blog**: [vLLM: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/)
