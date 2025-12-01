# PART 7: 실전 LLM 서비스 엔지니어링

> 프로덕션 환경에서 LLM을 안정적이고 효율적으로 운영하는 방법

---

## 목차
1. [OpenAI-compatible API 구현 시 주의점](#1-openai-compatible-api-구현-시-주의점)
2. [Tokenizer Latency와 Batching](#2-tokenizer-latency와-batching)
3. [Prompt Cache 재사용](#3-prompt-cache-재사용)
4. [Context Length와 GPU 비용](#4-context-length와-gpu-비용)
5. [Streaming 응답 최적화](#5-streaming-응답-최적화)
6. [CPU 오프로딩과 KV 압축](#6-cpu-오프로딩과-kv-압축)
7. [로컬 추론 환경 최적화](#7-로컬-추론-환경-최적화)
8. [vLLM vs SGLang vs TensorRT-LLM](#8-vllm-vs-sglang-vs-tensorrt-llm)
9. [Multi-User Fairness](#9-multi-user-fairness)
10. [면접 예상 질문 및 답변](#10-면접-예상-질문-및-답변)

---

## 1. OpenAI-compatible API 구현 시 주의점

### 1.1 API 호환성 요구사항

```python
# OpenAI API 형식

# 요청
{
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": true
}

# 응답 (non-streaming)
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-4",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello! How can I help?"},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30
    }
}
```

### 1.2 구현 시 주의점

```python
# vllm/entrypoints/openai/ 참조

class OpenAIServingChat:
    """OpenAI Chat Completion API 구현"""

    async def create_chat_completion(self, request: ChatCompletionRequest):
        # 1. 입력 검증
        self._validate_request(request)

        # 2. 메시지를 프롬프트로 변환
        # 주의: 모델별 chat template 적용 필요
        prompt = self._apply_chat_template(request.messages)

        # 3. Sampling 파라미터 변환
        # 주의: OpenAI와 vLLM 파라미터 매핑
        sampling_params = self._convert_sampling_params(request)

        # 4. 생성 요청
        results = await self.engine.generate(prompt, sampling_params)

        # 5. 응답 포맷팅
        # 주의: 정확한 token count, finish_reason
        return self._format_response(results)
```

### 1.3 Chat Template 처리

```python
# 모델별 다른 chat template

# LLaMA 2 Chat
LLAMA2_TEMPLATE = """<s>[INST] <<SYS>>
{system_message}
<</SYS>>

{user_message} [/INST]"""

# ChatML (Mistral, many others)
CHATML_TEMPLATE = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

def apply_chat_template(messages, model_type):
    """모델에 맞는 template 적용"""
    if model_type == "llama2":
        return format_llama2(messages)
    elif model_type == "chatml":
        return format_chatml(messages)
    else:
        # Hugging Face tokenizer의 apply_chat_template 사용
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
```

### 1.4 주의해야 할 차이점

```
┌──────────────────────────────────────────────────────────────────┐
│  OpenAI API 특수 케이스                                         │
├──────────────────────────────────────────────────────────────────┤
│  1. function_call / tools:                                      │
│     - JSON schema 기반 출력 강제                                │
│     - vLLM: guided decoding 또는 outlines 통합 필요            │
│                                                                  │
│  2. logprobs:                                                    │
│     - top_logprobs 파라미터                                     │
│     - 응답에 token별 확률 포함                                  │
│                                                                  │
│  3. stop sequences:                                              │
│     - 여러 stop string 지원                                     │
│     - 부분 매칭 처리 주의                                       │
│                                                                  │
│  4. n > 1:                                                       │
│     - 동일 프롬프트에 여러 completion                          │
│     - Beam search와 다름 (독립 샘플링)                          │
│                                                                  │
│  5. presence_penalty, frequency_penalty:                        │
│     - 반복 억제 파라미터                                        │
│     - 정확한 구현 필요                                          │
└──────────────────────────────────────────────────────────────────┘
```

### 1.5 Token Counting 정확성

```python
# Token count는 billing과 직결!

def count_tokens_accurately(text, tokenizer):
    """정확한 토큰 카운트"""

    # 주의: special tokens 처리
    tokens = tokenizer.encode(
        text,
        add_special_tokens=True  # BOS, EOS 포함
    )

    return len(tokens)

# 응답에서 usage 정보
{
    "usage": {
        "prompt_tokens": count_tokens(prompt),
        "completion_tokens": count_tokens(completion),
        "total_tokens": prompt + completion
    }
}

# 주의사항:
# - Chat template에 추가되는 토큰 포함
# - Special tokens (BOS, EOS) 포함 여부 일관성
# - Multi-turn에서 누적 계산
```

---

## 2. Tokenizer Latency와 Batching

### 2.1 Tokenizer 병목

```
요청 처리 파이프라인:
┌─────────────────────────────────────────────────────────────────┐
│  HTTP 수신 → Tokenize → Scheduler → GPU → Detokenize → 응답   │
│              ~1-10ms    ~0.1ms    ~10ms+  ~1-5ms              │
│              ↑ 병목 가능!                                      │
└─────────────────────────────────────────────────────────────────┘

긴 프롬프트에서:
- 10K tokens 프롬프트
- Python tokenizer: ~50ms
- 총 latency의 5-10% 차지
```

### 2.2 Tokenizer 최적화

```python
# 1. Rust tokenizer 사용 (tokenizers 라이브러리)
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-2-7b")
# Rust 구현으로 10배 빠름

# 2. 배치 tokenization
def batch_tokenize(texts: List[str]) -> List[List[int]]:
    # 개별 처리 대신 배치로
    return tokenizer.encode_batch(texts)

# 3. 비동기 tokenization
async def async_tokenize(text: str) -> List[int]:
    # CPU-bound 작업을 thread pool에서 실행
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        thread_pool,
        tokenizer.encode,
        text
    )
```

### 2.3 Detokenization 최적화

```python
# Streaming에서 detokenization 병목

# 나쁜 예: 매 토큰마다 전체 decode
def stream_bad(token_ids):
    for i, tok in enumerate(token_ids):
        text = tokenizer.decode(token_ids[:i+1])  # 전체 decode!
        yield text[-1]  # 마지막 문자만

# 좋은 예: Incremental decode
class IncrementalDetokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_buffer = []
        self.text_offset = 0

    def add_token(self, token_id: int) -> str:
        self.token_buffer.append(token_id)
        # 새 토큰만 decode
        full_text = self.tokenizer.decode(self.token_buffer)
        new_text = full_text[self.text_offset:]
        self.text_offset = len(full_text)
        return new_text
```

### 2.4 Tokenizer Batching 전략

```python
# Request-level batching

class TokenizerBatcher:
    def __init__(self, batch_size=32, timeout_ms=5):
        self.batch_size = batch_size
        self.timeout = timeout_ms / 1000
        self.pending = []

    async def tokenize(self, text: str) -> List[int]:
        future = asyncio.Future()
        self.pending.append((text, future))

        if len(self.pending) >= self.batch_size:
            self._process_batch()
        else:
            # timeout 후 처리
            asyncio.get_event_loop().call_later(
                self.timeout, self._process_batch
            )

        return await future

    def _process_batch(self):
        if not self.pending:
            return

        texts = [p[0] for p in self.pending]
        futures = [p[1] for p in self.pending]

        # 배치 tokenization
        results = self.tokenizer.encode_batch(texts)

        for future, result in zip(futures, results):
            future.set_result(result)

        self.pending.clear()
```

---

## 3. Prompt Cache 재사용

### 3.1 Prompt Caching 개념

```
┌──────────────────────────────────────────────────────────────────┐
│  시나리오: 동일한 system prompt 반복 사용                       │
│                                                                  │
│  Request 1: "You are a helpful assistant." + "What is AI?"     │
│  Request 2: "You are a helpful assistant." + "Tell me a joke"  │
│  Request 3: "You are a helpful assistant." + "Write code"      │
│             ↑ 공통 prefix ↑                                     │
│                                                                  │
│  Without caching:                                               │
│  - 매번 system prompt prefill 반복                             │
│  - 3 × prefill_time(30 tokens)                                 │
│                                                                  │
│  With caching:                                                  │
│  - 첫 번째만 prefill, 나머지는 cache hit                       │
│  - 1 × prefill_time + 2 × cache_lookup                         │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 vLLM의 Prefix Caching

```python
# vLLM 설정
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-chat",
    enable_prefix_caching=True,  # Prefix caching 활성화
)

# 자동으로 공통 prefix의 KV cache 재사용
```

### 3.3 애플리케이션 레벨 캐싱

```python
# System prompt 캐싱 전략

class PromptCacheManager:
    def __init__(self, llm_client):
        self.client = llm_client
        self.prefix_cache = {}  # hash → prefix_id

    async def generate_with_cache(
        self,
        system_prompt: str,
        user_message: str
    ):
        # System prompt hash
        prefix_hash = hashlib.md5(system_prompt.encode()).hexdigest()

        # Cache hit 확인
        if prefix_hash in self.prefix_cache:
            # Cached prefix 사용
            return await self.client.generate(
                prompt=user_message,
                prefix_id=self.prefix_cache[prefix_hash]
            )
        else:
            # Full generation 후 cache
            result = await self.client.generate(
                prompt=system_prompt + user_message
            )
            self.prefix_cache[prefix_hash] = result.prefix_id
            return result
```

### 3.4 Multi-turn 대화 캐싱

```python
# 대화 히스토리 캐싱

class ConversationCache:
    def __init__(self):
        self.conversations = {}  # session_id → kv_cache_blocks

    async def continue_conversation(
        self,
        session_id: str,
        new_message: str
    ):
        if session_id in self.conversations:
            # 이전 대화의 KV cache 재사용
            cached_blocks = self.conversations[session_id]
            result = await generate_with_kv_cache(
                new_message,
                kv_cache=cached_blocks
            )
            # 새로운 KV cache 저장
            self.conversations[session_id] = result.kv_cache
        else:
            # 새 대화 시작
            result = await generate(new_message)
            self.conversations[session_id] = result.kv_cache

        return result

    def cleanup_old_conversations(self, max_age_seconds=3600):
        """오래된 대화 캐시 정리"""
        # Memory pressure 시 LRU eviction
        pass
```

---

## 4. Context Length와 GPU 비용

### 4.1 Context Length의 영향

```
┌──────────────────────────────────────────────────────────────────┐
│  Context Length 증가 시:                                        │
│                                                                  │
│  1. KV Cache 메모리: O(L)                                       │
│     - L=2K: 0.5GB per request (LLaMA-7B)                        │
│     - L=8K: 2GB per request                                     │
│     - L=32K: 8GB per request                                    │
│                                                                  │
│  2. Prefill 연산: O(L²)                                         │
│     - L=2K: ~100ms                                              │
│     - L=8K: ~400ms (4배 길이, 16배 연산)                       │
│     - L=32K: ~6.4초                                             │
│                                                                  │
│  3. Decode 연산: O(L)                                           │
│     - L=2K: ~20ms/token                                         │
│     - L=8K: ~80ms/token                                         │
│     - L=32K: ~320ms/token                                       │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 비용 모델

```python
def estimate_gpu_cost(
    context_length: int,
    output_length: int,
    model_size: str = "7B",
    gpu_type: str = "A100"
):
    """GPU 시간 기반 비용 추정"""

    # 모델별 기본 latency (ms)
    base_prefill = {
        "7B": 0.05,   # ms per token²
        "13B": 0.08,
        "70B": 0.25
    }

    base_decode = {
        "7B": 0.01,   # ms per context token
        "13B": 0.015,
        "70B": 0.04
    }

    # Prefill 시간 (O(L²))
    prefill_time = base_prefill[model_size] * (context_length ** 2)

    # Decode 시간 (O(L) per token)
    avg_context = context_length + output_length / 2
    decode_time = base_decode[model_size] * avg_context * output_length

    total_ms = prefill_time + decode_time

    # GPU 시간당 비용 (예: A100 $2/hour)
    gpu_cost_per_hour = {"A100": 2.0, "H100": 3.5, "4090": 0.5}
    cost = (total_ms / 1000 / 3600) * gpu_cost_per_hour[gpu_type]

    return {
        "prefill_ms": prefill_time,
        "decode_ms": decode_time,
        "total_ms": total_ms,
        "estimated_cost_usd": cost
    }

# 예시
print(estimate_gpu_cost(8000, 500, "7B", "A100"))
# {'prefill_ms': 3200, 'decode_ms': 41250, 'total_ms': 44450, 'cost': 0.025}
```

### 4.3 Context Length 최적화 전략

```python
# 1. 동적 context truncation
def smart_truncate(messages, max_context=4096, tokenizer=None):
    """중요도 기반 truncation"""

    # System prompt는 항상 유지
    system = messages[0] if messages[0]["role"] == "system" else None

    # 최근 메시지 우선
    recent_messages = messages[-10:]  # 최근 10개

    # 토큰 수 확인하며 추가
    total_tokens = 0
    kept_messages = []

    for msg in reversed(recent_messages):
        msg_tokens = len(tokenizer.encode(msg["content"]))
        if total_tokens + msg_tokens < max_context:
            kept_messages.insert(0, msg)
            total_tokens += msg_tokens
        else:
            break

    if system:
        kept_messages.insert(0, system)

    return kept_messages

# 2. 요약 기반 압축
async def compress_history(messages, summarizer):
    """오래된 대화를 요약으로 압축"""

    if len(messages) < 20:
        return messages

    # 오래된 메시지 요약
    old_messages = messages[1:-10]  # system 제외, 최근 10개 제외
    summary = await summarizer.summarize(old_messages)

    # 압축된 히스토리
    return [
        messages[0],  # system
        {"role": "system", "content": f"Previous conversation summary: {summary}"},
        *messages[-10:]  # 최근 메시지
    ]
```

### 4.4 Context 길이별 배치 전략

```python
# 길이별 다른 처리

class AdaptiveScheduler:
    def __init__(self):
        self.short_queue = []   # < 1K tokens
        self.medium_queue = []  # 1K - 4K tokens
        self.long_queue = []    # > 4K tokens

    def add_request(self, request):
        context_len = len(request.prompt_tokens)

        if context_len < 1000:
            self.short_queue.append(request)
        elif context_len < 4000:
            self.medium_queue.append(request)
        else:
            self.long_queue.append(request)

    def get_batch(self, max_tokens=4096):
        """균형 잡힌 배치 구성"""

        batch = []
        total_tokens = 0

        # 짧은 요청 많이 + 긴 요청 조금
        for queue in [self.short_queue, self.medium_queue, self.long_queue]:
            while queue and total_tokens < max_tokens:
                req = queue[0]
                if total_tokens + req.context_len <= max_tokens:
                    batch.append(queue.pop(0))
                    total_tokens += req.context_len
                else:
                    break

        return batch
```

---

## 5. Streaming 응답 최적화

### 5.1 Streaming 기본 구조

```python
# Server-Sent Events (SSE) 기반 streaming

async def stream_completion(request):
    """SSE 기반 스트리밍 응답"""

    async def event_generator():
        async for token in generate_tokens(request):
            # SSE 형식
            data = {
                "id": request.id,
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"

        # 종료 이벤트
        yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

### 5.2 Chunk Size 최적화

```python
# 토큰 단위 vs 청크 단위

# 1. 토큰 단위 (기본)
# 장점: 최소 latency
# 단점: 네트워크 오버헤드, detokenization 이슈

# 2. Word/Subword 단위
class WordChunker:
    def __init__(self):
        self.buffer = ""

    def add_token(self, token_text: str) -> Optional[str]:
        self.buffer += token_text

        # 단어 경계에서 flush
        if self.buffer.endswith((' ', '\n', '.', ',', '!', '?')):
            result = self.buffer
            self.buffer = ""
            return result
        return None

# 3. 시간 기반 청크
class TimeBasedChunker:
    def __init__(self, interval_ms=50):
        self.interval = interval_ms / 1000
        self.buffer = ""
        self.last_flush = time.time()

    def add_token(self, token_text: str) -> Optional[str]:
        self.buffer += token_text

        if time.time() - self.last_flush >= self.interval:
            result = self.buffer
            self.buffer = ""
            self.last_flush = time.time()
            return result
        return None
```

### 5.3 Backpressure 처리

```python
# 클라이언트가 느릴 때 처리

class BackpressureHandler:
    def __init__(self, max_buffer_size=100):
        self.buffer = asyncio.Queue(maxsize=max_buffer_size)
        self.dropped_count = 0

    async def produce(self, token: str):
        try:
            self.buffer.put_nowait(token)
        except asyncio.QueueFull:
            # 버퍼 가득 참 - 오래된 토큰 drop 또는 대기
            self.dropped_count += 1
            # 옵션 1: 대기
            await self.buffer.put(token)
            # 옵션 2: drop (실시간성 우선)
            # pass

    async def consume(self):
        while True:
            token = await self.buffer.get()
            yield token

    async def stream_with_backpressure(self, generator):
        """생성과 전송 분리"""

        async def producer():
            async for token in generator:
                await self.produce(token)

        async def consumer():
            async for token in self.consume():
                yield f"data: {token}\n\n"

        # 병렬 실행
        producer_task = asyncio.create_task(producer())

        async for chunk in consumer():
            yield chunk
            if producer_task.done():
                break
```

### 5.4 네트워크 최적화

```python
# HTTP/2 또는 WebSocket 사용

# 1. HTTP/2 Server Push (SSE)
# - 단일 연결로 여러 요청
# - 헤더 압축

# 2. WebSocket
class WebSocketStreamer:
    async def stream_ws(self, websocket, request):
        async for token in generate_tokens(request):
            await websocket.send_json({
                "type": "token",
                "content": token
            })

        await websocket.send_json({
            "type": "done"
        })

# WebSocket 장점:
# - 양방향 통신 (취소 쉬움)
# - 낮은 오버헤드
# - 재연결 로직 필요

# 3. gRPC Streaming
# - 바이너리 프로토콜
# - 더 효율적
# - 클라이언트 지원 제한적
```

---

## 6. CPU 오프로딩과 KV 압축

### 6.1 CPU Offloading

```python
# KV Cache를 CPU로 이동

class CPUOffloadManager:
    def __init__(self, gpu_budget_gb=20, cpu_budget_gb=100):
        self.gpu_budget = gpu_budget_gb * 1e9
        self.cpu_budget = cpu_budget_gb * 1e9
        self.gpu_usage = 0
        self.cpu_usage = 0

        # Pinned memory pool
        self.cpu_cache = torch.empty(
            int(cpu_budget_gb * 1e9 / 2),  # FP16
            dtype=torch.float16,
            pin_memory=True
        )

    def should_offload(self, request) -> bool:
        """GPU 메모리 pressure 확인"""
        return self.gpu_usage > self.gpu_budget * 0.9

    async def offload_kv_cache(self, request):
        """GPU → CPU 비동기 전송"""
        kv_blocks = request.kv_cache_blocks

        # 비동기 복사
        cpu_blocks = self.allocate_cpu_blocks(len(kv_blocks))
        for gpu_block, cpu_block in zip(kv_blocks, cpu_blocks):
            cpu_block.copy_(gpu_block, non_blocking=True)

        # GPU 블록 해제
        self.free_gpu_blocks(kv_blocks)
        request.kv_cache_location = "cpu"
        request.cpu_blocks = cpu_blocks

    async def reload_kv_cache(self, request):
        """CPU → GPU 전송"""
        if request.kv_cache_location != "cpu":
            return

        gpu_blocks = self.allocate_gpu_blocks(len(request.cpu_blocks))
        for cpu_block, gpu_block in zip(request.cpu_blocks, gpu_blocks):
            gpu_block.copy_(cpu_block, non_blocking=True)

        request.kv_cache_blocks = gpu_blocks
        request.kv_cache_location = "gpu"
```

### 6.2 KV Cache Quantization

```python
# KV Cache 압축

# 1. FP16 → INT8 Quantization
def quantize_kv_cache(kv_cache_fp16):
    """Per-channel INT8 quantization"""
    # 채널별 scale 계산
    scale = kv_cache_fp16.abs().max(dim=-1, keepdim=True)[0] / 127

    # Quantize
    kv_cache_int8 = (kv_cache_fp16 / scale).round().clamp(-128, 127).to(torch.int8)

    return kv_cache_int8, scale

def dequantize_kv_cache(kv_cache_int8, scale):
    """Dequantization"""
    return kv_cache_int8.float() * scale

# 메모리 절약: 50%
# 품질 영향: 거의 없음 (attention은 상대적 크기가 중요)

# 2. FP16 → FP8 (H100)
# NVIDIA FP8 형식 사용
# 하드웨어 가속 지원
```

### 6.3 Sliding Window KV Compression

```python
# 오래된 KV 압축/삭제

class SlidingWindowKVManager:
    def __init__(self, window_size=4096, compression_ratio=4):
        self.window_size = window_size
        self.compression_ratio = compression_ratio

    def manage_kv_cache(self, kv_cache, current_position):
        """
        Recent: Full resolution
        Old: Compressed or dropped
        """
        if current_position <= self.window_size:
            return kv_cache

        # 최근 window_size는 유지
        recent_kv = kv_cache[:, -self.window_size:, :, :]

        # 오래된 부분 압축
        old_kv = kv_cache[:, :-self.window_size, :, :]

        # 압축: 평균 풀링 또는 중요 토큰만 유지
        compressed_old = self.compress_kv(old_kv)

        # 결합
        return torch.cat([compressed_old, recent_kv], dim=1)

    def compress_kv(self, kv):
        """KV cache 압축"""
        # 방법 1: Average pooling
        pooled = F.avg_pool1d(
            kv.transpose(1, 2),
            kernel_size=self.compression_ratio
        ).transpose(1, 2)

        # 방법 2: Attention score 기반 선택 (H2O)
        # 중요한 토큰만 유지

        return pooled
```

### 6.4 H2O (Heavy-Hitter Oracle)

```python
# Attention score 기반 KV 압축

class H2OKVCache:
    """
    Heavy-Hitter Oracle:
    Attention을 많이 받는 토큰의 KV만 유지
    """

    def __init__(self, budget_ratio=0.2):
        self.budget_ratio = budget_ratio
        self.attention_scores = []

    def update_scores(self, attention_weights):
        """각 토큰이 받은 attention 누적"""
        # attention_weights: [batch, heads, query_len, key_len]
        scores = attention_weights.sum(dim=(0, 1, 2))  # key 위치별 총합
        self.attention_scores.append(scores)

    def select_heavy_hitters(self, kv_cache, current_len):
        """중요 토큰 선택"""
        budget = int(current_len * self.budget_ratio)

        # 누적 attention score
        total_scores = torch.stack(self.attention_scores).sum(dim=0)

        # Top-k 선택
        _, indices = total_scores.topk(budget)
        indices = indices.sort()[0]  # 순서 유지

        # 선택된 KV만 유지
        compressed_kv = kv_cache[:, indices, :, :]

        return compressed_kv, indices
```

---

## 7. 로컬 추론 환경 최적화

### 7.1 4090/3090 환경 최적화

```python
# Consumer GPU 최적화

# RTX 4090: 24GB VRAM, 82 TFLOPS FP16
# RTX 3090: 24GB VRAM, 35 TFLOPS FP16

def optimize_for_consumer_gpu(model_size="7B", gpu="4090"):
    config = {
        "gpu_memory_utilization": 0.90,  # 더 보수적으로
        "max_model_len": 4096,           # 컨텍스트 제한

        # Quantization
        "quantization": "awq",  # 또는 "gptq"

        # Batch size
        "max_num_seqs": 8 if gpu == "4090" else 4,
    }

    # 7B 모델 기준
    if model_size == "7B":
        # FP16: 14GB → 24GB에 fit
        # INT4: 3.5GB → 여유 있음
        config["quantization"] = None  # FP16 가능
        config["max_num_seqs"] = 16

    elif model_size == "13B":
        # FP16: 26GB → 안 됨!
        # INT4: 6.5GB → fit
        config["quantization"] = "awq"
        config["max_num_seqs"] = 8

    elif model_size == "70B":
        # INT4: 35GB → 안 됨!
        # 이 경우 CPU offload 또는 더 작은 모델 사용
        raise ValueError("70B requires multi-GPU or cloud")

    return config

# vLLM 실행
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    **optimize_for_consumer_gpu("7B", "4090")
)
```

### 7.2 최적 Batch Size 찾기

```python
# 메모리 기반 배치 크기 계산

def calculate_optimal_batch_size(
    gpu_memory_gb: float,
    model_memory_gb: float,
    kv_cache_per_token_mb: float,
    avg_seq_length: int,
    max_seq_length: int
):
    """
    가용 메모리 기반 최적 배치 크기 계산
    """
    # 가용 KV cache 메모리
    available_memory = (gpu_memory_gb - model_memory_gb - 2) * 1024  # MB
    # 2GB는 activations, CUDA context 등

    # 요청당 평균 KV cache 메모리
    avg_kv_per_request = kv_cache_per_token_mb * (avg_seq_length + max_seq_length) / 2

    # 최대 배치 크기
    max_batch = int(available_memory / avg_kv_per_request)

    # 실용적 상한
    return min(max_batch, 64)

# 예시: 4090 + LLaMA-7B
optimal_batch = calculate_optimal_batch_size(
    gpu_memory_gb=24,
    model_memory_gb=14,  # FP16
    kv_cache_per_token_mb=0.5,  # 7B 기준
    avg_seq_length=500,
    max_seq_length=2048
)
print(f"Optimal batch size: {optimal_batch}")  # ~16
```

### 7.3 llama.cpp vs vLLM

```
┌──────────────────────────────────────────────────────────────────┐
│  특성              │ llama.cpp         │ vLLM                   │
├──────────────────────────────────────────────────────────────────┤
│  대상 환경         │ 로컬, 엣지        │ 서버                   │
│  언어              │ C++               │ Python + CUDA          │
│  CPU 추론          │ ✓ (주요 기능)     │ ✗                      │
│  Quantization      │ 다양 (Q4, Q5...)  │ AWQ, GPTQ              │
│  Batching          │ 제한적            │ Continuous batching    │
│  Throughput        │ 낮음              │ 높음                   │
│  메모리 효율       │ 중간              │ 높음 (PagedAttention)  │
│  설치 용이성       │ ✓ (단일 바이너리) │ 복잡 (CUDA 필요)       │
│  Apple Silicon     │ ✓ (Metal)         │ ✗                      │
└──────────────────────────────────────────────────────────────────┘

선택 가이드:
- 개인 사용, CPU 추론: llama.cpp
- 서버 배포, 높은 throughput: vLLM
- Mac 사용: llama.cpp 또는 MLX
```

---

## 8. vLLM vs SGLang vs TensorRT-LLM

### 8.1 기능 비교

```
┌──────────────────────────────────────────────────────────────────┐
│  Feature           │ vLLM   │ SGLang │ TensorRT-LLM            │
├──────────────────────────────────────────────────────────────────┤
│  PagedAttention    │ ✓      │ ✓      │ ✓ (Paged KV Cache)      │
│  Continuous Batch  │ ✓      │ ✓      │ ✓                       │
│  Tensor Parallel   │ ✓      │ ✓      │ ✓                       │
│  Pipeline Parallel │ ✓      │ ✗      │ ✓                       │
│  Speculative       │ ✓      │ ✓      │ ✓                       │
│  FlashAttention    │ ✓      │ ✓      │ ✓ (custom)              │
│  Quantization      │ AWQ,   │ AWQ    │ FP8, INT8, INT4         │
│                    │ GPTQ   │        │ (최적화됨)              │
│  Structured Output │ ✓      │ ✓✓     │ 제한적                  │
│  Multi-modal       │ ✓      │ ✓      │ 제한적                  │
│  RadixAttention    │ ✗      │ ✓      │ ✗                       │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 성능 비교

```
벤치마크 (LLaMA-2-70B, 8×A100):

Throughput (tokens/sec):
┌──────────────────────────────────────────────────────────────────┐
│  Workload          │ vLLM   │ SGLang │ TensorRT-LLM            │
├──────────────────────────────────────────────────────────────────┤
│  Short (128 in)    │ 2500   │ 2800   │ 3200                    │
│  Medium (512 in)   │ 2000   │ 2300   │ 2800                    │
│  Long (2048 in)    │ 1200   │ 1400   │ 1800                    │
└──────────────────────────────────────────────────────────────────┘

Latency P50 (ms):
┌──────────────────────────────────────────────────────────────────┐
│  Metric            │ vLLM   │ SGLang │ TensorRT-LLM            │
├──────────────────────────────────────────────────────────────────┤
│  TTFT (Time to     │ 50     │ 45     │ 35                      │
│  First Token)      │        │        │                         │
│  Inter-token       │ 25     │ 22     │ 18                      │
│  latency           │        │        │                         │
└──────────────────────────────────────────────────────────────────┘

TensorRT-LLM > SGLang > vLLM (raw 성능)
하지만 사용 편의성은 반대
```

### 8.3 사용 편의성

```
배포 복잡도:

vLLM:
pip install vllm
vllm serve meta-llama/Llama-2-7b-chat-hf
# 끝! 5분 내 시작 가능

SGLang:
pip install sglang
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf
# 비교적 간단

TensorRT-LLM:
# 1. Docker 환경 설정
# 2. 모델 변환 (trtllm-build)
# 3. 엔진 최적화
# 4. 서버 배포
# 전체 과정 수 시간 ~ 하루
```

### 8.4 선택 가이드

```
┌──────────────────────────────────────────────────────────────────┐
│  상황                          │ 추천                           │
├──────────────────────────────────────────────────────────────────┤
│  빠른 프로토타이핑             │ vLLM                          │
│  프로덕션, 최대 성능 필요      │ TensorRT-LLM                  │
│  복잡한 프롬프트 프로그래밍    │ SGLang                        │
│  OpenAI 호환 API 필요          │ vLLM                          │
│  NVIDIA 전용 환경              │ TensorRT-LLM                  │
│  오픈소스 선호                 │ vLLM 또는 SGLang              │
│  Multi-modal                   │ vLLM                          │
│  트리 구조 생성 (ToT 등)       │ SGLang                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. Multi-User Fairness

### 9.1 공정성 문제

```
시나리오:
- User A: 초당 100 요청 (대량 배치 작업)
- User B: 초당 1 요청 (대화형 사용)

FCFS 스케줄링 시:
User A의 요청이 큐를 점령
→ User B의 latency 10배 증가
→ User B 경험 악화
```

### 9.2 Rate Limiting

```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 10000
    ):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.user_requests = defaultdict(list)
        self.user_tokens = defaultdict(list)

    def check_limit(self, user_id: str, num_tokens: int) -> tuple[bool, str]:
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        self.user_requests[user_id] = [
            t for t in self.user_requests[user_id] if t > minute_ago
        ]
        self.user_tokens[user_id] = [
            (t, n) for t, n in self.user_tokens[user_id] if t > minute_ago
        ]

        # Check RPM
        if len(self.user_requests[user_id]) >= self.rpm:
            return False, f"Rate limit: {self.rpm} requests/min exceeded"

        # Check TPM
        total_tokens = sum(n for _, n in self.user_tokens[user_id])
        if total_tokens + num_tokens > self.tpm:
            return False, f"Rate limit: {self.tpm} tokens/min exceeded"

        # Record request
        self.user_requests[user_id].append(now)
        self.user_tokens[user_id].append((now, num_tokens))

        return True, ""
```

### 9.3 Weighted Fair Queuing

```python
class WeightedFairScheduler:
    """
    사용자별 가중치에 따른 공정한 스케줄링
    """

    def __init__(self):
        self.user_queues = defaultdict(list)
        self.user_weights = {}  # user_id → weight
        self.user_virtual_time = defaultdict(float)

    def set_user_weight(self, user_id: str, weight: float):
        """높은 weight = 더 많은 리소스"""
        self.user_weights[user_id] = weight

    def add_request(self, user_id: str, request):
        # Virtual time 계산
        weight = self.user_weights.get(user_id, 1.0)
        cost = self.estimate_cost(request)

        # Finish time = start_time + cost/weight
        start_time = max(
            self.user_virtual_time[user_id],
            self.global_virtual_time
        )
        finish_time = start_time + cost / weight

        request.finish_time = finish_time
        self.user_virtual_time[user_id] = finish_time

        heapq.heappush(
            self.user_queues[user_id],
            (finish_time, request)
        )

    def get_next_request(self):
        """가장 낮은 finish_time을 가진 요청 반환"""
        candidates = []

        for user_id, queue in self.user_queues.items():
            if queue:
                candidates.append((queue[0][0], user_id))

        if not candidates:
            return None

        # 가장 낮은 virtual finish time
        _, best_user = min(candidates)
        _, request = heapq.heappop(self.user_queues[best_user])

        return request
```

### 9.4 Priority Tiers

```python
class PriorityTierManager:
    """
    서비스 티어별 우선순위 관리
    """

    TIERS = {
        "enterprise": {"priority": 0, "max_concurrent": 100, "sla_ms": 100},
        "pro": {"priority": 1, "max_concurrent": 20, "sla_ms": 500},
        "free": {"priority": 2, "max_concurrent": 5, "sla_ms": 2000},
    }

    def __init__(self):
        self.tier_queues = {tier: [] for tier in self.TIERS}
        self.tier_active = {tier: 0 for tier in self.TIERS}

    def add_request(self, user_tier: str, request):
        tier_config = self.TIERS[user_tier]

        # 동시 요청 제한 확인
        if self.tier_active[user_tier] >= tier_config["max_concurrent"]:
            raise TooManyRequestsError(user_tier)

        request.priority = tier_config["priority"]
        request.sla_deadline = time.time() + tier_config["sla_ms"] / 1000

        heapq.heappush(
            self.tier_queues[user_tier],
            (request.priority, request.arrival_time, request)
        )

    def get_next_batch(self, batch_size: int):
        """우선순위에 따라 배치 구성"""
        batch = []

        for tier in ["enterprise", "pro", "free"]:
            while len(batch) < batch_size and self.tier_queues[tier]:
                _, _, request = heapq.heappop(self.tier_queues[tier])
                batch.append(request)

        return batch
```

---

## 10. 면접 예상 질문 및 답변

### Q1: OpenAI API를 구현할 때 가장 주의해야 할 점은?

**답변:**
1. **Chat Template**: 모델별로 다른 형식 적용 필요
2. **Token Counting**: 정확한 billing을 위해 필수
3. **Streaming**: SSE 형식 준수, 청크 단위 최적화
4. **Stop Sequences**: 부분 매칭 처리
5. **Function Calling**: JSON schema 기반 출력 강제

특히 **chat template**이 잘못되면 모델 성능이 크게 저하됩니다.

### Q2: Context length가 길어지면 비용이 어떻게 증가하나요?

**답변:**
```
Prefill: O(L²) - 2배 길이 → 4배 시간
Decode: O(L) - 2배 길이 → 2배 시간
Memory: O(L) - 2배 길이 → 2배 KV cache

실제 비용:
L=2K: 기준
L=8K: Prefill 16배, Decode 4배, 메모리 4배
L=32K: Prefill 256배, Decode 16배, 메모리 16배
```

최적화: 동적 truncation, 요약 기반 압축, KV cache quantization

### Q3: Streaming 응답에서 Chunk size를 어떻게 결정하나요?

**답변:**
Trade-off:
- **토큰 단위**: 최소 latency, 높은 네트워크 오버헤드
- **단어 단위**: 자연스러운 출력, 중간 latency
- **시간 기반 (50ms)**: 일정한 업데이트 주기

권장: **단어/구문 경계**에서 flush
- 공백, 문장부호에서 버퍼 전송
- 사용자 경험과 효율성 균형

### Q4: Consumer GPU(4090)에서 70B 모델을 돌릴 수 있나요?

**답변:**
단일 4090 (24GB)에서는 **불가능**합니다.

```
70B FP16: 140GB → 안됨
70B INT4: 35GB → 안됨
70B INT4 + offload: 가능하지만 매우 느림
```

대안:
1. **작은 모델**: 7B, 13B 사용 (품질 충분한 경우)
2. **Multi-GPU**: 2×4090 + TP
3. **Cloud**: A100/H100 사용
4. **llama.cpp**: CPU+GPU 혼합 추론 (느림)

### Q5: vLLM, SGLang, TensorRT-LLM 중 어떤 것을 선택해야 하나요?

**답변:**

| 상황 | 선택 |
|------|------|
| 빠른 시작, 간편함 | **vLLM** |
| 최대 성능 필요 | **TensorRT-LLM** |
| 복잡한 프롬프트 로직 | **SGLang** |
| OpenAI 호환 필요 | **vLLM** |
| NVIDIA 환경 | **TensorRT-LLM** |

일반적으로 **vLLM으로 시작**하고, 성능이 중요해지면 TensorRT-LLM 검토.

### Q6: Multi-user 환경에서 공정성을 어떻게 보장하나요?

**답변:**
세 가지 레이어에서 처리:

1. **Rate Limiting**: 사용자별 요청/토큰 제한
2. **Weighted Fair Queuing**: 가중치 기반 리소스 분배
3. **Priority Tiers**: 서비스 등급별 SLA

```python
# 예: Free tier는 Pro tier보다 낮은 우선순위
# 하지만 완전히 starve되지 않도록 보장
```

핵심: **어떤 사용자도 무한 대기하지 않도록** deficit 보상.

---

## 참고 자료

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [SGLang: Efficient LLM Execution](https://github.com/sgl-project/sglang)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- vLLM 소스코드:
  - `vllm/entrypoints/openai/` - OpenAI API 구현
  - `vllm/transformers_utils/tokenizer.py` - Tokenizer
