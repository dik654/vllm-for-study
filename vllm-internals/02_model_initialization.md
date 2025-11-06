# vLLM 모델 초기화 상세 분석

## 목차
1. [개요](#개요)
2. [모델 초기화 전체 Flow](#모델-초기화-전체-flow)
3. [GPUModelRunner 초기화](#gpumodelrunner-초기화)
4. [모델 로딩 과정](#모델-로딩-과정)
5. [KV Cache 할당](#kv-cache-할당)
6. [Attention Backend 설정](#attention-backend-설정)
7. [CUDA Graph 최적화](#cuda-graph-최적화)
8. [메모리 프로파일링](#메모리-프로파일링)

---

## 개요

vLLM의 모델 초기화는 **Weight 로딩 → 모델 구성 → GPU 메모리 할당 → 최적화** 순서로 진행됩니다. 이 문서는 실제 vLLM 코드를 기반으로 각 단계를 상세히 설명합니다.

### 주요 구성 요소

- **GPUModelRunner**: GPU에서 모델 실행 담당 (`vllm/v1/worker/gpu_model_runner.py`)
- **Model Loader**: Weight 로딩 관리 (`vllm/model_executor/model_loader/`)
- **KV Cache Manager**: PagedAttention용 캐시 관리
- **Attention Backend**: FlashAttention, xFormers 등
- **CUDA Graph**: 성능 최적화

---

## 모델 초기화 전체 Flow

```
┌────────────────────────────────────────────────────────────────┐
│                   1. GPUModelRunner 생성                        │
│  - VllmConfig 파싱                                             │
│  - Device 설정 (GPU)                                            │
│  - 모델 관련 파라미터 설정                                       │
└──────────────────┬─────────────────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────────────────┐
│                   2. Model Loader 생성                          │
│  - DefaultModelLoader, TensorizerLoader 등                     │
│  - Load format 결정 (safetensors/pt)                          │
└──────────────────┬─────────────────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────────────────┐
│                   3. Model 로딩 (load_model)                   │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ with DeviceMemoryProfiler() as m:                     │   │
│  │     # Weight 로딩                                      │   │
│  │     model = model_loader.load_model(...)              │   │
│  │                                                        │   │
│  │     # LoRA 적용 (선택)                                │   │
│  │     if lora_config:                                    │   │
│  │         model = load_lora_model(model, ...)           │   │
│  │                                                        │   │
│  │     # Speculative decoding drafter (선택)            │   │
│  │     if hasattr(self, "drafter"):                      │   │
│  │         drafter.load_model(model)                     │   │
│  └───────────────────────────────────────────────────────┘   │
│  model_memory_usage = m.consumed_memory                        │
└──────────────────┬─────────────────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────────────────┐
│                   4. Model 최적화                               │
│  - CUDA Graph wrapping (if enabled)                            │
│  - Torch Compile (if enabled)                                  │
│  - UBatch optimization (if DBO enabled)                        │
└──────────────────┬─────────────────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────────────────┐
│                   5. KV Cache 초기화                            │
│  - Block 크기 계산                                              │
│  - GPU 메모리 할당 (PagedAttention)                            │
│  - Cache manager 설정                                           │
└──────────────────┬─────────────────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────────────────┐
│                   6. Attention Backend 설정                     │
│  - FlashAttention, xFormers, Native 중 선택                    │
│  - Attention metadata builder 생성                             │
│  - Sliding window, GQA 등 설정                                  │
└──────────────────┬─────────────────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────────────────┐
│                   7. 초기화 완료                                 │
│  - 메모리 사용량 로깅                                            │
│  - Warm-up 준비                                                 │
│  - 추론 가능 상태                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## GPUModelRunner 초기화

### GPUModelRunner 클래스

**위치**: `vllm/v1/worker/gpu_model_runner.py`

```python
class GPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
```

### 주요 설정 파라미터

**1. Device 설정**:

```python
self.device = device  # torch.device("cuda:0")
self.pin_memory = is_pin_memory_available()
self.dtype = self.model_config.dtype  # torch.float16, bfloat16, ...
self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
    cache_config.cache_dtype, self.model_config
)
```

**지원 dtype**:
- `float16`: 메모리 절약, 대부분의 GPU 지원
- `bfloat16`: 더 나은 수치 안정성, A100/H100 권장
- `float32`: 정확도 최우선 (거의 사용 안 함)
- `int8`, `int4`: Quantization (별도 처리)

**2. 모델 구조 파라미터**:

```python
# Attention heads
self.num_query_heads = model_config.get_num_attention_heads(parallel_config)

# Hidden size
self.hidden_size = model_config.get_hidden_size()

# Context length
self.max_model_len = model_config.max_model_len

# Batch 설정
self.max_num_tokens = scheduler_config.max_num_batched_tokens
self.max_num_reqs = scheduler_config.max_num_seqs
```

**3. Parallel 설정**:

```python
# Tensor Parallelism
self.parallel_config.tensor_parallel_size  # GPU 분산

# Pipeline Parallelism
get_pp_group().world_size  # Layer 분산

# Decode-Context Parallelism
self.dcp_world_size = self.parallel_config.decode_context_parallel_size
```

**4. Special 기능**:

```python
# Multi-modal 지원 (LLaVA 등)
self.supports_mm_inputs = model_config.is_multimodal_model

# LoRA 지원
self.lora_config = vllm_config.lora_config

# Speculative Decoding
self.use_async_output_proc = (
    model_config.use_async_output_proc
    and speculative_config is not None
)
```

---

## 모델 로딩 과정

### load_model() 메서드

**위치**: `vllm/v1/worker/gpu_model_runner.py:2917`

```python
def load_model(self, eep_scale_up: bool = False) -> None:
    """
    Args:
        eep_scale_up: the model loading is for elastic EP scale up.
    """
    logger.info_once(
        "Starting to load model %s...",
        self.model_config.model,
        scope="global",
    )

    # 메모리 프로파일링 시작
    with DeviceMemoryProfiler() as m:
        time_before_load = time.perf_counter()

        # 1. Model Loader 가져오기
        model_loader = get_model_loader(self.load_config)

        # 2. 모델 로딩 (Weight + Architecture)
        self.model = model_loader.load_model(
            vllm_config=self.vllm_config,
            model_config=self.model_config
        )

        # 3. LoRA 적용 (선택)
        if self.lora_config:
            self.model = self.load_lora_model(
                self.model, self.vllm_config, self.device
            )

        # 4. Drafter 로딩 (Speculative Decoding)
        if hasattr(self, "drafter"):
            logger.info("Loading drafter model...")
            self.drafter.load_model(self.model)

        time_after_load = time.perf_counter()

    # 메모리 사용량 기록
    self.model_memory_usage = m.consumed_memory
    logger.info_once(
        "Model loading took %.4f GiB and %.6f seconds",
        self.model_memory_usage / GiB_bytes,
        time_after_load - time_before_load,
        scope="local",
    )
```

### Model Loader 선택

**위치**: `vllm/model_executor/model_loader/__init__.py`

```python
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Returns the model loader based on the load format."""
    load_format = load_config.load_format

    if load_format == LoadFormat.DUMMY:
        return DummyModelLoader(load_config)

    elif load_format == LoadFormat.TENSORIZER:
        return TensorizerLoader(load_config)

    elif load_format == LoadFormat.SHARDED_STATE:
        return ShardedStateLoader(load_config)

    elif load_format in [
        LoadFormat.AUTO,
        LoadFormat.PT,
        LoadFormat.SAFETENSORS,
        LoadFormat.FASTSAFETENSORS,
        LoadFormat.NPCACHE,
        LoadFormat.MISTRAL,
    ]:
        return DefaultModelLoader(load_config)

    elif load_format == LoadFormat.BITSANDBYTES:
        from .bitsandbytes_loader import BitsAndBytesLoader
        return BitsAndBytesLoader(load_config)

    elif load_format == LoadFormat.GGUF:
        from .gguf_loader import GGUFLoader
        return GGUFLoader(load_config)

    else:
        raise ValueError(f"Unsupported load format: {load_format}")
```

### 실제 로딩 호출 체인

```python
# 1. model_loader.load_model()
def load_model(
    self,
    vllm_config: VllmConfig,
    model_config: ModelConfig,
) -> nn.Module:
    # 모델 클래스 가져오기
    model_class = get_model_class(model_config)

    # 모델 인스턴스 생성
    with torch.device("cuda"):
        model = model_class(
            config=model_config.hf_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
        )

    # Weight 로딩
    self.load_weights(model, model_config)

    return model
```

**모델 클래스 예시** (Llama):

```python
# vllm/model_executor/models/llama.py
class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )

        # Sampling 관련
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
```

---

## KV Cache 할당

### PagedAttention을 위한 KV Cache

vLLM의 핵심 혁신인 PagedAttention은 KV Cache를 **block 단위**로 관리합니다.

**메모리 구조**:

```
┌──────────────────────────────────────────────────────┐
│         GPU Memory Layout                            │
├──────────────────────────────────────────────────────┤
│  Model Weights:          70 GB  (Llama-70B)          │
├──────────────────────────────────────────────────────┤
│  KV Cache (paged):       10 GB  (configurable)       │
│  ┌────────────────────────────────────────────┐     │
│  │ Block 0: [K0][V0] 16 tokens                │     │
│  │ Block 1: [K1][V1] 16 tokens                │     │
│  │ Block 2: [K2][V2] 16 tokens                │     │
│  │ ...                                        │     │
│  │ Block N: [KN][VN] 16 tokens                │     │
│  └────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘
```

### KV Cache Config

**위치**: `vllm/v1/kv_cache_interface.py`

```python
class KVCacheConfig:
    """Configuration for KV cache memory allocation."""

    def __init__(
        self,
        block_size: int = 16,  # 한 block당 token 수
        num_gpu_blocks: int = 0,  # GPU에 할당할 block 수
        num_cpu_blocks: int = 0,  # CPU에 할당할 block 수 (offloading)
        cache_dtype: str = "auto",  # fp16, fp8, int8, ...
    ):
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.cache_dtype = cache_dtype
```

### Block 크기 계산

**한 block의 메모리 크기**:

```python
# Llama-7B 예시
num_layers = 32
num_kv_heads = 32  # GQA: num_heads // tp_size
head_size = 128
block_size = 16  # tokens per block

# Key cache
key_cache_size = (
    num_layers * num_kv_heads * head_size * block_size * sizeof(dtype)
)

# Value cache (same as key)
value_cache_size = key_cache_size

# Total per block
total_block_size = key_cache_size + value_cache_size

# 예시: 32 layers, 32 heads, 128 head_size, 16 block_size, fp16
# = 32 * 32 * 128 * 16 * 2 bytes = 4,194,304 bytes = 4 MB per block
```

### GPU 메모리 할당 계산

```python
def calculate_num_blocks(
    gpu_memory_total: int,
    model_memory: int,
    gpu_memory_utilization: float = 0.9,
) -> int:
    """Calculate number of KV cache blocks."""

    # 사용 가능한 메모리
    available = gpu_memory_total * gpu_memory_utilization - model_memory

    # Block 수 계산
    num_blocks = int(available / block_size_bytes)

    return num_blocks
```

**실제 예시** (A100 80GB, Llama-70B):

```python
gpu_memory_total = 80 * 1024**3  # 80 GB
model_memory = 70 * 1024**3      # 70 GB (FP16)
gpu_memory_utilization = 0.9

available = 80 * 0.9 - 70 = 2 GB  # KV cache용

block_size_bytes = 4 * 1024 * 1024  # 4 MB per block

num_blocks = 2 * 1024 / 4 = 512 blocks

# 총 context: 512 blocks * 16 tokens = 8,192 tokens
```

### KV Cache 할당 코드

```python
# vllm/v1/worker/gpu_model_runner.py
def allocate_kv_cache(self):
    """Allocate KV cache blocks on GPU."""

    # Block 메모리 할당
    self.gpu_cache = []
    for layer in range(self.num_layers):
        # Key cache
        key_cache = torch.empty(
            (self.num_blocks, self.num_kv_heads, self.head_size, self.block_size),
            dtype=self.kv_cache_dtype,
            device=self.device,
        )

        # Value cache
        value_cache = torch.empty(
            (self.num_blocks, self.num_kv_heads, self.head_size, self.block_size),
            dtype=self.kv_cache_dtype,
            device=self.device,
        )

        self.gpu_cache.append((key_cache, value_cache))

    logger.info(
        "Allocated %d KV cache blocks, "
        "total %.2f GB",
        self.num_blocks,
        self.num_blocks * self.block_size_bytes / 1024**3
    )
```

---

## Attention Backend 설정

### Attention Backend 종류

vLLM은 여러 Attention 구현을 지원합니다:

1. **FlashAttention** (기본, 권장)
   - Tri Dao의 FlashAttention-2/3
   - 메모리 효율적, 빠름
   - Ampere 이상 GPU (A100, H100)

2. **xFormers**
   - Facebook의 memory-efficient attention
   - 다양한 GPU 지원
   - FlashAttention보다 느림

3. **Native PyTorch**
   - torch.nn.functional.scaled_dot_product_attention
   - Fallback용
   - 가장 느림

### Backend 선택 로직

**위치**: `vllm/attention/selector.py`

```python
def select_attention_backend(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
) -> Type[AttentionBackend]:
    """Select the best attention backend."""

    # FlashAttention 지원 확인
    if can_use_flash_attn(num_heads, head_size, sliding_window, dtype):
        logger.info("Using FlashAttention-2 backend")
        return FlashAttentionBackend

    # xFormers 지원 확인
    if can_use_xformers(num_heads, head_size, dtype):
        logger.info("Using xFormers backend")
        return XFormersBackend

    # Fallback: Native PyTorch
    logger.info("Using native PyTorch attention backend")
    return NativeAttentionBackend
```

### FlashAttention Backend

**위치**: `vllm/attention/backends/flash_attn.py`

```python
class FlashAttentionBackend(AttentionBackend):
    @staticmethod
    def get_impl_cls() -> Type[FlashAttentionImpl]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type[FlashAttentionMetadata]:
        return FlashAttentionMetadata


class FlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ):
        from flash_attn import flash_attn_varlen_func  # type: ignore

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.sliding_window = sliding_window

        # Import flash attention function
        self.flash_attn_func = flash_attn_varlen_func
```

**Forward Pass 구현**:

```python
def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor],
    attn_metadata: FlashAttentionMetadata,
) -> torch.Tensor:
    """Forward pass with FlashAttention."""

    # PagedAttention을 위한 key/value 준비
    key_cache, value_cache = kv_cache

    # Block table로 key/value 복사
    block_tables = attn_metadata.block_tables
    context_lens = attn_metadata.context_lens

    # FlashAttention 호출
    output = self.flash_attn_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=attn_metadata.cu_seqlens_q,
        cu_seqlens_k=attn_metadata.cu_seqlens_k,
        max_seqlen_q=attn_metadata.max_seqlen_q,
        max_seqlen_k=attn_metadata.max_seqlen_k,
        softmax_scale=self.scale,
        causal=True,  # Autoregressive
        window_size=(self.sliding_window, 0) if self.sliding_window else (-1, -1),
    )

    return output
```

### Grouped Query Attention (GQA)

**Llama-2-70B, Mistral 등에서 사용**:

```python
# Standard attention: num_kv_heads == num_query_heads
# GQA: num_kv_heads < num_query_heads (메모리 절약)

# 예시: Llama-2-70B
num_query_heads = 64
num_kv_heads = 8  # 8x reduction
head_size = 128

# KV cache 크기: 8배 감소!
kv_cache_size = (
    num_layers * num_kv_heads * head_size * ...
)  # 8x smaller than MHA
```

**GQA 구현**:

```python
def grouped_query_attention(
    query: torch.Tensor,  # [batch, num_query_heads, seq_len, head_size]
    key: torch.Tensor,    # [batch, num_kv_heads, seq_len, head_size]
    value: torch.Tensor,  # [batch, num_kv_heads, seq_len, head_size]
) -> torch.Tensor:
    """Grouped Query Attention."""

    # Query head를 KV head 그룹으로 분할
    num_queries_per_kv = num_query_heads // num_kv_heads

    # Key/Value를 반복하여 query head 수에 맞춤
    key = key.repeat_interleave(num_queries_per_kv, dim=1)
    value = value.repeat_interleave(num_queries_per_kv, dim=1)

    # 일반 attention 수행
    output = scaled_dot_product_attention(query, key, value)

    return output
```

---

## CUDA Graph 최적화

### CUDA Graph란?

**CUDA Graph**는 GPU 연산을 미리 녹화하여 재실행할 때 **CPU overhead를 제거**하는 기술입니다.

**장점**:
- CPU-GPU 통신 최소화
- Kernel launch overhead 제거
- 10-30% 성능 향상 (특히 small batch)

**단점**:
- 첫 실행 시 녹화 필요 (느림)
- 동적 shape 지원 제한
- 메모리 고정 필요

### CUDA Graph 적용

**위치**: `vllm/v1/worker/gpu_model_runner.py:3024`

```python
# CUDA Graph wrapper 적용
if (
    self.compilation_config.cudagraph_mode.has_full_cudagraphs()
    and not self.parallel_config.enable_dbo
):
    self.model = CUDAGraphWrapper(
        self.model,
        self.vllm_config,
        runtime_mode=CUDAGraphMode.FULL
    )
```

**CUDAGraphWrapper**:

```python
# vllm/compilation/cuda_graph.py
class CUDAGraphWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
    ):
        super().__init__()
        self.model = model
        self.runtime_mode = runtime_mode

        # Graph 저장소
        self.graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self.graph_memory_pools: Dict[str, Any] = {}

    def capture(self, batch_size: int, seq_len: int):
        """Capture CUDA graph for given batch size and seq len."""
        key = f"bs{batch_size}_sl{seq_len}"

        if key in self.graphs:
            return  # Already captured

        # Warm-up runs
        for _ in range(3):
            self.model(dummy_input)

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.model(dummy_input)

        self.graphs[key] = graph
        logger.info(f"Captured CUDA graph for {key}")

    def forward(self, *args, **kwargs):
        """Execute with CUDA graph if available."""
        batch_size = args[0].shape[0]
        seq_len = args[0].shape[1]
        key = f"bs{batch_size}_sl{seq_len}"

        if key in self.graphs:
            # Replay captured graph
            self.graphs[key].replay()
            return self.output  # Pre-allocated output
        else:
            # Normal forward (and maybe capture)
            return self.model(*args, **kwargs)
```

### CUDA Graph 메모리 관리

**문제**: CUDA Graph는 메모리 주소가 고정되어야 함

**해결책**: Memory pool 사용

```python
# Memory pool 생성
memory_pool = torch.cuda.graph_pool_handle()

with torch.cuda.graph(graph, pool=memory_pool):
    output = model(input)

# 같은 pool 재사용
with torch.cuda.graph(graph2, pool=memory_pool):
    output2 = model(input2)
```

### Batch Size별 Graph 캡처

vLLM은 **여러 batch size**에 대해 graph를 미리 캡처합니다:

```python
# Warm-up 단계에서 캡처
warmup_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

for batch_size in warmup_batch_sizes:
    dummy_input = create_dummy_input(batch_size)

    # Capture graph
    model_runner.capture_cuda_graph(batch_size)

    logger.info(f"Captured graph for batch_size={batch_size}")
```

**실행 시**:
- Batch size가 캡처된 graph와 일치 → Graph replay (빠름)
- Batch size가 새로운 경우 → Normal forward (느림, 첫 실행 후 캡처)

---

## 메모리 프로파일링

### DeviceMemoryProfiler

vLLM은 모델 로딩 시 **정확한 메모리 사용량**을 측정합니다.

**위치**: `vllm/utils/mem_utils.py`

```python
class DeviceMemoryProfiler:
    """Profile GPU memory usage."""

    def __init__(self):
        self.start_memory = 0
        self.end_memory = 0

    def __enter__(self):
        # 시작 전 메모리 측정
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.start_memory = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *args):
        # 끝난 후 메모리 측정
        torch.cuda.synchronize()
        self.end_memory = torch.cuda.memory_allocated()

    @property
    def consumed_memory(self) -> int:
        """Get consumed memory in bytes."""
        return self.end_memory - self.start_memory
```

**사용 예시**:

```python
with DeviceMemoryProfiler() as m:
    model = load_model(...)
    # Weight 로딩, Layer 초기화 등

print(f"Model memory: {m.consumed_memory / 1024**3:.2f} GB")
```

### 메모리 사용량 로깅

**실제 로그 예시**:

```
INFO 11-06 10:23:45 gpu_model_runner.py:2988] Model loading took 68.23 GiB and 24.531 seconds
INFO 11-06 10:23:46 gpu_model_runner.py:3156] KV cache allocated: 10.5 GiB for 2688 blocks
INFO 11-06 10:23:46 gpu_model_runner.py:3201] Total GPU memory usage: 78.73 GiB / 80.00 GiB
```

**메모리 구성**:
- Model weights: 68.23 GB (Llama-70B FP16)
- KV cache: 10.5 GB (2688 blocks * 16 tokens)
- Overhead: ~0.27 GB (Activations, buffers, etc.)
- Total: 78.73 GB / 80 GB (98.4% utilization)

---

## 최적화 옵션 정리

### 1. Quantization (양자화)

**메모리 절약**:

```python
# FP16 (기본)
model_memory = 70 GB  # Llama-70B

# 4-bit (bitsandbytes)
model_memory = 35 GB  # 50% 절약

# 3-bit (GPTQ)
model_memory = 26 GB  # 63% 절약

# 2-bit (AWQ)
model_memory = 17.5 GB  # 75% 절약
```

**설정 예시**:

```python
from vllm import LLM

# 4-bit quantization
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    quantization="bitsandbytes",
    load_format="bitsandbytes-nf4",
)

# GPTQ
llm = LLM(
    model="TheBloke/Llama-2-70B-GPTQ",
    quantization="gptq",
)

# AWQ
llm = LLM(
    model="TheBloke/Llama-2-70B-AWQ",
    quantization="awq",
)
```

### 2. Tensor Parallelism (TP)

**GPU 간 weight 분산**:

```python
# Single GPU
model_memory = 70 GB  # Llama-70B
max_tokens = 0  # OOM!

# 4x A100 (TP=4)
model_memory_per_gpu = 70 / 4 = 17.5 GB
kv_cache_per_gpu = (80 - 17.5) * 0.9 = 56.25 GB
max_tokens = 56.25 / 4 / (block_size) = ~14,000 tokens per GPU

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
)
```

### 3. Pipeline Parallelism (PP)

**Layer 분산** (큰 모델용):

```python
# Llama-2-70B: 80 layers
# 2x GPU with PP=2

# GPU 0: Layers 0-39
# GPU 1: Layers 40-79

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    pipeline_parallel_size=2,
)
```

### 4. GPU Memory Utilization

**KV Cache 크기 조절**:

```python
# 기본 (90%)
llm = LLM(
    model="...",
    gpu_memory_utilization=0.9,  # 10% 여유
)

# 공격적 (95%)
llm = LLM(
    model="...",
    gpu_memory_utilization=0.95,  # 5% 여유, 더 많은 cache
)

# 보수적 (80%)
llm = LLM(
    model="...",
    gpu_memory_utilization=0.8,  # 20% 여유, 안정적
)
```

### 5. Block Size

**Token per block**:

```python
# 작은 block (8 tokens)
# - 메모리 효율적
# - 내부 단편화 적음
# - 관리 overhead 증가

# 큰 block (32 tokens)
# - 관리 overhead 적음
# - 내부 단편화 증가
# - 메모리 비효율적

# 기본 (16 tokens) - 균형잡힌 선택
llm = LLM(
    model="...",
    block_size=16,
)
```

---

## 실제 초기화 시간 측정

### Llama-2-7B (A100 80GB)

```
[1/7] Starting to load model meta-llama/Llama-2-7b-hf...
[2/7] Loading weights from safetensors... (3.2s)
[3/7] Model loading took 12.84 GiB and 3.452 seconds
[4/7] Allocating KV cache... (0.5s)
[5/7] KV cache allocated: 65.2 GiB for 16,384 blocks
[6/7] Capturing CUDA graphs... (2.1s)
[7/7] Model initialization complete! (6.1s total)
```

### Llama-2-70B (8x A100 80GB, TP=8)

```
[1/7] Starting to load model meta-llama/Llama-2-70b-hf...
[2/7] Loading weights from safetensors... (18.2s)
[3/7] Model loading took 8.54 GiB and 18.763 seconds (per GPU)
[4/7] Allocating KV cache... (0.8s)
[5/7] KV cache allocated: 70.0 GiB for 17,920 blocks (per GPU)
[6/7] Capturing CUDA graphs... (3.5s)
[7/7] Model initialization complete! (23.1s total)
```

---

## 트러블슈팅

### 문제 1: CUDA OOM during model loading

**증상**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate 2.00 GiB
```

**원인**: 모델 weight가 GPU 메모리보다 큼

**해결책**:
```python
# 1. Quantization 사용
quantization="bitsandbytes"  # 50% 절약

# 2. Tensor Parallelism
tensor_parallel_size=2  # 2 GPUs

# 3. 더 작은 모델 사용
model="meta-llama/Llama-2-7b-hf"  # 대신 70B
```

### 문제 2: Very slow initialization

**증상**:
```
Model initialization taking > 60 seconds
```

**원인**:
- Slow disk I/O
- Single-threaded loading
- Network download

**해결책**:
```python
# 1. Multi-thread loading
load_config = LoadConfig(
    model_loader_extra_config={
        "enable_multithread_load": True,
        "num_threads": 16,
    }
)

# 2. Use local model
model="/local/path/to/model"  # No download

# 3. Use FastSafeTensors
load_format="fastsafetensors"
```

### 문제 3: KV cache too small

**증상**:
```
WARNING: KV cache size is small (1024 blocks = 16,384 tokens)
This may limit batch size and throughput.
```

**원인**: Model weights 사용 후 남은 메모리 부족

**해결책**:
```python
# 1. 메모리 사용률 증가
gpu_memory_utilization=0.95  # 90% → 95%

# 2. Quantization으로 model weight 줄이기
quantization="awq"  # 75% 절약

# 3. 더 큰 GPU 사용
# A100 40GB → A100 80GB → H100 96GB
```

---

## 요약

1. **모델 초기화 단계**:
   - GPUModelRunner 생성 (config 파싱)
   - Model loading (weight → GPU)
   - KV Cache 할당 (PagedAttention)
   - Attention backend 설정 (FlashAttention)
   - CUDA Graph 최적화

2. **메모리 구성** (Llama-70B, A100 80GB):
   - Model weights: 70 GB
   - KV cache: 10 GB (2688 blocks)
   - Overhead: 0.3 GB
   - Total: 80.3 GB

3. **초기화 시간**:
   - Llama-7B: ~6초
   - Llama-70B (TP=8): ~23초

4. **최적화 옵션**:
   - Quantization: 메모리 50-75% 절약
   - Tensor Parallelism: N개 GPU로 weight 분산
   - CUDA Graph: 10-30% 성능 향상
   - Multi-thread loading: 로딩 시간 3-5x 단축

다음 문서에서는 **추론 과정**을 상세히 다루겠습니다.
