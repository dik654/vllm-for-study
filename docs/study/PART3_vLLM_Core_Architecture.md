# PART 3: vLLM Core Architecture

> 면접에서 합격/탈락의 갈림점 - PagedAttention과 메모리 관리의 완벽 이해

---

## 목차
1. [vLLM의 PagedAttention 구조](#1-vllm의-pagedattention-구조)
2. [PyTorch 기본 KV Cache의 OOM 문제](#2-pytorch-기본-kv-cache의-oom-문제)
3. [Block Allocation / Block Reuse 전략](#3-block-allocation--block-reuse-전략)
4. [Memory Pool Allocator의 동작](#4-memory-pool-allocator의-동작)
5. [Trust-Region Allocator와 비교](#5-trust-region-allocator와-비교)
6. [Virtual KV Memory로서의 PagedAttention](#6-virtual-kv-memory로서의-pagedattention)
7. [GPU 메모리 Fragmentation 해결](#7-gpu-메모리-fragmentation-해결)
8. [KV Block Swap-in/out 동작](#8-kv-block-swap-inout-동작)
9. [Block Size 조정과 성능](#9-block-size-조정과-성능)
10. [면접 예상 질문 및 답변](#10-면접-예상-질문-및-답변)

---

## 1. vLLM의 PagedAttention 구조

### 1.1 핵심 아이디어

**운영체제의 가상 메모리에서 영감:**

```
┌─────────────────────────────────────────────────────────────────┐
│  운영체제 가상 메모리           │  vLLM PagedAttention           │
├─────────────────────────────────────────────────────────────────┤
│  프로세스의 가상 주소 공간      │  요청의 KV Cache               │
│  물리적 메모리 페이지           │  GPU 메모리 블록               │
│  페이지 테이블                  │  블록 테이블                   │
│  페이지 폴트 시 할당            │  필요 시 블록 할당             │
│  스왑 아웃/인                   │  GPU ↔ CPU 스왑               │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 기존 방식 vs PagedAttention

**기존 방식 (연속 할당):**
```
GPU 메모리:
┌────────────────────────────────────────────────────────────┐
│ Req1: [████████████████]     max_len만큼 미리 할당         │
│ Req2: [████████████████]     실제 사용: [████░░░░░░░░░░]  │
│ Req3: [████████████████]     → 큰 낭비!                   │
└────────────────────────────────────────────────────────────┘
```

**PagedAttention (페이지 단위 할당):**
```
GPU 메모리:
┌────────────────────────────────────────────────────────────┐
│ [B0][B1][B2][B3][B4][B5][B6][B7][B8][B9][...]  고정 크기 블록│
└────────────────────────────────────────────────────────────┘
           ↑    ↑    ↑
           │    │    └─ Req3가 사용
           │    └─ Req1이 사용
           └─ Req2가 사용

블록 테이블:
Req1 → [0, 3, 7]     필요한 만큼만 할당
Req2 → [1, 5]        연속일 필요 없음
Req3 → [2, 4, 6, 8]  동적 확장 가능
```

### 1.3 PagedAttention의 구조

**블록 정의:**
```python
# vllm/v1/core/kv_cache_utils.py 기반

@dataclass
class KVCacheBlock:
    """단일 KV 캐시 블록"""
    block_id: int
    ref_cnt: int = 0          # 참조 카운트 (공유 시 사용)
    block_hash: Optional[int] = None  # Prefix caching용 해시

    # 블록 크기: block_size × num_kv_heads × head_dim × 2 (K,V)
    # 예: 16 × 8 × 128 × 2 = 32KB per block
```

**블록 테이블:**
```python
# 논리적 블록 인덱스 → 물리적 블록 ID 매핑
block_table = {
    "request_1": [0, 3, 7, 12],  # 4개 블록 사용 중
    "request_2": [1, 5],         # 2개 블록 사용 중
    "request_3": [2, 4, 6, 8],   # 4개 블록 사용 중
}
```

### 1.4 Attention 계산에서의 블록 활용

```python
# vllm/attention/ops/paged_attn.py 참조

def paged_attention_forward(
    query: Tensor,           # [batch, 1, num_heads, head_dim]
    key_cache: Tensor,       # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: Tensor,     # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: Tensor,    # [batch, max_blocks_per_seq]
    seq_lens: Tensor,        # [batch]
):
    """
    각 쿼리가 자신의 block_table을 따라 KV를 참조
    """
    for batch_idx in range(batch):
        # 이 요청의 블록 테이블
        blocks = block_tables[batch_idx]
        seq_len = seq_lens[batch_idx]

        # 블록별로 attention 계산
        for block_idx in range(num_blocks):
            physical_block = blocks[block_idx]
            k_block = key_cache[physical_block]    # [block_size, heads, dim]
            v_block = value_cache[physical_block]

            # Q × K^T 계산 및 V 가중합
            scores = query[batch_idx] @ k_block.T
            # ...
```

### 1.5 vLLM 코드에서의 PagedAttention

```python
# vllm/attention/ops/paged_attn.py:41-55

@dataclass
class PagedAttentionMetadata:
    """PagedAttention을 위한 메타데이터"""

    # (batch_size,). 각 시퀀스의 전체 길이
    seq_lens_tensor: torch.Tensor | None

    # 배치 내 최대 디코드 시퀀스 길이
    max_decode_seq_len: int

    # (batch_size, max_blocks_per_seq)
    # 시퀀스별 블록 주소 테이블
    # 예: [0, 1, 2]는 토큰이 0, 1, 2번 블록에 저장됨을 의미
    block_tables: torch.Tensor | None
```

---

## 2. PyTorch 기본 KV Cache의 OOM 문제

### 2.1 기존 방식의 문제점

**PyTorch의 연속 메모리 할당:**
```python
# 기존 방식
class NaiveKVCache:
    def __init__(self, max_batch, max_seq_len, num_layers, num_heads, head_dim):
        # 최대 크기로 미리 할당!
        self.k_cache = torch.zeros(
            max_batch, max_seq_len, num_layers, num_heads, head_dim
        )
        self.v_cache = torch.zeros(
            max_batch, max_seq_len, num_layers, num_heads, head_dim
        )
```

**문제 1: 과다 할당 (Over-allocation)**
```
설정: max_seq_len = 4096
실제 요청들:
- Req1: 실제 길이 100  → 3996 토큰 낭비
- Req2: 실제 길이 500  → 3596 토큰 낭비
- Req3: 실제 길이 200  → 3896 토큰 낭비

낭비율: (3996 + 3596 + 3896) / (4096 × 3) = 93%!
```

**문제 2: 내부 파편화 (Internal Fragmentation)**
```
GPU 메모리:
┌──────────────────────────────────────────────────────────┐
│ [Req1: █████░░░░░░░░░░░][Req2: ██████████░░░░░░]        │
│         ↑ 사용          ↑ 낭비     ↑ 사용    ↑ 낭비     │
└──────────────────────────────────────────────────────────┘

→ 고정 크기 할당으로 인한 내부 파편화
```

**문제 3: 외부 파편화 (External Fragmentation)**
```
시간 경과 후:
┌──────────────────────────────────────────────────────────┐
│ [Free][Req2][Free][Req4][Free][Free][Req6][Free]        │
└──────────────────────────────────────────────────────────┘

총 여유 공간: 충분
연속 여유 공간: 부족
→ 새 요청 할당 불가 (OOM)
```

### 2.2 수치로 보는 낭비

**LLaMA-7B 예시:**
```
KV Cache per token: 0.5 MB
max_seq_len: 4096

Naive 방식:
- 1 요청당 예약: 4096 × 0.5 MB = 2 GB
- 8 요청 동시: 16 GB

실제 평균 시퀀스 길이가 500이라면:
- 실제 사용: 500 × 0.5 MB × 8 = 2 GB
- 낭비: 14 GB (87.5%)

PagedAttention:
- 실제 사용량만 할당: 2 GB
- 추가 오버헤드: ~1-5%
```

### 2.3 가변 길이의 어려움

```
요청 도착 시나리오:
t=0: Req1 도착 (예측 길이 불명)
t=1: Req2 도착
t=2: Req1이 예상보다 길어짐
t=3: Req3 도착, 메모리 부족

Naive: Req3 거부 또는 OOM
PagedAttention: 동적 블록 할당으로 해결
```

---

## 3. Block Allocation / Block Reuse 전략

### 3.1 블록 할당 기본 원리

```python
# vllm/v1/core/block_pool.py:125-168 기반

class BlockPool:
    """KV 캐시 블록 풀 관리자"""

    def __init__(self, num_gpu_blocks: int, enable_caching: bool):
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching

        # 모든 블록 생성
        self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]

        # 프리 블록 큐 (이중 연결 리스트)
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Prefix caching용 해시 캐시
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Null 블록 (padding용)
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
```

### 3.2 블록 할당 과정

```python
# 블록 할당 로직

def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    """프리 블록 풀에서 새 블록 할당"""

    if num_blocks > self.get_num_free_blocks():
        raise ValueError("Not enough free blocks")

    # 프리 큐에서 블록 가져오기
    ret = self.free_block_queue.popleft_n(num_blocks)

    if self.enable_caching:
        for block in ret:
            # 캐시된 블록이면 캐시에서 제거 (eviction)
            self._maybe_evict_cached_block(block)
            block.ref_cnt += 1
    else:
        for block in ret:
            block.ref_cnt += 1

    return ret
```

### 3.3 블록 재사용 (Prefix Caching)

**핵심 아이디어:**
동일한 프롬프트 prefix를 가진 요청들이 KV 캐시를 공유

```
Request A: "You are a helpful AI assistant. Answer: What is 2+2?"
Request B: "You are a helpful AI assistant. Answer: What is the capital?"
                ↑ 공통 prefix ↑

→ "You are a helpful AI assistant. Answer:" 부분의 KV 캐시 공유!
```

**구현:**
```python
# vllm/v1/core/block_pool.py:196-266

def cache_full_blocks(
    self,
    request: Request,
    blocks: list[KVCacheBlock],
    num_cached_blocks: int,
    num_full_blocks: int,
    block_size: int,
    kv_cache_group_id: int,
):
    """완성된 블록을 캐시에 추가"""

    new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
    new_block_hashes = request.block_hashes[num_cached_blocks:]

    for i, blk in enumerate(new_full_blocks):
        block_hash = new_block_hashes[i]

        # 블록에 해시 저장
        block_hash_with_group = make_block_hash_with_group_id(
            block_hash, kv_cache_group_id
        )
        blk.block_hash = block_hash_with_group

        # 캐시에 등록
        self.cached_block_hash_to_block.insert(block_hash_with_group, blk)
```

### 3.4 블록 해제와 참조 카운팅

```python
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    """블록 해제 (참조 카운트 기반)"""

    for block in ordered_blocks:
        block.ref_cnt -= 1

    # ref_cnt가 0이 된 블록만 프리 큐에 반환
    self.free_block_queue.append_n([
        block for block in ordered_blocks
        if block.ref_cnt == 0 and not block.is_null
    ])

def touch(self, blocks: tuple[Sequence[KVCacheBlock], ...]) -> None:
    """캐시 히트 시 참조 카운트 증가"""
    for blocks_per_group in blocks:
        for block in blocks_per_group:
            if block.ref_cnt == 0 and not block.is_null:
                # 프리 큐에서 제거 (다시 사용 중)
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
```

### 3.5 블록 재사용 시나리오

```
시나리오: 3개의 유사한 요청

Time 0:
Req1: "Summarize this article: [긴 문서]"
블록 할당: [B0, B1, B2, B3, B4]
         prefix  ↑    content ↑

Time 1:
Req2: "Summarize this article: [다른 문서]"
캐시 히트! "Summarize this article:" 재사용
블록: [B0 (공유), B5, B6, B7]
       ↑ ref_cnt=2

Time 2:
Req1 완료, 블록 해제
B0: ref_cnt 2→1 (아직 Req2가 사용 중)
B1~B4: ref_cnt 0 → 프리 큐로

Time 3:
Req3: "Summarize this article: [또 다른 문서]"
캐시 히트! B0 재사용
블록: [B0 (공유), B1, B2, B8]
       ↑ ref_cnt=2
```

---

## 4. Memory Pool Allocator의 동작

### 4.1 메모리 풀 개념

**전통적 malloc vs 메모리 풀:**

```
malloc 방식:
┌─────────────────────────────────────────────┐
│ 각 요청마다 시스템 콜                         │
│ 가변 크기 → 파편화                           │
│ 해제 시 운영체제에 반환                       │
└─────────────────────────────────────────────┘

메모리 풀 방식:
┌─────────────────────────────────────────────┐
│ 시작 시 큰 메모리 블록 한 번 할당            │
│ 고정 크기 슬롯으로 분할                      │
│ 내부에서 할당/해제 관리 (빠름)               │
│ 파편화 없음                                  │
└─────────────────────────────────────────────┘
```

### 4.2 vLLM의 GPU 메모리 풀

```python
# 개념적 구조

class GPUMemoryPool:
    def __init__(self, total_gpu_memory: int, block_size_bytes: int):
        # 전체 GPU 메모리를 한 번에 할당
        self.memory = torch.empty(
            total_gpu_memory,
            dtype=torch.uint8,
            device='cuda'
        )

        # 고정 크기 블록으로 분할
        self.num_blocks = total_gpu_memory // block_size_bytes
        self.blocks = [
            self.memory[i * block_size_bytes : (i+1) * block_size_bytes]
            for i in range(self.num_blocks)
        ]

        # 프리 리스트
        self.free_list = list(range(self.num_blocks))

    def allocate_block(self) -> int:
        """O(1) 블록 할당"""
        return self.free_list.pop()

    def free_block(self, block_id: int):
        """O(1) 블록 해제"""
        self.free_list.append(block_id)
```

### 4.3 FreeKVCacheBlockQueue의 구현

```python
# vllm/v1/core/kv_cache_utils.py 기반

class FreeKVCacheBlockQueue:
    """이중 연결 리스트 기반 프리 블록 큐"""

    def __init__(self, blocks: list[KVCacheBlock]):
        self.num_free_blocks = len(blocks)

        # 이중 연결 리스트 구성
        self.head = blocks[0]
        self.tail = blocks[-1]

        for i, block in enumerate(blocks):
            block.prev = blocks[i-1] if i > 0 else None
            block.next = blocks[i+1] if i < len(blocks)-1 else None

    def popleft(self) -> KVCacheBlock:
        """O(1) 앞에서 제거"""
        block = self.head
        self.head = block.next
        if self.head:
            self.head.prev = None
        self.num_free_blocks -= 1
        return block

    def append(self, block: KVCacheBlock):
        """O(1) 뒤에 추가"""
        block.prev = self.tail
        block.next = None
        if self.tail:
            self.tail.next = block
        self.tail = block
        self.num_free_blocks += 1

    def remove(self, block: KVCacheBlock):
        """O(1) 중간에서 제거"""
        if block.prev:
            block.prev.next = block.next
        else:
            self.head = block.next

        if block.next:
            block.next.prev = block.prev
        else:
            self.tail = block.prev

        self.num_free_blocks -= 1
```

### 4.4 Page 단위 동작 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                      GPU Memory Pool                            │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐          │
│  │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ B8 │ B9 │          │
│  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘          │
│    ↑              ↑                        ↑                   │
│    │              │                        │                   │
│  Free Queue: [B0] ←→ [B3] ←→ [B8] (LRU 순서)                  │
│                                                                 │
│  Allocated: B1(Req1), B2(Req1), B4(Req2), B5(Req2), ...       │
│                                                                 │
│  Block Tables:                                                  │
│    Req1 → [1, 2, 6]                                            │
│    Req2 → [4, 5, 7, 9]                                         │
└─────────────────────────────────────────────────────────────────┘

할당 요청: Req3가 2개 블록 필요
1. free_queue.popleft() → B0
2. free_queue.popleft() → B3
3. Req3 block_table: [0, 3]
```

---

## 5. Trust-Region Allocator와 비교

### 5.1 Trust-Region Allocator란?

**수학적 최적화에서 차용:**
"신뢰 구역" 내에서만 메모리 할당을 허용

```
개념:
┌─────────────────────────────────────────────────────────────────┐
│  전체 메모리                                                    │
│  ┌──────────────────────────────────────────────────┐          │
│  │     Trust Region (안전한 할당 영역)              │          │
│  │  ┌──────────────────────────────────┐           │          │
│  │  │   현재 사용 중                    │           │          │
│  │  └──────────────────────────────────┘           │          │
│  │          ↑ 확장 가능                            │          │
│  └──────────────────────────────────────────────────┘          │
│              ↑ 신뢰 구역 경계                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 비교: PagedAttention vs Trust-Region

| 특성 | PagedAttention | Trust-Region |
|------|----------------|--------------|
| 할당 단위 | 고정 크기 블록 | 가변 크기 |
| 파편화 | 거의 없음 | 가능성 있음 |
| 오버헤드 | 블록 테이블 유지 | 경계 추적 |
| 유연성 | 매우 높음 | 중간 |
| 공유 | 블록 단위 쉬움 | 복잡함 |
| 구현 복잡도 | 중간 | 상대적 단순 |

### 5.3 Trust-Region의 장단점

**장점:**
- 블록 테이블 오버헤드 없음
- 연속 메모리 접근 가능

**단점:**
- 요청 간 공유 어려움
- 외부 파편화 가능성
- 동적 확장 시 이동 필요

### 5.4 왜 vLLM은 PagedAttention을 선택했나?

```
1. 메모리 효율: 99% 활용률 달성
2. 공유 용이: Prefix caching 자연스럽게 지원
3. 동적 확장: 길이 예측 불필요
4. 파편화 제거: 고정 블록 크기
5. 스왑 효율: 블록 단위 GPU↔CPU 이동
```

---

## 6. Virtual KV Memory로서의 PagedAttention

### 6.1 운영체제 가상 메모리와의 유사성

```
┌────────────────────────────────────────────────────────────────┐
│              운영체제                  │        vLLM            │
├────────────────────────────────────────────────────────────────┤
│  프로세스                             │  요청 (Request)        │
│  가상 주소 공간                       │  논리적 KV 시퀀스      │
│  물리 메모리 페이지                   │  GPU 블록              │
│  페이지 테이블                        │  블록 테이블           │
│  페이지 크기 (4KB)                    │  블록 크기 (16 tokens) │
│  스왑 공간 (디스크)                   │  CPU 메모리            │
│  Copy-on-Write                        │  Prefix 블록 공유      │
│  Demand Paging                        │  필요 시 블록 할당     │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 가상 메모리의 핵심 이점

**1. 주소 공간 격리:**
```
요청 A의 관점: 토큰 0~99 → 연속된 KV 공간
요청 B의 관점: 토큰 0~199 → 연속된 KV 공간

실제 물리 배치:
GPU: [B_A0][B_B0][B_A1][B_B1][B_B2][...]

→ 각 요청은 자신만의 연속 공간을 "본다"
```

**2. Demand Allocation:**
```python
# 시퀀스 시작 시
request = new_request()
request.blocks = []  # 블록 0개로 시작

# 토큰 생성 중 블록 필요 시
if len(current_tokens) > len(request.blocks) * block_size:
    new_block = allocator.get_new_block()
    request.blocks.append(new_block)

# 미래에 필요할 블록을 미리 할당하지 않음!
```

**3. Copy-on-Write:**
```
시나리오: 두 요청이 동일한 prefix 공유

Req1: "Hello world! Answer:" → blocks [B0, B1, B2]
Req2: "Hello world! Answer:" → 캐시 히트!

공유:
Req2.blocks = [B0, B1, B2]  (같은 블록 참조)
B0.ref_cnt = 2
B1.ref_cnt = 2
B2.ref_cnt = 2

Req2가 새 토큰 생성 시:
Req2.blocks.append(new_block)  → [B0, B1, B2, B3]
                                  공유    새 블록
```

### 6.3 주소 변환 과정

```
논리 주소 → 물리 주소 변환:

논리적 토큰 위치: position = 42
블록 크기: block_size = 16

계산:
block_index = position // block_size = 42 // 16 = 2
offset = position % block_size = 42 % 16 = 10

블록 테이블 참조:
physical_block = block_table[request_id][block_index]
              = block_table["req1"][2] = 7

최종 물리 주소:
kv_cache_address = &gpu_blocks[7] + offset * entry_size
```

### 6.4 Attention에서의 가상→물리 변환

```python
# CUDA 커널 내부 (개념적)

def paged_attention_kernel(query, kv_cache, block_table, seq_len):
    """각 query가 자신의 block_table을 통해 KV 접근"""

    output = zeros_like(query)

    for pos in range(seq_len):
        # 논리 → 물리 변환
        block_idx = pos // BLOCK_SIZE
        block_offset = pos % BLOCK_SIZE
        physical_block = block_table[block_idx]

        # 물리 블록에서 KV 읽기
        k = kv_cache.key[physical_block, block_offset]
        v = kv_cache.value[physical_block, block_offset]

        # Attention 계산
        score = dot(query, k) / sqrt(d)
        output += softmax_score * v

    return output
```

---

## 7. GPU 메모리 Fragmentation 해결

### 7.1 파편화의 종류

**내부 파편화 (Internal Fragmentation):**
```
할당 단위보다 작은 요청 시 발생

예: 블록 크기 = 16 tokens
요청 A: 10 tokens 필요 → 1 블록 할당, 6 slots 낭비
요청 B: 18 tokens 필요 → 2 블록 할당, 14 slots 낭비

해결: 블록 크기 최적화 (작을수록 내부 파편화↓, 오버헤드↑)
```

**외부 파편화 (External Fragmentation):**
```
기존 방식에서:
┌────────────────────────────────────────────────┐
│ [Req1:████][Free][Req2:██████][Free][Req3:██] │
└────────────────────────────────────────────────┘
              ↑           ↑
              │           └─ 3 블록 여유
              └─ 2 블록 여유

새 요청: 4 블록 필요
총 여유: 5 블록 있지만, 연속 4 블록 없음 → 할당 실패!

PagedAttention에서:
연속 메모리 불필요 → 외부 파편화 문제 없음
```

### 7.2 PagedAttention의 파편화 해결

```
┌─────────────────────────────────────────────────────────────────┐
│  PagedAttention: 외부 파편화 완전 제거                          │
│                                                                 │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐          │
│  │Free│Req1│Free│Req2│Req2│Free│Req1│Free│Req2│Free│          │
│  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘          │
│                                                                 │
│  새 요청 4 블록 필요:                                           │
│  - Free 블록: [0, 2, 5, 7, 9] → 5개!                           │
│  - 연속일 필요 없음                                             │
│  - 바로 할당: [0, 2, 5, 7]                                      │
│                                                                 │
│  결과:                                                          │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐          │
│  │Req3│Req1│Req3│Req2│Req2│Req3│Req1│Req3│Req2│Free│          │
│  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 메모리 효율성 달성

**vLLM 논문의 핵심 결과:**
```
메모리 낭비 비교:

Naive 방식:
- 예약 기반: ~60-80% 낭비
- 파편화로 인한 추가 손실

PagedAttention:
- 마지막 블록만 낭비: < 4% (block_size=16 가정)
- 파편화 없음

실제 메모리 활용률:
Naive: 20-40%
PagedAttention: 96-99%

→ 2-4배 더 많은 요청 동시 처리 가능!
```

### 7.4 Compaction 불필요

**기존 방식의 Compaction:**
```
파편화 발생 시:
1. 모든 요청 일시 중지
2. 메모리 재배치 (이동)
3. 포인터 업데이트
4. 재개

→ 긴 지연 시간, 복잡한 구현
```

**PagedAttention:**
```
파편화가 없으므로 Compaction 자체가 불필요!

대신 간단한 블록 테이블 업데이트로 처리:
- 블록 해제: free_list에 추가
- 블록 할당: free_list에서 제거

O(1) 연산, 데이터 이동 없음
```

---

## 8. KV Block Swap-in/out 동작

### 8.1 스왑의 필요성

**시나리오:**
```
GPU 메모리: 10 블록
현재 사용: 9 블록

새 요청 도착: 3 블록 필요
→ 2 블록 부족!

선택지:
1. 요청 거부 (throughput 저하)
2. 기존 요청 중단 (preemption)
3. 일부 블록을 CPU로 스왑 (swap-out)

vLLM: 옵션 2, 3 지원
```

### 8.2 Swap-out 과정

```python
# 개념적 코드

def swap_out_blocks(request: Request, blocks_to_swap: List[int]):
    """GPU → CPU로 블록 이동"""

    for block_id in blocks_to_swap:
        # GPU 블록 데이터
        gpu_block = gpu_kv_cache[block_id]

        # CPU 메모리에 복사
        cpu_block_id = cpu_allocator.allocate()
        cpu_kv_cache[cpu_block_id] = gpu_block.clone()

        # 매핑 저장
        request.swap_mapping[block_id] = cpu_block_id

        # GPU 블록 해제
        gpu_allocator.free(block_id)
```

**타이밍:**
```
비동기 스왑 (CUDA stream):

t=0: 스왑 명령 시작
t=1: 다른 요청 처리 (병렬)
t=2: 스왑 완료 콜백
t=3: CPU 블록 사용 가능

→ 스왑 동안 GPU가 놀지 않음!
```

### 8.3 Swap-in 과정

```python
def swap_in_blocks(request: Request, blocks_to_swap: List[int]):
    """CPU → GPU로 블록 이동"""

    for cpu_block_id in blocks_to_swap:
        # GPU 블록 할당
        gpu_block_id = gpu_allocator.allocate()

        # CPU → GPU 복사
        gpu_kv_cache[gpu_block_id] = cpu_kv_cache[cpu_block_id].clone()

        # 블록 테이블 업데이트
        request.update_block_table(cpu_block_id, gpu_block_id)

        # CPU 블록 해제
        cpu_allocator.free(cpu_block_id)
```

### 8.4 Preemption vs Swap 비교

```
┌─────────────────────────────────────────────────────────────────┐
│  Preemption (중단 후 재시작)                                    │
│  - KV 캐시 완전 삭제                                           │
│  - 재시작 시 처음부터 prefill                                  │
│  - 긴 프롬프트에서 비효율                                      │
│                                                                 │
│  Swap (CPU로 이동)                                             │
│  - KV 캐시 CPU에 보존                                          │
│  - 재개 시 swap-in만 필요                                      │
│  - 추가 메모리(CPU) 필요                                       │
│                                                                 │
│  Trade-off:                                                     │
│  프롬프트 길이 < swap 비용 → Preemption 선호                   │
│  프롬프트 길이 > swap 비용 → Swap 선호                         │
└─────────────────────────────────────────────────────────────────┘
```

### 8.5 실제 Swap 성능

```
A100 PCIe 4.0:
GPU ↔ CPU 대역폭: ~25 GB/s

블록 크기: 32KB (16 tokens × 2KB/token)
Swap 시간: 32KB / 25GB/s = 1.3μs/블록

100 블록 스왑: ~130μs

vs Prefill 재계산:
100 블록 = 1600 토큰
Prefill 시간: ~10-50ms

→ Swap이 100배 이상 빠름!
```

---

## 9. Block Size 조정과 성능

### 9.1 Block Size의 영향

**작은 블록 (예: 8 tokens):**
```
장점:
- 낮은 내부 파편화 (마지막 블록 낭비 적음)
- 세밀한 메모리 관리

단점:
- 많은 블록 수 → 큰 블록 테이블
- Attention 커널 오버헤드 증가
- 메모리 접근 패턴 비효율
```

**큰 블록 (예: 256 tokens):**
```
장점:
- 적은 블록 수 → 작은 블록 테이블
- 메모리 접근 효율적 (연속 읽기)
- 커널 오버헤드 감소

단점:
- 높은 내부 파편화
- Prefix caching 효율 저하
- 스왑 단위가 커짐
```

### 9.2 최적 블록 크기 분석

```
내부 파편화 분석:

평균 시퀀스 길이: S
블록 크기: B

예상 마지막 블록 낭비: B/2 tokens (평균)
낭비 비율: (B/2) / S = B / (2S)

예시 (S=1000):
B=8:   낭비 = 0.4%
B=16:  낭비 = 0.8%
B=64:  낭비 = 3.2%
B=256: 낭비 = 12.8%
```

### 9.3 블록 크기와 Attention 커널

```
CUDA 커널 고려사항:

Block size = 16의 경우:
- 16 KV 엔트리를 한 번에 처리
- Warp (32 threads)에서 2배 iteration

Block size = 64의 경우:
- 64 KV 엔트리를 한 번에 처리
- Warp 활용도 높음
- 메모리 coalescing 개선

실험적 최적값: 16~64 tokens
vLLM 기본값: 16
```

### 9.4 블록 크기 결정 가이드

```python
# 최적 블록 크기 선택 가이드

def choose_block_size(
    avg_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    gpu_arch: str
) -> int:
    """
    고려 요소:
    1. 내부 파편화 vs 오버헤드 trade-off
    2. GPU 아키텍처 (cache line, warp size)
    3. KV 캐시 구조
    """

    # 경험적 규칙
    if avg_seq_len < 256:
        return 8   # 짧은 시퀀스: 파편화 최소화
    elif avg_seq_len < 1024:
        return 16  # 중간: 균형
    else:
        return 32  # 긴 시퀀스: 오버헤드 최소화

# vLLM 기본값
DEFAULT_BLOCK_SIZE = 16  # 대부분의 워크로드에 적합
```

### 9.5 성능 벤치마크

```
Block Size 실험 (LLaMA-7B, A100):

│ Block Size │ Throughput │ Memory Util │ 내부 파편화 │
├────────────┼────────────┼─────────────┼─────────────┤
│     8      │   85%      │    98%      │    0.4%     │
│    16      │  100%      │    96%      │    0.8%     │ ← 기본값
│    32      │   98%      │    94%      │    1.6%     │
│    64      │   95%      │    90%      │    3.2%     │
│   128      │   88%      │    85%      │    6.4%     │

→ 16이 throughput/efficiency 균형점
```

---

## 10. 면접 예상 질문 및 답변

### Q1: PagedAttention이 기존 방식 대비 메모리를 얼마나 절약하나요?

**답변:**
기존 방식은 최대 시퀀스 길이로 미리 할당하여 60-80% 메모리가 낭비됩니다.

PagedAttention은:
1. **필요한 만큼만 할당**: 실제 토큰 수에 비례
2. **외부 파편화 제거**: 비연속 블록 사용 가능
3. **내부 파편화 최소화**: 마지막 블록만 부분 사용

결과적으로 메모리 활용률이 **20-40% → 96-99%**로 향상되어, 동일 GPU에서 **2-4배** 더 많은 요청을 동시 처리할 수 있습니다.

### Q2: 블록 테이블 오버헤드는 문제가 되지 않나요?

**답변:**
블록 테이블은 `(batch_size × max_blocks_per_seq)` 크기의 정수 배열입니다.

```
예: batch=256, max_blocks=256, int32
오버헤드 = 256 × 256 × 4 = 256KB

vs KV 캐시:
256 requests × 2048 tokens × 0.5MB/token = 256GB

비율: 256KB / 256GB = 0.0001%
```

따라서 오버헤드는 무시할 수 있는 수준입니다.

### Q3: Prefix caching의 동작 원리는?

**답변:**
1. 완성된 블록의 content를 해시
2. 새 요청의 prefix 해시와 비교
3. 일치 시 기존 블록 재사용 (ref_cnt 증가)
4. 불일치 시 새 블록 할당

```
효과:
- "You are a helpful assistant" prefix 공유 시
- 10개 요청이 같은 블록 공유 → 90% 메모리 절약
- Prefill 스킵 → latency 감소
```

### Q4: Swap-out/in은 언제 사용되나요?

**답변:**
GPU 메모리가 부족할 때 대안으로 사용됩니다:

1. **Preemption 대신**: 긴 프롬프트의 경우 swap이 re-prefill보다 빠름
2. **우선순위 관리**: 낮은 우선순위 요청을 CPU로 이동
3. **메모리 확보**: 새 요청을 위한 공간 마련

**Trade-off:**
- 짧은 시퀀스: Preemption 선호 (swap 비용 > prefill 비용)
- 긴 시퀀스: Swap 선호 (swap 비용 < prefill 비용)

### Q5: Block size를 어떻게 선택해야 하나요?

**답변:**
```
작은 블록 (8-16):
- 짧은 시퀀스, 다양한 길이
- 파편화 최소화 우선

큰 블록 (32-64):
- 긴 시퀀스, 일정한 길이
- 커널 효율 우선

vLLM 기본값 16은 대부분의 워크로드에서 균형잡힌 선택입니다.
특수한 경우 벤치마킹으로 튜닝이 필요합니다.
```

### Q6: PagedAttention이 OS 가상 메모리와 유사한 점은?

**답변:**
핵심 유사점:
1. **논리/물리 분리**: 요청은 연속 공간을 보지만, 물리적으로는 분산
2. **페이지 테이블**: 블록 테이블이 매핑 담당
3. **Demand paging**: 필요 시 블록 할당
4. **Copy-on-Write**: Prefix 블록 공유
5. **Swap 공간**: CPU 메모리를 보조 저장소로 활용

이러한 추상화 덕분에 효율적인 메모리 관리가 가능합니다.

---

## 참고 자료

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- vLLM 소스코드:
  - `vllm/v1/core/block_pool.py` - 블록 풀 관리
  - `vllm/v1/core/kv_cache_utils.py` - KV 캐시 유틸리티
  - `vllm/attention/ops/paged_attn.py` - PagedAttention 커널
  - `vllm/v1/kv_cache_interface.py` - KV 캐시 인터페이스
