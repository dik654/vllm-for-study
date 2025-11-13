# Distributed Training: 대규모 모델 훈련

## 🎯 왜 Distributed Training?

```
문제: 모델이 너무 커서 single GPU에 안 들어감!

GPT-3 175B (FP16): 350GB
A100 GPU: 80GB

→ 최소 5개 GPU 필요!
```

---

## 📚 병렬화 전략

### 1. Data Parallelism (DP)

**개념**: 각 GPU가 전체 모델 + 서로 다른 데이터

```python
# 가장 간단한 방법
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. 초기화
torch.distributed.init_process_group(backend='nccl')

# 2. 모델을 각 GPU에 복제
model = MyModel().cuda()
model = DDP(model, device_ids=[local_rank])

# 3. 훈련
for batch in dataloader:
    outputs = model(batch)
    loss = criterion(outputs, labels)
    loss.backward()  # Gradients 자동 동기화!
    optimizer.step()

# 장점: 구현 쉬움
# 단점: 모델이 GPU 메모리보다 크면 불가능
```

**동작 원리**:
```
GPU 0: Model copy + Batch 0-31
GPU 1: Model copy + Batch 32-63
GPU 2: Model copy + Batch 64-95
GPU 3: Model copy + Batch 96-127

Forward → Backward → All-Reduce gradients → Update
```

**실습 코드**:
```python
# train_ddp.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Model
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Data (중요: DistributedSampler 사용!)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    
    optimizer = torch.optim.Adam(ddp_model.parameters())
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffling 동기화
        
        for batch in dataloader:
            batch = batch.to(rank)
            
            optimizer.zero_grad()
            loss = ddp_model(batch)
            loss.backward()
            optimizer.step()
    
    cleanup()

# Launch
if __name__ == "__main__":
    world_size = 4  # 4 GPUs
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

```bash
# 또는 torchrun 사용
torchrun --nproc_per_node=4 train_ddp.py
```

---

### 2. Model Parallelism

#### 2.1 Pipeline Parallelism

**개념**: 모델을 layer별로 나눔

```python
# GPU 0: Layers 0-5
# GPU 1: Layers 6-11
# GPU 2: Layers 12-17
# GPU 3: Layers 18-23

from torch.distributed.pipeline.sync import Pipe

# Split model
layers = [Layer1(), Layer2(), ..., Layer24()]
model = nn.Sequential(*layers)

# Pipeline across 4 GPUs
model = Pipe(model, balance=[6, 6, 6, 6], chunks=8)

# chunks: micro-batches (pipeline parallelism의 핵심!)
```

**Pipeline Schedule (GPipe)**:
```
Time →
GPU 0: [F0] [F1] [F2] [F3] [B0] [B1] [B2] [B3]
GPU 1:      [F0] [F1] [F2] [F3] [B0] [B1] [B2] [B3]
GPU 2:           [F0] [F1] [F2] [F3] [B0] [B1] [B2]
GPU 3:                [F0] [F1] [F2] [F3] [B0] [B1]

F = Forward, B = Backward
문제: Bubble (GPU idle time)
```

**개선: 1F1B (One Forward One Backward)**:
```
GPU 0: [F0] [F1] [F2] [F3] [B0] [F4] [B1] [F5] [B2]
GPU 1: [  ] [F0] [F1] [F2] [B0] [F3] [B1] [F4] [B2]
GPU 2: [  ] [  ] [F0] [F1] [B0] [F2] [B1] [F3] [B2]
GPU 3: [  ] [  ] [  ] [F0] [B0] [F1] [B1] [F2] [B2]

→ Bubble 크기 감소!
```

#### 2.2 Tensor Parallelism (Megatron-LM)

**개념**: 각 layer의 tensor를 나눔

```python
# Transformer Layer
class TransformerLayer:
    def __init__(self, d_model, n_heads):
        # Q, K, V projection을 column-wise split
        self.qkv_proj = ColumnParallelLinear(d_model, 3*d_model)
        
        # Output projection을 row-wise split
        self.out_proj = RowParallelLinear(d_model, d_model)

class ColumnParallelLinear(nn.Module):
    """Column-wise parallelism"""
    def __init__(self, in_features, out_features):
        super().__init__()
        # Each GPU has out_features // world_size columns
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features // world_size)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, in_features)
        # All GPUs have same input (replicated)
        output = x @ self.weight
        # output: (batch, seq_len, out_features // world_size)
        return output

class RowParallelLinear(nn.Module):
    """Row-wise parallelism"""
    def __init__(self, in_features, out_features):
        super().__init__()
        # Each GPU has in_features // world_size rows
        self.weight = nn.Parameter(
            torch.randn(in_features // world_size, out_features)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, in_features // world_size) - partitioned!
        output_partial = x @ self.weight
        
        # All-reduce across GPUs
        output = torch.distributed.all_reduce(output_partial)
        return output

# Attention with tensor parallelism
def tensor_parallel_attention(x):
    # QKV projection (column parallel)
    qkv = column_parallel_linear(x)  # Each GPU has 1/N of output
    
    # Split into Q, K, V
    q, k, v = qkv.chunk(3, dim=-1)
    
    # Attention (local on each GPU)
    scores = q @ k.transpose(-2, -1) / sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    output = attn @ v
    
    # Output projection (row parallel, includes all-reduce)
    output = row_parallel_linear(output)
    return output
```

---

### 3. ZeRO (DeepSpeed)

**Zero Redundancy Optimizer**

**문제**: Data parallelism은 중복이 많음
```
각 GPU가 저장:
- Model parameters
- Gradients
- Optimizer states (momentum, variance for Adam)

총 메모리 = 2x (params) + 2x (gradients) + 12x (optimizer) = 16x
→ 엄청난 중복!
```

**ZeRO-1**: Optimizer state만 shard
```python
# 각 GPU가 optimizer state의 1/N만 저장
# Parameters, gradients는 복제
→ 메모리: 16x → 6x
```

**ZeRO-2**: + Gradients shard
```python
# Optimizer state + Gradients shard
→ 메모리: 6x → 4x
```

**ZeRO-3**: + Parameters shard
```python
# 모든 것을 shard!
→ 메모리: 4x → 2x/N (N = GPU 수)
```

**DeepSpeed 사용**:
```python
import deepspeed

# Config
ds_config = {
    "train_batch_size": 128,
    "zero_optimization": {
        "stage": 3,  # ZeRO-3
        "offload_optimizer": {
            "device": "cpu",  # CPU로 offload
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu"
        }
    },
    "fp16": {
        "enabled": True
    }
}

# Initialize
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters()
)

# Training
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()

# 1T parameters on 64 GPUs 가능!
```

---

## 🔧 Gradient Accumulation

**목표**: GPU 메모리가 작아도 large batch training

```python
# 문제: Batch size 128이 안 들어감
# 해결: Micro-batch 32 × 4 accumulation

accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()  # Gradients 누적!
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Effective batch size = 32 × 4 = 128
```

**DDP + Gradient Accumulation**:
```python
model = DDP(model)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        with model.no_sync():  # Gradient 동기화 끄기
            loss = model(batch) / accumulation_steps
            loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            # 마지막에만 동기화
            loss = model(batch) / accumulation_steps
            loss.backward()  # Sync happens here
            optimizer.step()
            optimizer.zero_grad()
```

---

## 📊 통신 최적화

### All-Reduce

```python
# Naive: Ring All-Reduce
# GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0

# 최적화: Hierarchical All-Reduce
# 1. Node 내에서 all-reduce (NVLink: fast!)
# 2. Node 간 all-reduce (InfiniBand: slower)
# 3. Node 내에서 broadcast

# NCCL이 자동으로 최적화!
```

### Gradient Compression

```python
# 1-bit SGD
def compress_gradients(grad):
    # 부호만 전송!
    sign = torch.sign(grad)
    return sign

# Top-K sparsification
def topk_compress(grad, k=0.01):
    # 상위 1%만 전송
    values, indices = torch.topk(grad.abs().view(-1), 
                                  int(grad.numel() * k))
    return values, indices
```

---

## 🎓 실전 가이드

### 모델 크기별 전략

```
Small (< 1B parameters):
→ Single GPU or Data Parallelism

Medium (1B - 10B):
→ Data Parallelism + ZeRO-2

Large (10B - 100B):
→ ZeRO-3 + Tensor Parallelism

Huge (100B+):
→ ZeRO-3 + Tensor Parallelism + Pipeline Parallelism
   (3D parallelism)
```

### Debugging Tips

```python
# 1. Single GPU에서 먼저 테스트
python train.py

# 2. Single node multi-GPU
torchrun --nproc_per_node=4 train.py

# 3. Multi-node
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 --master_addr=... train.py

# 4. 각 단계별로 loss/memory 확인
```

---

## 📚 참고 자료

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [DeepSpeed](https://www.deepspeed.ai/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [GPipe Paper](https://arxiv.org/abs/1811.06965)

---

## ⏭️ 다음

Distributed Training을 마스터했습니다!

👉 [Model Serving](./model-serving.md)에서 **추론 최적화**를 배웁니다!

**"이제 1T 파라미터 모델도 훈련할 수 있습니다!"** 🚀
