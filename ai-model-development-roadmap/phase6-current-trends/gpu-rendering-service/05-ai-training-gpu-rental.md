# AI 학습용 GPU 클라우드 대여 서비스

## 목차
1. [AI 학습용 GPU 요구사항](#ai-학습용-gpu-요구사항)
2. [단일 GPU 학습 환경](#단일-gpu-학습-환경)
3. [분산 학습 아키텍처](#분산-학습-아키텍처)
4. [Multi-GPU 학습 (단일 노드)](#multi-gpu-학습-단일-노드)
5. [Multi-Node 분산 학습](#multi-node-분산-학습)
6. [Jupyter 환경 제공](#jupyter-환경-제공)
7. [가격 모델 및 최적화](#가격-모델-및-최적화)

---

## AI 학습용 GPU 요구사항

### 렌더링 vs AI 학습 GPU 사용 패턴 비교

```python
# GPU 사용 패턴 차이
usage_patterns = {
    "3D 렌더링": {
        "vram_usage": "매우 높음 (씬 데이터, 텍스처)",
        "compute_pattern": "병렬 레이 트레이싱",
        "duration": "수 분 ~ 수 시간 (작업당)",
        "memory_bandwidth": "높음",
        "tensor_core_usage": "중간 (AI 디노이징)",
        "typical_batch": "단일 프레임 또는 프레임 시퀀스",
        "io_pattern": "씬 로드 → 렌더 → 저장"
    },

    "AI 학습": {
        "vram_usage": "높음 (모델 + 배치 데이터)",
        "compute_pattern": "행렬 곱셈 (GEMM)",
        "duration": "수 시간 ~ 수 일 (에폭 반복)",
        "memory_bandwidth": "매우 높음",
        "tensor_core_usage": "매우 높음 (FP16/BF16)",
        "typical_batch": "수십~수천 샘플",
        "io_pattern": "연속적 데이터 스트리밍"
    }
}
```

### RTX A6000의 AI 학습 강점

```
┌─────────────────────────────────────────────────────────┐
│             RTX A6000 for AI Training                    │
├─────────────────────────────────────────────────────────┤
│                                                           │
│ 1. 대용량 VRAM (48GB)                                    │
│    - 큰 모델 학습 가능 (GPT-3 13B 등)                   │
│    - 큰 배치 사이즈 → 빠른 학습                         │
│                                                           │
│ 2. Tensor Core (3세대)                                   │
│    - FP16/BF16 Mixed Precision: 309.7 TFLOPS             │
│    - TF32 (TensorFloat32): 156 TFLOPS                    │
│    - INT8: 619 TOPS                                      │
│                                                           │
│ 3. NVLink (112.5 GB/s)                                   │
│    - 멀티 GPU 그래디언트 동기화 가속                     │
│    - 모델 병렬화 지원                                    │
│                                                           │
│ 4. ECC 메모리                                            │
│    - 장기 학습 안정성 (days~weeks)                       │
│    - 체크포인트 손상 방지                                │
│                                                           │
│ 5. 전문가용 드라이버                                     │
│    - CUDA 12.x, cuDNN 8.x 최적화                         │
│    - PyTorch, TensorFlow 안정성                          │
└─────────────────────────────────────────────────────────┘
```

### AI 워크로드별 GPU 요구사항

```python
class AIWorkloadRequirements:
    """AI 워크로드별 GPU 요구사항"""

    WORKLOADS = {
        "computer_vision": {
            "name": "컴퓨터 비전 (ResNet, EfficientNet)",
            "typical_batch_size": 64,
            "vram_requirement_gb": 16,
            "recommended_gpu": "RTX 3090 or better",
            "training_time_per_epoch": "10-30 minutes",
            "dataset_size": "10-100 GB"
        },

        "nlp_small": {
            "name": "소형 NLP (BERT-base, 110M params)",
            "typical_batch_size": 32,
            "vram_requirement_gb": 24,
            "recommended_gpu": "RTX 3090, A6000",
            "training_time_per_epoch": "1-3 hours",
            "dataset_size": "1-10 GB"
        },

        "nlp_large": {
            "name": "대형 NLP (GPT-3 13B params)",
            "typical_batch_size": 4,
            "vram_requirement_gb": 40,
            "recommended_gpu": "RTX A6000 (48GB)",
            "training_time_per_epoch": "hours-days",
            "dataset_size": "100+ GB",
            "requires_distributed": True
        },

        "diffusion_models": {
            "name": "Diffusion Models (Stable Diffusion)",
            "typical_batch_size": 8,
            "vram_requirement_gb": 32,
            "recommended_gpu": "RTX A6000",
            "training_time_per_epoch": "days",
            "dataset_size": "100+ GB (images)"
        },

        "reinforcement_learning": {
            "name": "강화학습 (PPO, SAC)",
            "typical_batch_size": 256,
            "vram_requirement_gb": 12,
            "recommended_gpu": "RTX 3060 or better",
            "training_time": "continuous (hours-days)",
            "special_requirements": "CPU도 중요 (환경 시뮬레이션)"
        }
    }

    @staticmethod
    def estimate_vram(model_params_millions, batch_size, sequence_length=512):
        """VRAM 요구사항 추정"""

        # 모델 파라미터 메모리 (FP16 기준)
        model_memory_gb = (model_params_millions * 1e6 * 2) / 1e9  # 2 bytes per param

        # 옵티마이저 상태 (Adam: 2배)
        optimizer_memory_gb = model_memory_gb * 2

        # 활성화 메모리 (배치 사이즈 비례)
        activation_memory_gb = (
            batch_size * sequence_length * model_params_millions * 2
        ) / 1e9

        # 그래디언트 메모리
        gradient_memory_gb = model_memory_gb

        total = (
            model_memory_gb +
            optimizer_memory_gb +
            activation_memory_gb +
            gradient_memory_gb
        )

        # 20% 오버헤드
        total_with_overhead = total * 1.2

        return {
            "model": model_memory_gb,
            "optimizer": optimizer_memory_gb,
            "activations": activation_memory_gb,
            "gradients": gradient_memory_gb,
            "total": total_with_overhead,
            "recommended_gpu": _recommend_gpu(total_with_overhead)
        }

# 사용 예시
# GPT-2 (117M parameters)
gpt2_vram = AIWorkloadRequirements.estimate_vram(
    model_params_millions=117,
    batch_size=8,
    sequence_length=1024
)
print(f"GPT-2 VRAM 요구사항: {gpt2_vram['total']:.2f} GB")
# 출력: ~6.5 GB → RTX 3060 12GB 충분

# GPT-3 13B parameters
gpt3_13b_vram = AIWorkloadRequirements.estimate_vram(
    model_params_millions=13000,
    batch_size=4,
    sequence_length=2048
)
print(f"GPT-3 13B VRAM 요구사항: {gpt3_13b_vram['total']:.2f} GB")
# 출력: ~42 GB → RTX A6000 48GB 필요
```

---

## 단일 GPU 학습 환경

### Docker 기반 PyTorch 환경

```dockerfile
# Dockerfile.pytorch-gpu
FROM nvidia/cuda:12.0-cudnn8-runtime-ubuntu22.04

# Python 및 기본 패키지
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 설치 (CUDA 12.0)
RUN pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu120

# 일반적인 ML 라이브러리
RUN pip3 install \
    transformers \
    accelerate \
    datasets \
    wandb \
    tensorboard \
    scikit-learn \
    pandas \
    matplotlib \
    jupyter

# 사용자 작업 디렉토리
WORKDIR /workspace
RUN chmod 777 /workspace

# Jupyter 포트
EXPOSE 8888

# 비특권 사용자로 실행
RUN useradd -m -u 1000 trainer
USER trainer

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
```

### Single GPU Training 예시

```python
# train_single_gpu.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel, AutoTokenizer
import wandb

class GPUTrainingManager:
    """단일 GPU 학습 관리"""

    def __init__(self, model, device_id=0):
        self.device = torch.device(f'cuda:{device_id}')
        self.model = model.to(self.device)

        # GPU 정보 출력
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")

    def train_epoch(self, dataloader, optimizer, criterion):
        """1 에폭 학습"""
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            # GPU로 데이터 전송
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # VRAM 사용량 모니터링
            if batch_idx % 100 == 0:
                vram_used = torch.cuda.memory_allocated() / 1e9
                vram_cached = torch.cuda.memory_reserved() / 1e9
                print(f"Batch {batch_idx}: Loss={loss.item():.4f}, "
                      f"VRAM: {vram_used:.2f}/{vram_cached:.2f} GB")

        return total_loss / len(dataloader)

# 사용 예시
model = AutoModel.from_pretrained("bert-base-uncased")
trainer = GPUTrainingManager(model, device_id=0)

# 데이터로더
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 옵티마이저
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# 학습
for epoch in range(10):
    loss = trainer.train_epoch(train_loader, optimizer, criterion)
    print(f"Epoch {epoch}: Loss={loss:.4f}")
```

### Mixed Precision Training (FP16)

```python
# mixed_precision_training.py
import torch
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    """Mixed Precision 학습 (2배 빠름, 50% VRAM 절약)"""

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.scaler = GradScaler()  # Loss scaling

    def train_step(self, data, target, optimizer, criterion):
        """Mixed precision training step"""

        # Automatic Mixed Precision
        with autocast():
            output = self.model(data)
            loss = criterion(output, target)

        # Scaled backward pass
        optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        return loss.item()

# 성능 비교
import time

# FP32 (기본)
model_fp32 = YourModel().cuda()
start = time.time()
for data, target in train_loader:
    output = model_fp32(data.cuda())
    loss = criterion(output, target.cuda())
    loss.backward()
fp32_time = time.time() - start

# FP16 (Mixed Precision)
model_fp16 = YourModel().cuda()
trainer = MixedPrecisionTrainer(model_fp16, 'cuda')
start = time.time()
for data, target in train_loader:
    trainer.train_step(data.cuda(), target.cuda(), optimizer, criterion)
fp16_time = time.time() - start

print(f"FP32: {fp32_time:.2f}s")
print(f"FP16: {fp16_time:.2f}s ({fp32_time/fp16_time:.1f}x speedup)")
# 예상 출력:
# FP32: 120.5s
# FP16: 58.3s (2.1x speedup)
```

---

## 분산 학습 아키텍처

### 분산 학습 개념

```
┌─────────────────────────────────────────────────────────┐
│           분산 학습 전략 3가지                           │
└─────────────────────────────────────────────────────────┘

1. 데이터 병렬화 (Data Parallelism)
───────────────────────────────────
┌──────────────────────────────────────────┐
│        Original Model (1 copy)           │
└──────────────────────────────────────────┘
                │
        ┌───────┼───────┐
        ▼       ▼       ▼
    ┌──────┐ ┌──────┐ ┌──────┐
    │ GPU0 │ │ GPU1 │ │ GPU2 │
    │Model │ │Model │ │Model │
    │Copy  │ │Copy  │ │Copy  │
    └──────┘ └──────┘ └──────┘
        │       │       │
    Batch1  Batch2  Batch3
        │       │       │
        └───────┼───────┘
                ▼
        Gradient Averaging
                ▼
        Parameter Update

- 가장 일반적
- 구현 간단
- 배치 크기 × GPU 개수


2. 모델 병렬화 (Model Parallelism)
───────────────────────────────────
    큰 모델을 여러 GPU에 분할

    ┌──────────────┐
    │  Layer 1-10  │ → GPU 0
    ├──────────────┤
    │  Layer 11-20 │ → GPU 1
    ├──────────────┤
    │  Layer 21-30 │ → GPU 2
    └──────────────┘

- 모델이 단일 GPU VRAM보다 클 때
- 파이프라인 병렬화 필요
- 통신 오버헤드 큼


3. 하이브리드 (Data + Model)
─────────────────────────────
    대규모 모델 (GPT-3 등)

    ┌─────────────┬─────────────┐
    │   GPU 0-1   │   GPU 2-3   │
    │ Layer 1-15  │ Layer 1-15  │ ← Model Parallel
    │   (Data     │   (Data     │
    │  Parallel)  │  Parallel)  │
    └─────────────┴─────────────┘
    │             │
    Batch 1-8     Batch 9-16      ← Data Parallel

- 최대 확장성
- 복잡한 구현
```

---

## Multi-GPU 학습 (단일 노드)

### PyTorch DistributedDataParallel (DDP)

```python
# train_multi_gpu.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """분산 환경 초기화"""
    # NCCL backend (GPU 간 통신)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, model, dataset, epochs):
    """각 GPU에서 실행되는 학습 함수"""

    print(f"[Rank {rank}] Starting training on GPU {rank}")

    # 1. 분산 환경 설정
    setup(rank, world_size)

    # 2. 모델을 현재 GPU로
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 3. DistributedSampler (각 GPU가 다른 데이터)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # 4. 옵티마이저
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    # 5. 학습 루프
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 셔플 시드 동기화

        ddp_model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            # 그래디언트 자동 동기화 (DDP가 처리)
            optimizer.step()

            total_loss += loss.item()

            if rank == 0 and batch_idx % 100 == 0:  # Master만 로그
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # 에폭 종료 - 체크포인트 저장 (Rank 0만)
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddp_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(dataloader),
            }, f'checkpoint_epoch_{epoch}.pt')

    cleanup()

def main():
    """메인 함수 - 여러 GPU 프로세스 생성"""

    world_size = torch.cuda.device_count()  # GPU 개수
    print(f"Training on {world_size} GPUs")

    # 모델 생성
    model = YourModel()

    # 데이터셋
    dataset = YourDataset()

    # 멀티프로세싱으로 각 GPU에 프로세스 생성
    mp.spawn(
        train_ddp,
        args=(world_size, model, dataset, 10),  # 10 epochs
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # 환경 변수 설정
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    main()
```

### NVLink 활용 최적화

```python
# nvlink_optimization.py
import torch
import torch.distributed as dist

class NVLinkOptimizedTraining:
    """NVLink 최적화 학습"""

    def __init__(self, gpus=[0, 1, 2, 3]):
        self.gpus = gpus
        self.num_gpus = len(gpus)

        # NVLink 토폴로지 확인
        self.check_nvlink_topology()

    def check_nvlink_topology(self):
        """NVLink 연결 확인"""
        import pynvml
        pynvml.nvmlInit()

        print("NVLink Topology:")
        for i, gpu_id in enumerate(self.gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            for j, other_gpu in enumerate(self.gpus):
                if i == j:
                    continue

                # NVLink 연결 상태
                try:
                    link_state = pynvml.nvmlDeviceGetNvLinkState(handle, j)
                    if link_state == pynvml.NVML_FEATURE_ENABLED:
                        print(f"  GPU {i} ↔ GPU {j}: NVLink Connected ✓")
                except:
                    print(f"  GPU {i} ↔ GPU {j}: PCIe only")

    def measure_bandwidth(self):
        """GPU 간 대역폭 측정"""
        # 큰 텐서 전송 벤치마크
        tensor_size = 1024 ** 3  # 1GB
        tensor = torch.randn(tensor_size // 4).cuda(self.gpus[0])

        import time

        # GPU 0 → GPU 1 전송
        start = time.time()
        tensor_copy = tensor.cuda(self.gpus[1])
        torch.cuda.synchronize()
        elapsed = time.time() - start

        bandwidth_gbps = (tensor_size / 1e9) / elapsed

        print(f"Measured bandwidth: {bandwidth_gbps:.2f} GB/s")

        if bandwidth_gbps > 50:
            print("→ NVLink detected! (>50 GB/s)")
        else:
            print("→ PCIe (typically 15-25 GB/s)")

        return bandwidth_gbps

# 사용
optimizer = NVLinkOptimizedTraining(gpus=[0, 1, 2, 3])
optimizer.measure_bandwidth()
```

### Gradient Accumulation (큰 배치 시뮬레이션)

```python
# gradient_accumulation.py

def train_with_gradient_accumulation(
    model,
    dataloader,
    optimizer,
    accumulation_steps=4
):
    """
    Gradient Accumulation으로 효과적인 배치 크기 증가

    실제 배치: 32
    accumulation_steps: 4
    → 효과적 배치: 128 (32 × 4)
    """

    model.train()
    optimizer.zero_grad()

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Normalize loss (accumulation 고려)
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Accumulation steps마다 옵티마이저 업데이트
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            print(f"Updated weights at batch {batch_idx} "
                  f"(effective batch size: {32 * accumulation_steps})")

# VRAM 부족 시 해결책
# 단일 GPU VRAM: 24GB
# 희망 배치: 128 → VRAM 초과!
# 해결: 배치 32 × accumulation 4 = 효과적 128
```

---

## Multi-Node 분산 학습

### Multi-Node 아키텍처

```
┌───────────────────────────────────────────────────────────┐
│            Multi-Node Distributed Training                 │
└───────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────┐
        │      Master Node (Rank 0)           │
        │  - Parameter Server (optional)       │
        │  - Gradient Aggregation             │
        │  - Checkpoint Saving                │
        └─────────────────────────────────────┘
                │
                │ Ethernet / InfiniBand
                │
        ┌───────┼───────┬───────┬───────┐
        │       │       │       │       │
    ┌───▼───┐ ┌▼──────┐ ┌▼──────┐ ┌──▼────┐
    │ Node 0│ │ Node 1│ │ Node 2│ │ Node 3│
    │       │ │       │ │       │ │       │
    │4xA6000│ │4xA6000│ │4xA6000│ │4xA6000│
    └───────┘ └───────┘ └───────┘ └───────┘
       GPU      GPU      GPU      GPU
       0-3      4-7      8-11     12-15

Total: 16 GPUs, 768GB VRAM
```

### PyTorch Multi-Node DDP

```python
# train_multi_node.py
import torch
import torch.distributed as dist
import argparse

def setup_multi_node(rank, world_size, master_addr, master_port):
    """Multi-node 분산 환경 설정"""

    # 환경 변수
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # NCCL 초기화
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    print(f"[Rank {rank}] Connected to master {master_addr}:{master_port}")
    print(f"[Rank {rank}] World size: {world_size}")

def train_multi_node(local_rank, args):
    """각 노드에서 실행"""

    # Global rank 계산
    # local_rank: 노드 내 GPU ID (0-3)
    # node_rank: 노드 ID (0, 1, 2, 3)
    # world_size: 전체 GPU 개수 (16)

    global_rank = args.node_rank * args.gpus_per_node + local_rank

    setup_multi_node(
        rank=global_rank,
        world_size=args.world_size,
        master_addr=args.master_addr,
        master_port=args.master_port
    )

    # GPU 설정
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # 모델
    model = YourLargeModel().to(device)
    model = DDP(model, device_ids=[local_rank])

    # 데이터
    sampler = DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=global_rank
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4
    )

    # 학습
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        for batch in dataloader:
            data, target = batch
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        # Master 노드만 체크포인트 저장
        if global_rank == 0:
            torch.save(model.module.state_dict(), f'checkpoint_{epoch}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-rank', type=int, required=True)
    parser.add_argument('--gpus-per-node', type=int, default=4)
    parser.add_argument('--world-size', type=int, default=16)
    parser.add_argument('--master-addr', type=str, required=True)
    parser.add_argument('--master-port', type=str, default='12355')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()

    # 각 GPU에 프로세스 생성
    mp.spawn(
        train_multi_node,
        args=(args,),
        nprocs=args.gpus_per_node,
        join=True
    )
```

### 실행 방법 (각 노드에서)

```bash
# Node 0 (Master)
python train_multi_node.py \
    --node-rank 0 \
    --gpus-per-node 4 \
    --world-size 16 \
    --master-addr 192.168.1.100 \
    --master-port 12355

# Node 1
python train_multi_node.py \
    --node-rank 1 \
    --gpus-per-node 4 \
    --world-size 16 \
    --master-addr 192.168.1.100 \
    --master-port 12355

# Node 2
python train_multi_node.py \
    --node-rank 2 \
    --gpus-per-node 4 \
    --world-size 16 \
    --master-addr 192.168.1.100 \
    --master-port 12355

# Node 3
python train_multi_node.py \
    --node-rank 3 \
    --gpus-per-node 4 \
    --world-size 16 \
    --master-addr 192.168.1.100 \
    --master-port 12355
```

---

## Jupyter 환경 제공

### JupyterHub with GPU 할당

```python
# jupyterhub_config.py
import os

c = get_config()

# GPU 할당 전략
class GPUSpawner(DockerSpawner):
    def start(self):
        # 사용자당 GPU 할당
        user = self.user.name
        gpu_id = self.get_user_gpu(user)

        self.extra_host_config = {
            'runtime': 'nvidia',
            'environment': {
                'CUDA_VISIBLE_DEVICES': str(gpu_id)
            }
        }

        return super().start()

    def get_user_gpu(self, username):
        # Redis에서 사용자 GPU 할당 정보 가져오기
        import redis
        r = redis.Redis()

        gpu_id = r.get(f'user:{username}:gpu')
        if gpu_id:
            return int(gpu_id)

        # 사용 가능한 GPU 찾기
        for i in range(4):
            if not r.exists(f'gpu:{i}:user'):
                r.setex(f'gpu:{i}:user', 3600, username)  # 1시간
                r.setex(f'user:{username}:gpu', 3600, i)
                return i

        raise Exception("No available GPU")

c.JupyterHub.spawner_class = GPUSpawner
```

### Pre-built 노트북 템플릿

```python
# template_single_gpu_training.ipynb
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Single GPU Training Template\n",
    "이 노트북은 단일 GPU 학습을 위한 템플릿입니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. GPU 확인\n",
    "import torch\n",
    "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
    "print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
    "print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. 데이터 로드\n",
    "from torch.utils.data import DataLoader\n",
    "from torchvision import datasets, transforms\n",
    "\n",
    "transform = transforms.Compose([\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize((0.5,), (0.5,))\n",
    "])\n",
    "\n",
    "train_dataset = datasets.CIFAR10(\n",
    "    root='./data',\n",
    "    train=True,\n",
    "    download=True,\n",
    "    transform=transform\n",
    ")\n",
    "\n",
    "train_loader = DataLoader(\n",
    "    train_dataset,\n",
    "    batch_size=128,\n",
    "    shuffle=True,\n",
    "    num_workers=4\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. 모델 정의\n",
    "import torch.nn as nn\n",
    "\n",
    "model = YourModel().cuda()\n",
    "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)\n",
    "criterion = nn.CrossEntropyLoss()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. 학습 루프\n",
    "from tqdm.notebook import tqdm\n",
    "\n",
    "epochs = 10\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    model.train()\n",
    "    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')\n",
    "    \n",
    "    for data, target in progress_bar:\n",
    "        data, target = data.cuda(), target.cuda()\n",
    "        \n",
    "        optimizer.zero_grad()\n",
    "        output = model(data)\n",
    "        loss = criterion(output, target)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        \n",
    "        progress_bar.set_postfix({'loss': loss.item()})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. 체크포인트 저장\n",
    "torch.save({\n",
    "    'model_state_dict': model.state_dict(),\n",
    "    'optimizer_state_dict': optimizer.state_dict(),\n",
    "}, '/workspace/checkpoint.pt')"
   ]
  }
 ]
}
```

---

## 가격 모델 및 최적화

### AI 학습용 GPU 대여 가격표

```python
class AITrainingPricing:
    """AI 학습용 GPU 대여 가격"""

    PRICING = {
        # 단일 GPU
        "rtx_3060_12gb": {
            "hourly": 0.50,
            "daily": 10.00,
            "weekly": 60.00,
            "monthly": 200.00,
            "typical_use": "소규모 CV, 실험"
        },

        "rtx_3090_24gb": {
            "hourly": 1.20,
            "daily": 24.00,
            "weekly": 150.00,
            "monthly": 500.00,
            "typical_use": "중규모 NLP, 학위 논문"
        },

        "rtx_a6000_48gb": {
            "hourly": 2.50,
            "daily": 50.00,
            "weekly": 320.00,
            "monthly": 1100.00,
            "typical_use": "대형 모델, 프로덕션"
        },

        # Multi-GPU (NVLink)
        "2x_a6000_nvlink": {
            "hourly": 4.50,  # 10% 할인
            "daily": 90.00,
            "weekly": 580.00,
            "monthly": 2000.00,
            "typical_use": "분산 학습, 큰 배치"
        },

        "4x_a6000_nvlink": {
            "hourly": 8.50,  # 15% 할인
            "daily": 170.00,
            "weekly": 1100.00,
            "monthly": 3800.00,
            "typical_use": "대규모 분산 학습"
        },

        # Spot Instance (저렴, 중단 가능)
        "spot_a6000": {
            "hourly": 1.00,  # 60% 할인
            "max_duration_hours": 24,
            "preemptible": True,
            "typical_use": "체크포인트 가능한 학습"
        },

        # Jupyter 환경 추가 비용
        "jupyter_addon": {
            "hourly": 0.10,
            "includes": ["JupyterLab", "VSCode Server", "TensorBoard"]
        },

        # 스토리지
        "storage_gb_month": 0.10,  # $0.10/GB/month
    }

    @staticmethod
    def estimate_cost(
        gpu_type: str,
        hours: float,
        storage_gb: int = 0,
        jupyter: bool = False
    ) -> dict:
        """비용 추정"""

        pricing = AITrainingPricing.PRICING[gpu_type]

        # GPU 비용
        gpu_cost = pricing["hourly"] * hours

        # Jupyter 비용
        jupyter_cost = 0
        if jupyter:
            jupyter_cost = AITrainingPricing.PRICING["jupyter_addon"]["hourly"] * hours

        # 스토리지 비용 (시간 비례)
        storage_cost = (storage_gb * AITrainingPricing.PRICING["storage_gb_month"]) * (hours / 720)

        total = gpu_cost + jupyter_cost + storage_cost

        return {
            "gpu_cost": gpu_cost,
            "jupyter_cost": jupyter_cost,
            "storage_cost": storage_cost,
            "total": total,
            "hourly_rate": total / hours
        }

# 사용 예시
# GPT-2 학습 (RTX 3090, 48시간)
cost = AITrainingPricing.estimate_cost(
    gpu_type="rtx_3090_24gb",
    hours=48,
    storage_gb=100,
    jupyter=True
)
print(f"GPT-2 학습 비용 추정: ${cost['total']:.2f}")
# 출력: ~$62.88

# Stable Diffusion 학습 (4x A6000, 1주일)
cost_sd = AITrainingPricing.estimate_cost(
    gpu_type="4x_a6000_nvlink",
    hours=168,  # 1 week
    storage_gb=500,
    jupyter=True
)
print(f"Stable Diffusion 학습 비용: ${cost_sd['total']:.2f}")
# 출력: ~$1,445
```

### 비용 최적화 전략

```python
class CostOptimizer:
    """학습 비용 최적화"""

    @staticmethod
    def recommend_strategy(
        model_size_gb: float,
        dataset_size_gb: float,
        deadline_days: float,
        budget_usd: float
    ) -> dict:
        """최적 전략 추천"""

        strategies = []

        # 전략 1: Spot Instance (저렴, 중단 가능)
        if deadline_days >= 3:  # 여유 있으면
            spot_hours = deadline_days * 24
            spot_cost = AITrainingPricing.estimate_cost(
                "spot_a6000",
                spot_hours,
                dataset_size_gb
            )

            if spot_cost["total"] <= budget_usd:
                strategies.append({
                    "name": "Spot Instance",
                    "gpu": "RTX A6000 (Spot)",
                    "cost": spot_cost["total"],
                    "pros": ["60% 저렴", "48GB VRAM"],
                    "cons": ["중단 가능", "체크포인트 필수"],
                    "recommended": deadline_days >= 5
                })

        # 전략 2: On-demand 단일 GPU
        ondemand_hours = deadline_days * 24
        ondemand_cost = AITrainingPricing.estimate_cost(
            "rtx_a6000_48gb",
            ondemand_hours,
            dataset_size_gb,
            jupyter=True
        )

        if ondemand_cost["total"] <= budget_usd:
            strategies.append({
                "name": "On-demand Single GPU",
                "gpu": "RTX A6000",
                "cost": ondemand_cost["total"],
                "pros": ["안정적", "중단 없음"],
                "cons": ["비쌈"],
                "recommended": deadline_days <= 3
            })

        # 전략 3: Multi-GPU (빠르지만 비쌈)
        multi_hours = (deadline_days * 24) / 3.5  # 3.5배 빠름
        multi_cost = AITrainingPricing.estimate_cost(
            "4x_a6000_nvlink",
            multi_hours,
            dataset_size_gb,
            jupyter=True
        )

        if multi_cost["total"] <= budget_usd * 1.2:  # 20% 초과 허용
            strategies.append({
                "name": "Multi-GPU Distributed",
                "gpu": "4x RTX A6000 NVLink",
                "cost": multi_cost["total"],
                "pros": ["3.5배 빠름", "192GB VRAM"],
                "cons": ["비쌈", "복잡함"],
                "recommended": deadline_days <= 1
            })

        # 최적 전략 선택 (비용/시간 비율)
        best = min(strategies, key=lambda s: s["cost"] / (deadline_days * 24))

        return {
            "recommended": best,
            "alternatives": [s for s in strategies if s != best]
        }

# 사용 예시
optimizer = CostOptimizer()

# GPT-2 Fine-tuning 시나리오
recommendation = optimizer.recommend_strategy(
    model_size_gb=5,
    dataset_size_gb=20,
    deadline_days=7,
    budget_usd=200
)

print("추천 전략:", recommendation["recommended"]["name"])
print(f"예상 비용: ${recommendation['recommended']['cost']:.2f}")
print("장점:", recommendation["recommended"]["pros"])
```

---

## 핵심 요약

### AI 학습 vs 렌더링 GPU 사용

| 특성 | 렌더링 | AI 학습 |
|------|--------|---------|
| **VRAM 사용** | 매우 높음 (씬) | 높음 (모델+데이터) |
| **작업 시간** | 분~시간 | 시간~일 |
| **Tensor Core** | 중간 (디노이징) | 필수 (훈련) |
| **분산 필요성** | 프레임 분할 | 대형 모델 필수 |
| **안정성 요구** | 높음 | 매우 높음 (체크포인트) |

### RTX A6000 강점 (AI 학습)

1. **48GB VRAM** - GPT-3 13B 학습 가능
2. **Tensor Core** - FP16 훈련 2배 빠름
3. **NVLink** - 멀티 GPU 그래디언트 동기화 가속
4. **ECC** - 장기 학습 안정성 (일주일+)

### 가격 경쟁력

- RTX A6000: $2.50/hour (48GB)
- AWS p4d.24xlarge: $32.77/hour (8x A100, 40GB each)
- **비용 대비 성능**: RTX A6000이 유리 (중소규모)

**다음**: 프로덕션 레벨 구현 가이드 →
