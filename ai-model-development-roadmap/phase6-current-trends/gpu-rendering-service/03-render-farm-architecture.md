# 클라우드 GPU 렌더 팜 아키텍처 설계

## 목차
1. [전체 시스템 아키텍처](#전체-시스템-아키텍처)
2. [핵심 컴포넌트](#핵심-컴포넌트)
3. [작업 스케줄링 시스템](#작업-스케줄링-시스템)
4. [분산 렌더링 전략](#분산-렌더링-전략)
5. [네트워크 및 스토리지](#네트워크-및-스토리지)
6. [모니터링 및 자동 스케일링](#모니터링-및-자동-스케일링)
7. [보안 및 격리](#보안-및-격리)

---

## 전체 시스템 아키텍처

### High-Level 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      Cloud GPU Render Farm                       │
└─────────────────────────────────────────────────────────────────┘

                         [User/Client]
                              │
                              │ HTTPS
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         API Gateway                               │
│  - Authentication/Authorization                                   │
│  - Rate Limiting                                                  │
│  - Load Balancing                                                 │
└──────────────────────────────────────────────────────────────────┘
                              │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────┐  ┌───────────────────┐  ┌────────────────┐
│   Web UI    │  │   REST API        │  │  File Upload   │
│  Dashboard  │  │   /jobs           │  │  Service       │
└─────────────┘  │   /pricing        │  └────────────────┘
                 │   /status         │           │
                 └───────────────────┘           │
                         │                       │
                         ▼                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Job Queue (Redis/RabbitMQ)                     │
│  Priority Queues: [High] [Normal] [Low] [Spot]                   │
└──────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Job Scheduler & Orchestrator                    │
│  - GPU Resource Manager                                           │
│  - Task Distribution                                              │
│  - Load Balancing                                                 │
│  - Auto-scaling                                                   │
└──────────────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Render Node  │  │ Render Node  │  │ Render Node  │
│ 4x RTX A6000 │  │ 4x RTX A6000 │  │ 4x RTX A6000 │
│ (48GB each)  │  │ (48GB each)  │  │ (48GB each)  │
│              │  │              │  │              │
│ - Blender    │  │ - V-Ray      │  │ - Octane     │
│ - Cycles     │  │ - Corona     │  │ - Redshift   │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │                │
        └────────────────┴────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│             Shared Storage (NFS/Ceph/S3)                          │
│  - Scene Files                                                    │
│  - Asset Library (Textures, Models)                               │
│  - Rendered Outputs                                               │
│  - Cache                                                          │
└──────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│              Monitoring & Metrics (Prometheus/Grafana)            │
│  - GPU Utilization                                                │
│  - Render Queue Depth                                             │
│  - Cost Tracking                                                  │
│  - SLA Monitoring                                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 핵심 컴포넌트

### 1. API Gateway

```python
# api_gateway.py
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional, List
import redis
import boto3
import uuid

app = FastAPI(title="Cloud GPU Render Farm API")

# Redis for job queue
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# S3 for file storage
s3 = boto3.client('s3')
BUCKET_NAME = 'render-farm-assets'

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class RenderJobRequest(BaseModel):
    """렌더 작업 요청"""
    scene_file_url: str
    output_format: str = "PNG"
    resolution: tuple[int, int] = (1920, 1080)
    samples: int = 128
    priority: str = "normal"  # high, normal, low, spot
    gpu_tier: str = "studio"  # hobby, pro, studio, enterprise
    frame_range: Optional[tuple[int, int]] = None  # 애니메이션용
    notification_webhook: Optional[str] = None

class RenderJobResponse(BaseModel):
    job_id: str
    status: str
    estimated_cost: float
    estimated_time_minutes: float
    queue_position: int

@app.post("/api/v1/jobs/submit", response_model=RenderJobResponse)
async def submit_render_job(
    job: RenderJobRequest,
    token: str = Depends(oauth2_scheme)
):
    """새 렌더 작업 제출"""

    # 1. 사용자 인증 및 크레딧 확인
    user = authenticate_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    # 2. 비용 계산
    pricing = calculate_pricing(job)

    if user.credits < pricing['estimated_cost']:
        raise HTTPException(status_code=402, detail="Insufficient credits")

    # 3. 작업 ID 생성
    job_id = str(uuid.uuid4())

    # 4. Job 메타데이터 저장
    job_data = {
        "job_id": job_id,
        "user_id": user.id,
        "status": "queued",
        "scene_file": job.scene_file_url,
        "resolution": job.resolution,
        "samples": job.samples,
        "priority": job.priority,
        "gpu_tier": job.gpu_tier,
        "frame_range": job.frame_range,
        "estimated_cost": pricing['estimated_cost'],
        "estimated_time": pricing['estimated_time'],
        "created_at": datetime.utcnow().isoformat()
    }

    # 5. Redis Queue에 추가
    queue_name = f"render_queue:{job.priority}"
    redis_client.rpush(queue_name, json.dumps(job_data))

    # 6. DB에 작업 기록
    await db.jobs.insert_one(job_data)

    # 7. 큐 위치 계산
    queue_position = redis_client.llen(queue_name)

    return RenderJobResponse(
        job_id=job_id,
        status="queued",
        estimated_cost=pricing['estimated_cost'],
        estimated_time_minutes=pricing['estimated_time'],
        queue_position=queue_position
    )

def calculate_pricing(job: RenderJobRequest) -> dict:
    """렌더 비용 및 시간 계산"""

    # GPU 티어별 시간당 요금
    tier_pricing = {
        "hobby": 0.75,      # RTX 3060
        "pro": 1.50,        # RTX 3090
        "studio": 3.00,     # RTX A6000
        "enterprise": 10.00 # 4x RTX A6000
    }

    hourly_rate = tier_pricing[job.gpu_tier]

    # 렌더 시간 추정 (해상도 + 샘플 수 기반)
    pixels = job.resolution[0] * job.resolution[1]
    base_time_seconds = (pixels / 1_000_000) * (job.samples / 100) * 30

    # 프레임 수
    if job.frame_range:
        num_frames = job.frame_range[1] - job.frame_range[0] + 1
    else:
        num_frames = 1

    total_time_seconds = base_time_seconds * num_frames

    # GPU 티어별 속도 보정
    speed_multiplier = {
        "hobby": 1.0,
        "pro": 0.5,       # 2배 빠름
        "studio": 0.4,    # 2.5배 빠름
        "enterprise": 0.15 # 6.7배 빠름
    }

    adjusted_time = total_time_seconds * speed_multiplier[job.gpu_tier]
    hours = adjusted_time / 3600

    estimated_cost = hours * hourly_rate

    return {
        "estimated_time": adjusted_time / 60,  # minutes
        "estimated_cost": round(estimated_cost, 2)
    }

@app.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(job_id: str, token: str = Depends(oauth2_scheme)):
    """작업 상태 조회"""

    job = await db.jobs.find_one({"job_id": job_id})

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "status": job["status"],  # queued, rendering, completed, failed
        "progress": job.get("progress", 0),  # 0-100
        "current_frame": job.get("current_frame"),
        "output_url": job.get("output_url"),
        "error": job.get("error")
    }

@app.post("/api/v1/upload/scene")
async def upload_scene_file(file: UploadFile = File(...)):
    """씬 파일 업로드"""

    # S3에 업로드
    file_key = f"scenes/{uuid.uuid4()}/{file.filename}"

    s3.upload_fileobj(
        file.file,
        BUCKET_NAME,
        file_key,
        ExtraArgs={'ContentType': file.content_type}
    )

    # 서명된 URL 생성 (24시간 유효)
    url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': file_key},
        ExpiresIn=86400
    )

    return {"file_url": url, "file_key": file_key}
```

### 2. Job Scheduler

```python
# job_scheduler.py
import redis
import asyncio
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class GPUResourceManager:
    """GPU 리소스 관리"""

    def __init__(self):
        self.nodes = self._discover_render_nodes()
        self.gpu_status = {}  # node_id -> gpu_id -> status

    def _discover_render_nodes(self) -> List[Dict]:
        """렌더 노드 자동 발견 (Kubernetes Service Discovery)"""
        # In production, use K8s API or Consul
        return [
            {
                "node_id": "node-001",
                "hostname": "render-node-001",
                "gpus": [
                    {"gpu_id": 0, "model": "RTX A6000", "vram_gb": 48},
                    {"gpu_id": 1, "model": "RTX A6000", "vram_gb": 48},
                    {"gpu_id": 2, "model": "RTX A6000", "vram_gb": 48},
                    {"gpu_id": 3, "model": "RTX A6000", "vram_gb": 48},
                ],
                "status": "online"
            },
            # ... more nodes
        ]

    async def find_available_gpu(self, required_vram_gb: int, tier: str) -> Dict:
        """사용 가능한 GPU 찾기"""

        for node in self.nodes:
            if node["status"] != "online":
                continue

            for gpu in node["gpus"]:
                gpu_key = f"{node['node_id']}:{gpu['gpu_id']}"

                # GPU 사용 상태 확인
                if redis_client.get(f"gpu:{gpu_key}:status") == "busy":
                    continue

                # VRAM 용량 확인
                if gpu["vram_gb"] >= required_vram_gb:
                    # GPU 예약
                    redis_client.setex(
                        f"gpu:{gpu_key}:status",
                        3600,  # 1시간 타임아웃
                        "busy"
                    )

                    return {
                        "node_id": node["node_id"],
                        "gpu_id": gpu["gpu_id"],
                        "hostname": node["hostname"],
                        "vram_gb": gpu["vram_gb"]
                    }

        return None  # 사용 가능한 GPU 없음

    async def release_gpu(self, node_id: str, gpu_id: int):
        """GPU 해제"""
        gpu_key = f"{node_id}:{gpu_id}"
        redis_client.delete(f"gpu:{gpu_key}:status")


class RenderJobScheduler:
    """렌더 작업 스케줄러"""

    def __init__(self):
        self.redis_client = redis.Redis()
        self.gpu_manager = GPUResourceManager()

    async def run(self):
        """스케줄러 메인 루프"""

        logger.info("Starting render job scheduler...")

        while True:
            # 우선순위 큐 순서대로 처리
            for priority in ["high", "normal", "low", "spot"]:
                queue_name = f"render_queue:{priority}"

                # Queue에서 작업 하나 가져오기
                job_data = self.redis_client.lpop(queue_name)

                if job_data:
                    job = json.loads(job_data)
                    await self.process_job(job)

            # 1초 대기
            await asyncio.sleep(1)

    async def process_job(self, job: Dict):
        """작업 처리"""

        logger.info(f"Processing job {job['job_id']}")

        # 1. 필요한 VRAM 추정
        required_vram = estimate_vram_requirement(job)

        # 2. 사용 가능한 GPU 찾기
        gpu = await self.gpu_manager.find_available_gpu(
            required_vram,
            job["gpu_tier"]
        )

        if not gpu:
            # GPU 없음 - 다시 큐에 넣기
            logger.warning(f"No available GPU for job {job['job_id']}")
            queue_name = f"render_queue:{job['priority']}"
            self.redis_client.rpush(queue_name, json.dumps(job))
            return

        # 3. 렌더 노드에 작업 할당
        try:
            await self.assign_job_to_node(job, gpu)

            # 4. 작업 상태 업데이트
            await db.jobs.update_one(
                {"job_id": job["job_id"]},
                {"$set": {
                    "status": "rendering",
                    "assigned_node": gpu["node_id"],
                    "assigned_gpu": gpu["gpu_id"],
                    "started_at": datetime.utcnow()
                }}
            )

        except Exception as e:
            logger.error(f"Error assigning job: {e}")
            # GPU 해제
            await self.gpu_manager.release_gpu(
                gpu["node_id"],
                gpu["gpu_id"]
            )

    async def assign_job_to_node(self, job: Dict, gpu: Dict):
        """렌더 노드에 작업 할당"""

        # gRPC 또는 HTTP로 렌더 노드에 작업 전송
        import grpc
        from protos import render_pb2, render_pb2_grpc

        channel = grpc.aio.insecure_channel(
            f"{gpu['hostname']}:50051"
        )
        stub = render_pb2_grpc.RenderServiceStub(channel)

        request = render_pb2.RenderRequest(
            job_id=job["job_id"],
            scene_file=job["scene_file"],
            resolution=job["resolution"],
            samples=job["samples"],
            output_format=job.get("output_format", "PNG"),
            gpu_id=gpu["gpu_id"]
        )

        response = await stub.StartRender(request)

        logger.info(f"Job {job['job_id']} assigned to {gpu['hostname']}:GPU{gpu['gpu_id']}")


def estimate_vram_requirement(job: Dict) -> int:
    """VRAM 요구사항 추정"""

    # 해상도 기반 추정
    pixels = job["resolution"][0] * job["resolution"][1]

    # 기본 VRAM (GB)
    base_vram = 4

    # 해상도당 VRAM
    resolution_vram = (pixels / 1_000_000) * 2  # 2GB per megapixel

    # 샘플당 VRAM
    sample_vram = (job["samples"] / 100) * 1  # 1GB per 100 samples

    total = base_vram + resolution_vram + sample_vram

    return int(total)
```

---

## 작업 스케줄링 시스템

### 스케줄링 전략

```python
class SmartScheduler:
    """비용 최적화 스마트 스케줄러"""

    def __init__(self):
        self.predictor = RenderTimePredictor()  # ML 모델

    async def optimize_job_assignment(self, job: Dict) -> Dict:
        """최적의 GPU 할당 전략"""

        # 1. 렌더 시간 예측
        predicted_time = await self.predictor.predict(job)

        # 2. 비용 효율성 계산
        strategies = [
            self._single_gpu_strategy(job, predicted_time),
            self._multi_gpu_strategy(job, predicted_time),
            self._spot_instance_strategy(job, predicted_time)
        ]

        # 3. 최적 전략 선택
        best = min(strategies, key=lambda s: s['cost'])

        return best

    def _single_gpu_strategy(self, job, predicted_time):
        """단일 GPU 전략"""
        return {
            "strategy": "single_gpu",
            "gpus": 1,
            "cost": predicted_time * 3.00,  # $3/hour
            "time": predicted_time
        }

    def _multi_gpu_strategy(self, job, predicted_time):
        """멀티 GPU 전략 (프레임 분할)"""
        if not job.get("frame_range"):
            return {"cost": float('inf')}  # 단일 프레임은 불가

        num_frames = job["frame_range"][1] - job["frame_range"][0] + 1

        if num_frames < 4:
            return {"cost": float('inf')}

        # 4 GPU로 분산
        time_per_frame = predicted_time / num_frames
        parallel_time = (num_frames / 4) * time_per_frame

        return {
            "strategy": "multi_gpu",
            "gpus": 4,
            "cost": parallel_time * 10.00,  # $10/hour for 4 GPUs
            "time": parallel_time
        }

    def _spot_instance_strategy(self, job, predicted_time):
        """Spot Instance 전략 (저렴하지만 중단 가능)"""

        if job["priority"] == "high":
            return {"cost": float('inf')}  # High priority는 spot 불가

        # Spot price: 70% 할인
        spot_price = 3.00 * 0.30  # $0.90/hour

        # 중단 리스크 고려 (+20% 시간)
        adjusted_time = predicted_time * 1.20

        return {
            "strategy": "spot",
            "gpus": 1,
            "cost": adjusted_time * spot_price,
            "time": adjusted_time,
            "interruptible": True
        }
```

---

## 분산 렌더링 전략

### Frame-based Distribution

```python
class FrameDistributor:
    """프레임 기반 분산 렌더링"""

    def distribute_frames(self, job: Dict, num_gpus: int) -> List[Dict]:
        """프레임을 여러 GPU에 분배"""

        start, end = job["frame_range"]
        total_frames = end - start + 1

        frames_per_gpu = total_frames // num_gpus
        remainder = total_frames % num_gpus

        tasks = []
        current_frame = start

        for i in range(num_gpus):
            # 나머지 프레임 분배
            chunk_size = frames_per_gpu + (1 if i < remainder else 0)

            tasks.append({
                "task_id": f"{job['job_id']}_chunk_{i}",
                "frame_start": current_frame,
                "frame_end": current_frame + chunk_size - 1,
                "gpu_id": i
            })

            current_frame += chunk_size

        return tasks

# 사용 예시
distributor = FrameDistributor()

job = {
    "job_id": "job-123",
    "frame_range": (1, 100),  # 100 프레임
    # ...
}

tasks = distributor.distribute_frames(job, num_gpus=4)
# 결과:
# GPU 0: frames 1-25
# GPU 1: frames 26-50
# GPU 2: frames 51-75
# GPU 3: frames 76-100
```

### Tile-based Distribution (단일 프레임)

```python
class TileRenderer:
    """타일 기반 분산 렌더링 (단일 고해상도 프레임)"""

    def split_into_tiles(
        self,
        resolution: tuple,
        num_gpus: int,
        tile_overlap: int = 64
    ) -> List[Dict]:
        """이미지를 타일로 분할"""

        width, height = resolution

        # 그리드 계산
        cols = math.ceil(math.sqrt(num_gpus))
        rows = math.ceil(num_gpus / cols)

        tile_width = width // cols
        tile_height = height // rows

        tiles = []

        for row in range(rows):
            for col in range(cols):
                if len(tiles) >= num_gpus:
                    break

                # 타일 경계 (오버랩 포함)
                x_start = max(0, col * tile_width - tile_overlap)
                y_start = max(0, row * tile_height - tile_overlap)
                x_end = min(width, (col + 1) * tile_width + tile_overlap)
                y_end = min(height, (row + 1) * tile_height + tile_overlap)

                tiles.append({
                    "tile_id": len(tiles),
                    "region": (x_start, y_start, x_end, y_end),
                    "crop": (
                        tile_overlap if col > 0 else 0,
                        tile_overlap if row > 0 else 0,
                        tile_overlap if col < cols - 1 else 0,
                        tile_overlap if row < rows - 1 else 0
                    )
                })

        return tiles

# 8K 이미지를 4개 GPU로 분할
tiler = TileRenderer()
tiles = tiler.split_into_tiles(
    resolution=(7680, 4320),
    num_gpus=4
)

# 결과:
# Tile 0: (0, 0, 3904, 2224)      - 좌상
# Tile 1: (3776, 0, 7680, 2224)   - 우상
# Tile 2: (0, 2096, 3904, 4320)   - 좌하
# Tile 3: (3776, 2096, 7680, 4320) - 우하
```

---

## 네트워크 및 스토리지

### 고성능 스토리지 시스템

```python
# storage_manager.py
import boto3
from cachetools import LRUCache

class RenderAssetManager:
    """렌더 에셋 관리 (S3 + Local Cache)"""

    def __init__(self):
        self.s3 = boto3.client('s3')
        self.local_cache = LRUCache(maxsize=100)  # 100 GB cache
        self.cache_dir = "/mnt/nvme-cache"  # NVMe SSD

    async def get_scene_file(self, s3_key: str) -> str:
        """씬 파일 가져오기 (캐시 우선)"""

        # 1. 로컬 캐시 확인
        if s3_key in self.local_cache:
            logger.info(f"Cache hit: {s3_key}")
            return self.local_cache[s3_key]

        # 2. S3에서 다운로드
        local_path = f"{self.cache_dir}/{s3_key}"

        logger.info(f"Downloading from S3: {s3_key}")
        self.s3.download_file(
            'render-farm-assets',
            s3_key,
            local_path
        )

        # 3. 캐시에 추가
        self.local_cache[s3_key] = local_path

        return local_path

    async def upload_result(self, local_path: str, job_id: str) -> str:
        """렌더 결과 업로드"""

        s3_key = f"outputs/{job_id}/{os.path.basename(local_path)}"

        self.s3.upload_file(
            local_path,
            'render-farm-results',
            s3_key,
            ExtraArgs={
                'ContentType': 'image/png',
                'StorageClass': 'INTELLIGENT_TIERING'  # 비용 최적화
            }
        )

        # 서명된 URL 생성 (7일 유효)
        url = self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': 'render-farm-results', 'Key': s3_key},
            ExpiresIn=604800
        )

        return url
```

---

## 모니터링 및 자동 스케일링

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# 메트릭 정의
render_jobs_total = Counter(
    'render_jobs_total',
    'Total number of render jobs',
    ['status', 'tier']
)

gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['node_id', 'gpu_id']
)

render_duration_seconds = Histogram(
    'render_duration_seconds',
    'Render job duration',
    ['tier'],
    buckets=[60, 300, 900, 1800, 3600, 7200]
)

queue_depth = Gauge(
    'queue_depth',
    'Number of jobs in queue',
    ['priority']
)

revenue_total = Counter(
    'revenue_usd_total',
    'Total revenue in USD',
    ['tier']
)

# 메트릭 수집
class MetricsCollector:
    def __init__(self):
        # Prometheus HTTP 서버 시작
        start_http_server(9090)

    async def collect_gpu_metrics(self):
        """GPU 메트릭 수집"""

        import pynvml
        pynvml.nvmlInit()

        for node in gpu_manager.nodes:
            for gpu in node["gpus"]:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu["gpu_id"])

                # 사용률
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                gpu_utilization.labels(
                    node_id=node["node_id"],
                    gpu_id=gpu["gpu_id"]
                ).set(utilization.gpu)

    async def record_job_completion(self, job: Dict):
        """작업 완료 기록"""

        # 작업 수 증가
        render_jobs_total.labels(
            status='completed',
            tier=job['gpu_tier']
        ).inc()

        # 렌더 시간 기록
        duration = (job['completed_at'] - job['started_at']).total_seconds()
        render_duration_seconds.labels(tier=job['gpu_tier']).observe(duration)

        # 수익 기록
        revenue_total.labels(tier=job['gpu_tier']).inc(job['actual_cost'])
```

### Auto-scaling

```python
# autoscaler.py
class AutoScaler:
    """자동 스케일링 (Kubernetes HPA 기반)"""

    def __init__(self):
        self.k8s_client = kubernetes.client.AppsV1Api()

    async def scale_render_nodes(self):
        """렌더 노드 자동 스케일링"""

        # 1. 큐 깊이 확인
        queue_sizes = {
            priority: redis_client.llen(f"render_queue:{priority}")
            for priority in ["high", "normal", "low"]
        }

        total_queued = sum(queue_sizes.values())

        # 2. 현재 노드 수
        deployment = self.k8s_client.read_namespaced_deployment(
            name="render-nodes",
            namespace="default"
        )
        current_replicas = deployment.spec.replicas

        # 3. 목표 노드 수 계산
        # 큐에 있는 작업 10개당 노드 1개
        target_replicas = max(1, min(20, total_queued // 10))

        # 4. 스케일링
        if target_replicas != current_replicas:
            logger.info(
                f"Scaling render nodes: {current_replicas} -> {target_replicas}"
            )

            deployment.spec.replicas = target_replicas
            self.k8s_client.patch_namespaced_deployment(
                name="render-nodes",
                namespace="default",
                body=deployment
            )
```

---

## 보안 및 격리

### Docker Container 격리

```dockerfile
# Dockerfile.render-node
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Blender 설치
RUN apt-get update && apt-get install -y \
    blender \
    && rm -rf /var/lib/apt/lists/*

# 비특권 사용자
RUN useradd -m -u 1000 renderer
USER renderer

# 렌더 스크립트
COPY render_worker.py /app/
WORKDIR /app

CMD ["python3", "render_worker.py"]
```

### 사용자 격리 및 권한

```python
# security.py
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """JWT 토큰 검증"""

    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=["HS256"]
        )

        user_id = payload.get("user_id")
        tier = payload.get("tier")  # free, pro, enterprise

        return {"user_id": user_id, "tier": tier}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# 티어별 제한
TIER_LIMITS = {
    "free": {
        "max_jobs_per_day": 5,
        "max_resolution": (1920, 1080),
        "max_samples": 128,
        "priority": "low"
    },
    "pro": {
        "max_jobs_per_day": 100,
        "max_resolution": (3840, 2160),
        "max_samples": 512,
        "priority": "normal"
    },
    "enterprise": {
        "max_jobs_per_day": float('inf'),
        "max_resolution": (7680, 4320),
        "max_samples": 4096,
        "priority": "high"
    }
}
```

---

## 핵심 요약

### 아키텍처 핵심 원칙

1. **스케일러빌리티**: Kubernetes 기반 자동 스케일링
2. **효율성**: 스마트 스케줄링으로 GPU 활용률 최대화
3. **비용 최적화**: Spot Instance, 티어 기반 가격
4. **안정성**: ECC 메모리, 자동 재시도, 모니터링
5. **보안**: 컨테이너 격리, JWT 인증, 티어별 제한

### 기술 스택

```
Frontend:  React + TypeScript
API:       FastAPI (Python)
Queue:     Redis / RabbitMQ
DB:        MongoDB / PostgreSQL
Storage:   S3 / Ceph
Container: Docker + Kubernetes
Monitor:   Prometheus + Grafana
Render:    Blender, V-Ray, Octane
```

### 예상 처리량 (10x RTX A6000 노드)

- 시간당 렌더 작업: ~120 jobs (Full HD, 128 samples)
- 일일 처리 능력: ~2,880 jobs
- 월간 수익: ~$221,400 (70% 가동률 기준)

**다음**: 렌더링 엔진 통합 가이드 →
