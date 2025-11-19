# 프로덕션 레벨 GPU 클라우드 서비스 구현

## 목차
1. [전체 시스템 스택](#전체-시스템-스택)
2. [Kubernetes 기반 오케스트레이션](#kubernetes-기반-오케스트레이션)
3. [완전한 API 구현](#완전한-api-구현)
4. [빌링 및 결제 시스템](#빌링-및-결제-시스템)
5. [모니터링 및 알람](#모니터링-및-알람)
6. [보안 및 네트워크 격리](#보안-및-네트워크-격리)
7. [CI/CD 파이프라인](#cicd-파이프라인)
8. [실제 배포 가이드](#실제-배포-가이드)

---

## 전체 시스템 스택

### 프로덕션 인프라 구성

```yaml
# production-stack.yaml
version: '3.8'

services:
  # API Gateway
  api_gateway:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend_api

  # Backend API
  backend_api:
    image: gpu-rental-api:latest
    replicas: 3
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/gpurental
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      - postgres
      - redis

  # PostgreSQL Database
  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: gpurental
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}

  # Redis (Queue & Cache)
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  # Job Scheduler
  scheduler:
    image: gpu-rental-scheduler:latest
    environment:
      - REDIS_URL=redis://redis:6379
      - K8S_NAMESPACE=gpu-compute

  # Prometheus (Metrics)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  # Grafana (Visualization)
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 기술 스택 상세

```python
# tech_stack.py

PRODUCTION_STACK = {
    "frontend": {
        "framework": "React 18 + TypeScript",
        "ui_library": "Material-UI (MUI)",
        "state_management": "Redux Toolkit",
        "build_tool": "Vite",
        "deployment": "Vercel / Cloudflare Pages"
    },

    "backend_api": {
        "framework": "FastAPI (Python 3.11)",
        "async_runtime": "uvicorn + gunicorn",
        "validation": "Pydantic v2",
        "orm": "SQLAlchemy 2.0",
        "migration": "Alembic",
        "testing": "pytest + pytest-asyncio"
    },

    "database": {
        "primary": "PostgreSQL 15 (user data, jobs, billing)",
        "cache": "Redis 7 (sessions, job queue)",
        "timeseries": "InfluxDB (metrics)",
        "object_storage": "S3 compatible (MinIO / AWS S3)"
    },

    "compute": {
        "orchestration": "Kubernetes 1.28",
        "gpu_operator": "NVIDIA GPU Operator",
        "networking": "Cilium (eBPF)",
        "storage": "Rook-Ceph (distributed storage)",
        "monitoring": "Prometheus + Grafana"
    },

    "ml_frameworks": {
        "pytorch": "2.1+",
        "tensorflow": "2.14+",
        "jax": "0.4+",
        "onnx": "1.15+",
        "rapids": "23.10+ (GPU-accelerated)"
    },

    "rendering": {
        "blender": "4.0+",
        "v-ray": "6.0+",
        "octane": "2023+",
        "redshift": "3.5+"
    },

    "observability": {
        "metrics": "Prometheus",
        "logs": "Loki + Promtail",
        "tracing": "Jaeger (OpenTelemetry)",
        "visualization": "Grafana",
        "alerting": "Alertmanager + PagerDuty"
    },

    "security": {
        "authentication": "OAuth 2.0 + JWT",
        "secrets": "HashiCorp Vault",
        "network_policy": "Cilium NetworkPolicy",
        "ssl": "Let's Encrypt (cert-manager)",
        "scanning": "Trivy (container scanning)"
    },

    "cicd": {
        "git": "GitHub / GitLab",
        "ci": "GitHub Actions / GitLab CI",
        "registry": "Harbor (private registry)",
        "gitops": "ArgoCD",
        "deployment": "Helm charts"
    }
}
```

---

## Kubernetes 기반 오케스트레이션

### Kubernetes 클러스터 구성

```yaml
# k8s-cluster-config.yaml

# GPU Node Pool
apiVersion: v1
kind: Node
metadata:
  name: gpu-node-001
  labels:
    gpu: "true"
    gpu-model: "rtx-a6000"
    gpu-count: "4"
    gpu-memory: "192gb"  # 4x48GB
    node-type: "compute"
spec:
  # Node가 가진 리소스
  capacity:
    nvidia.com/gpu: "4"
    memory: "256Gi"
    cpu: "64"
  # 다른 워크로드 방지 (GPU 전용)
  taints:
    - key: nvidia.com/gpu
      effect: NoSchedule

---
# GPU 작업 Pod 예시
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-training-job
  namespace: gpu-compute
spec:
  restartPolicy: Never

  # GPU 노드에만 스케줄
  nodeSelector:
    gpu: "true"

  # GPU 리소스 요청
  containers:
  - name: trainer
    image: nvcr.io/nvidia/pytorch:23.10-py3
    command: ["python", "train.py"]

    resources:
      limits:
        nvidia.com/gpu: 2  # 2개 GPU 요청
        memory: "64Gi"
      requests:
        nvidia.com/gpu: 2
        memory: "32Gi"

    volumeMounts:
    - name: data
      mountPath: /data
    - name: workspace
      mountPath: /workspace

    env:
    - name: CUDA_VISIBLE_DEVICES
      value: "0,1"

  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: training-data-pvc
  - name: workspace
    persistentVolumeClaim:
      claimName: user-workspace-pvc
```

### NVIDIA GPU Operator 설치

```bash
# GPU Operator 설치 (자동으로 드라이버, CUDA 설치)
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator-system \
  --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true \  # GPU metrics
  --set gfd.enabled=true  # GPU Feature Discovery

# 설치 확인
kubectl get pods -n gpu-operator-system

# GPU 리소스 확인
kubectl describe nodes | grep nvidia.com/gpu
```

### GPU Time-slicing (여러 사용자 공유)

```yaml
# gpu-time-slicing-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator-system
data:
  config: |
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4  # 1개 GPU를 4개로 가상화

---
# 적용
kubectl apply -f gpu-time-slicing-config.yaml

# Device Plugin 재시작
kubectl rollout restart daemonset/nvidia-device-plugin-daemonset \
  -n gpu-operator-system
```

이제 1개의 물리 GPU가 4개의 가상 GPU로 보이게 됩니다:

```bash
kubectl describe node gpu-node-001
# ...
# Capacity:
#   nvidia.com/gpu: 16  # 4 physical × 4 replicas
# ...
```

### Dynamic GPU Allocation

```python
# gpu_allocator.py - Kubernetes Custom Controller
from kubernetes import client, config, watch
import logging

logger = logging.getLogger(__name__)

class GPUAllocator:
    """동적 GPU 할당 컨트롤러"""

    def __init__(self):
        config.load_incluster_config()
        self.v1 = client.CoreV1Api()
        self.batch_v1 = client.BatchV1Api()

    def create_training_job(
        self,
        user_id: str,
        image: str,
        command: list,
        num_gpus: int,
        gpu_memory_gb: int,
        cpu_cores: int = 4,
        memory_gb: int = 16
    ) -> str:
        """사용자 학습 작업 생성"""

        job_name = f"training-{user_id}-{int(time.time())}"

        # Job 스펙 정의
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace="gpu-compute",
                labels={
                    "user": user_id,
                    "type": "training",
                    "gpu-count": str(num_gpus)
                }
            ),
            spec=client.V1JobSpec(
                ttl_seconds_after_finished=3600,  # 1시간 후 자동 삭제
                backoff_limit=2,  # 최대 2번 재시도

                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"job": job_name, "user": user_id}
                    ),

                    spec=client.V1PodSpec(
                        restart_policy="Never",

                        # GPU 노드 선택
                        node_selector={
                            "gpu": "true",
                            f"gpu-memory-min": f"{gpu_memory_gb}gb"
                        },

                        # GPU 전용 노드 Toleration
                        tolerations=[
                            client.V1Toleration(
                                key="nvidia.com/gpu",
                                operator="Exists",
                                effect="NoSchedule"
                            )
                        ],

                        containers=[
                            client.V1Container(
                                name="trainer",
                                image=image,
                                command=command,

                                # 리소스 요청/제한
                                resources=client.V1ResourceRequirements(
                                    requests={
                                        "nvidia.com/gpu": str(num_gpus),
                                        "memory": f"{memory_gb}Gi",
                                        "cpu": str(cpu_cores)
                                    },
                                    limits={
                                        "nvidia.com/gpu": str(num_gpus),
                                        "memory": f"{memory_gb * 2}Gi",
                                        "cpu": str(cpu_cores * 2)
                                    }
                                ),

                                # 볼륨 마운트
                                volume_mounts=[
                                    client.V1VolumeMount(
                                        name="workspace",
                                        mount_path="/workspace"
                                    ),
                                    client.V1VolumeMount(
                                        name="datasets",
                                        mount_path="/datasets",
                                        read_only=True
                                    )
                                ],

                                # 환경 변수
                                env=[
                                    client.V1EnvVar(
                                        name="USER_ID",
                                        value=user_id
                                    ),
                                    client.V1EnvVar(
                                        name="WANDB_API_KEY",
                                        value_from=client.V1EnvVarSource(
                                            secret_key_ref=client.V1SecretKeySelector(
                                                name=f"user-{user_id}-secrets",
                                                key="wandb_api_key"
                                            )
                                        )
                                    )
                                ]
                            )
                        ],

                        # 볼륨
                        volumes=[
                            client.V1Volume(
                                name="workspace",
                                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                    claim_name=f"user-{user_id}-workspace"
                                )
                            ),
                            client.V1Volume(
                                name="datasets",
                                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                    claim_name="shared-datasets"
                                )
                            )
                        ]
                    )
                )
            )
        )

        # Job 생성
        try:
            self.batch_v1.create_namespaced_job(
                namespace="gpu-compute",
                body=job
            )

            logger.info(f"Created training job: {job_name}")

            return job_name

        except client.exceptions.ApiException as e:
            logger.error(f"Failed to create job: {e}")
            raise

    def watch_job_status(self, job_name: str, callback):
        """작업 상태 모니터링"""

        w = watch.Watch()

        for event in w.stream(
            self.batch_v1.list_namespaced_job,
            namespace="gpu-compute",
            field_selector=f"metadata.name={job_name}"
        ):
            job = event['object']
            status = job.status

            if status.succeeded:
                callback("completed")
                w.stop()
            elif status.failed:
                callback("failed")
                w.stop()
            elif status.active:
                callback("running")

# 사용 예시
allocator = GPUAllocator()

job_name = allocator.create_training_job(
    user_id="user-123",
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    command=["python", "/workspace/train.py"],
    num_gpus=2,
    gpu_memory_gb=48,
    cpu_cores=8,
    memory_gb=64
)

# 상태 모니터링
allocator.watch_job_status(
    job_name,
    callback=lambda status: print(f"Job status: {status}")
)
```

---

## 완전한 API 구현

### FastAPI 백엔드 (완전 구현)

```python
# main.py - Production FastAPI Application
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import jwt
import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
import redis.asyncio as redis
import stripe
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱
app = FastAPI(
    title="GPU Cloud Rental API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gpu-rental.example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip 압축
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Database
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Redis
redis_client = redis.from_url(os.getenv("REDIS_URL"))

# Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# JWT
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# ============================================
# Pydantic 모델
# ============================================

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    credits: float
    tier: str  # free, pro, enterprise
    created_at: datetime

class JobCreate(BaseModel):
    job_type: str  # training, rendering
    gpu_tier: str  # hobby, pro, studio, enterprise
    num_gpus: int = 1
    docker_image: str
    command: List[str]
    environment: Optional[Dict[str, str]] = {}
    datasets: Optional[List[str]] = []

class JobResponse(BaseModel):
    job_id: str
    status: str  # queued, running, completed, failed
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    estimated_cost: float
    actual_cost: Optional[float]
    gpu_hours: float
    logs_url: Optional[str]

# ============================================
# 인증 & 사용자 관리
# ============================================

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """JWT 토큰에서 현재 사용자 추출"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")

        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        # DB에서 사용자 조회
        async with async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()

            if user is None:
                raise HTTPException(status_code=401, detail="User not found")

            return user

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/api/auth/register", response_model=UserResponse)
async def register(user: UserCreate):
    """사용자 등록"""

    # 비밀번호 해싱
    hashed_password = bcrypt.hashpw(
        user.password.encode('utf-8'),
        bcrypt.gensalt()
    )

    # DB에 저장
    async with async_session() as session:
        # 이메일 중복 체크
        result = await session.execute(
            select(User).where(User.email == user.email)
        )
        if result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Email already registered")

        # 새 사용자
        new_user = User(
            email=user.email,
            password_hash=hashed_password.decode('utf-8'),
            full_name=user.full_name,
            credits=10.0,  # 가입 보너스 $10
            tier="free",
            created_at=datetime.utcnow()
        )

        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

        logger.info(f"New user registered: {user.email}")

        return new_user

@app.post("/api/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """로그인"""

    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.email == form_data.username)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # 비밀번호 확인
        if not bcrypt.checkpw(
            form_data.password.encode('utf-8'),
            user.password_hash.encode('utf-8')
        ):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # JWT 토큰 생성
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = jwt.encode(
            {
                "sub": user.id,
                "exp": datetime.utcnow() + access_token_expires
            },
            SECRET_KEY,
            algorithm=ALGORITHM
        )

        return {
            "access_token": access_token,
            "token_type": "bearer"
        }

# ============================================
# GPU 작업 관리
# ============================================

@app.post("/api/jobs/submit", response_model=JobResponse)
async def submit_job(
    job: JobCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """GPU 작업 제출"""

    # 비용 계산
    estimated_cost = estimate_job_cost(job)

    # 크레딧 확인
    if current_user.credits < estimated_cost:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Need ${estimated_cost:.2f}, have ${current_user.credits:.2f}"
        )

    # Job ID 생성
    job_id = f"job-{current_user.id}-{int(datetime.utcnow().timestamp())}"

    # DB에 저장
    async with async_session() as session:
        new_job = Job(
            job_id=job_id,
            user_id=current_user.id,
            job_type=job.job_type,
            gpu_tier=job.gpu_tier,
            num_gpus=job.num_gpus,
            docker_image=job.docker_image,
            command=job.command,
            status="queued",
            estimated_cost=estimated_cost,
            created_at=datetime.utcnow()
        )

        session.add(new_job)
        await session.commit()

    # Redis 큐에 추가
    await redis_client.rpush(
        f"job_queue:{job.gpu_tier}",
        job_id
    )

    # 백그라운드 작업: Kubernetes Job 생성
    background_tasks.add_task(create_k8s_job, job_id, job)

    logger.info(f"Job submitted: {job_id} by user {current_user.email}")

    return JobResponse(
        job_id=job_id,
        status="queued",
        created_at=datetime.utcnow(),
        estimated_cost=estimated_cost,
        gpu_hours=0
    )

@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """작업 상태 조회"""

    async with async_session() as session:
        result = await session.execute(
            select(Job).where(
                Job.job_id == job_id,
                Job.user_id == current_user.id
            )
        )
        job = result.scalar_one_or_none()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # 진행률 계산 (Kubernetes에서)
        if job.status == "running":
            progress = await get_job_progress_from_k8s(job_id)
            job.progress = progress

        return job

@app.get("/api/jobs", response_model=List[JobResponse])
async def list_jobs(
    current_user: User = Depends(get_current_user),
    status: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
):
    """사용자의 작업 목록"""

    async with async_session() as session:
        query = select(Job).where(Job.user_id == current_user.id)

        if status:
            query = query.where(Job.status == status)

        query = query.order_by(Job.created_at.desc()).limit(limit).offset(offset)

        result = await session.execute(query)
        jobs = result.scalars().all()

        return jobs

@app.delete("/api/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """작업 취소"""

    async with async_session() as session:
        result = await session.execute(
            select(Job).where(
                Job.job_id == job_id,
                Job.user_id == current_user.id
            )
        )
        job = result.scalar_one_or_none()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status not in ["queued", "running"]:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled")

        # Kubernetes Job 삭제
        await delete_k8s_job(job_id)

        # 상태 업데이트
        job.status = "cancelled"
        job.completed_at = datetime.utcnow()

        await session.commit()

        logger.info(f"Job cancelled: {job_id}")

        return {"message": "Job cancelled successfully"}

# ============================================
# 결제 & 크레딧
# ============================================

@app.post("/api/credits/purchase")
async def purchase_credits(
    amount_usd: float,
    current_user: User = Depends(get_current_user)
):
    """크레딧 구매 (Stripe)"""

    # Stripe 결제 세션 생성
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price_data': {
                'currency': 'usd',
                'product_data': {
                    'name': 'GPU Cloud Credits',
                },
                'unit_amount': int(amount_usd * 100),  # cents
            },
            'quantity': 1,
        }],
        mode='payment',
        success_url=f'https://gpu-rental.example.com/payment/success?session_id={{CHECKOUT_SESSION_ID}}',
        cancel_url='https://gpu-rental.example.com/payment/cancel',
        client_reference_id=str(current_user.id),
        metadata={
            'user_id': current_user.id,
            'amount_usd': amount_usd
        }
    )

    return {"checkout_url": session.url}

@app.post("/api/webhooks/stripe")
async def stripe_webhook(request: Request):
    """Stripe 웹훅 (결제 완료 처리)"""

    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET")
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # 결제 성공 이벤트
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']

        user_id = int(session['metadata']['user_id'])
        amount_usd = float(session['metadata']['amount_usd'])

        # 크레딧 추가
        async with async_session() as db_session:
            result = await db_session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one()

            user.credits += amount_usd

            # 트랜잭션 기록
            transaction = Transaction(
                user_id=user_id,
                amount_usd=amount_usd,
                type="credit_purchase",
                stripe_session_id=session['id'],
                created_at=datetime.utcnow()
            )

            db_session.add(transaction)
            await db_session.commit()

            logger.info(f"Credits added: ${amount_usd} for user {user_id}")

    return {"status": "success"}

# ============================================
# 헬스 체크
# ============================================

@app.get("/health")
async def health_check():
    """헬스 체크"""

    # DB 연결 확인
    try:
        async with async_session() as session:
            await session.execute(select(1))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    # Redis 연결 확인
    try:
        await redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"

    return {
        "status": "ok",
        "database": db_status,
        "redis": redis_status,
        "version": "1.0.0"
    }

# ============================================
# 헬퍼 함수들
# ============================================

def estimate_job_cost(job: JobCreate) -> float:
    """작업 비용 추정"""

    tier_pricing = {
        "hobby": 0.75,
        "pro": 1.50,
        "studio": 3.00,
        "enterprise": 10.00
    }

    hourly_rate = tier_pricing[job.gpu_tier] * job.num_gpus

    # 예상 시간 (추후 ML 모델로 개선)
    estimated_hours = 2.0

    return hourly_rate * estimated_hours

async def create_k8s_job(job_id: str, job_spec: JobCreate):
    """Kubernetes Job 생성"""
    # 앞에서 구현한 GPUAllocator 사용
    allocator = GPUAllocator()

    await allocator.create_training_job(
        user_id=job_id,
        image=job_spec.docker_image,
        command=job_spec.command,
        num_gpus=job_spec.num_gpus,
        gpu_memory_gb=48,
        cpu_cores=8,
        memory_gb=64
    )
```

이 API는 프로덕션 레벨 기능을 모두 포함합니다:
- JWT 인증
- Stripe 결제 통합
- Kubernetes 작업 관리
- 비동기 DB 작업
- 웹훅 처리
- 헬스 체크

계속해서 모니터링과 보안 부분을 작성하겠습니다...
