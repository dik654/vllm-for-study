# 분산 LLM 인프라 플랫폼

## 🎯 프로젝트 개요

여러 서버(Provider)가 LLM API, Memory, Storage 리소스를 제공하고, 중앙 Router가 이를 통합하여 사용자에게 서비스를 제공하는 **분산 LLM 인프라 플랫폼**입니다.

### 핵심 기능
- **Provider**: vLLM 기반 LLM inference + Context/GP Memory/Storage 제공
- **Central Router**: 요청 라우팅, metric 수집, 정산 처리
- **Billing System**: OpenRouter 스타일 commission (5%) + 부가 서비스 마진 (20%)

### 서비스 타입
1. **LLM API** - vLLM inference (5% commission)
2. **Context Memory** - LLM 컨텍스트용 (KV cache, conversation history)
3. **Context Storage** - LLM 데이터 영구 저장 (chat history, preferences)
4. **GP Memory** - 범용 메모리 (application, database buffer)
5. **GP Storage** - 범용 스토리지 (Hot/Cold tier)

---

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        Central Router                        │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │ Request Router │  │ Metric Collector│  │ Billing Engine │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │Provider1│          │Provider2│          │Provider3│
   │  vLLM   │          │  vLLM   │          │  vLLM   │
   │ Memory  │          │ Memory  │          │ Memory  │
   │ Storage │          │ Storage │          │ Storage │
   └─────────┘          └─────────┘          └─────────┘
```

---

## 📚 문서

- **[SPECIFICATION.md](./SPECIFICATION.md)** - 상세 기술 스펙 (API, DB 스키마, 보안 등)
- **[WORKFLOW_SPECIFICATION.md](./WORKFLOW_SPECIFICATION.md)** - n8n 스타일 워크플로우 UI 사양
- **[TODO.md](./TODO.md)** - 구현 로드맵 및 체크리스트
- **[examples/workflows/](./examples/workflows/)** - 워크플로우 예시 (JSON)

---

## 🚀 빠른 시작

### 사전 요구사항
- Python 3.10+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+

### 로컬 개발 환경 구축

```bash
# 1. Repository clone
git clone <repo-url>
cd distributed-llm-platform

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일을 편집하여 설정값 입력

# 3. 의존성 설치
pip install -r requirements.txt
# 또는 Poetry 사용
poetry install

# 4. Docker 컨테이너 시작 (PostgreSQL, Redis, etc.)
make docker-up
# 또는
docker-compose up -d

# 5. Database migration
make db-upgrade
# 또는
alembic upgrade head

# 6. Router 실행
make run-router
# 또는
python -m router.main

# 7. Provider 실행 (별도 터미널)
make run-provider
# 또는
python -m provider.main
```

### Makefile 명령어

```bash
make help          # 사용 가능한 명령어 보기
make install       # 의존성 설치
make test          # 테스트 실행
make lint          # 린트 체크
make format        # 코드 포맷팅
make docker-up     # Docker 서비스 시작
make docker-down   # Docker 서비스 중지
make migrate       # 마이그레이션 생성
make db-upgrade    # 마이그레이션 적용
make run-router    # Router 로컬 실행
make run-provider  # Provider 로컬 실행
```

### 서비스 접속 주소

- **Router API**: http://localhost:8000
- **Provider API**: http://localhost:8001
- **API Documentation (Router)**: http://localhost:8000/docs
- **API Documentation (Provider)**: http://localhost:8001/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

---

## 🧪 테스트

```bash
# Unit tests
poetry run pytest tests/unit

# Integration tests
poetry run pytest tests/integration

# E2E tests
poetry run pytest tests/e2e

# Load tests
cd tests/load
k6 run load_test.js
```

---

## 📊 주요 지표 (KPI)

### Technical KPIs
- **Uptime**: 99.9% 이상
- **Latency**: P95 < 3초
- **Error rate**: < 1%
- **Throughput**: 1000 RPS 이상

### Business KPIs
- **Active Providers**: 10개 이상
- **Active Users**: 100명 이상
- **Monthly Revenue**: $10,000 이상

---

## 🛠️ 기술 스택

### Backend
- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL + TimescaleDB (metrics)
- **Cache**: Redis
- **Message Queue**: Kafka or RabbitMQ
- **LLM Engine**: vLLM

### Monitoring
- **Metrics**: Prometheus + Grafana
- **Logging**: ELK stack
- **Tracing**: OpenTelemetry + Jaeger

### DevOps
- **Containerization**: Docker
- **Orchestration**: Kubernetes or Docker Swarm
- **CI/CD**: GitHub Actions
- **Deployment**: Blue-green deployment

---

## 📁 프로젝트 구조

```
distributed-llm-platform/
├── router/                      # Central Router
│   ├── api/                     # API endpoints
│   │   ├── v1/
│   │   │   ├── providers.py
│   │   │   ├── users.py
│   │   │   ├── llm.py
│   │   │   └── billing.py
│   │   └── dependencies.py
│   ├── core/                    # Core logic
│   │   ├── routing.py
│   │   ├── metrics.py
│   │   └── billing.py
│   ├── models/                  # SQLAlchemy models
│   │   ├── provider.py
│   │   ├── user.py
│   │   └── request.py
│   ├── services/                # Business logic
│   │   ├── provider_service.py
│   │   ├── user_service.py
│   │   └── billing_service.py
│   ├── config.py
│   └── main.py
│
├── provider/                    # Provider Agent
│   ├── api/                     # API endpoints
│   │   ├── llm.py
│   │   ├── memory.py
│   │   └── storage.py
│   ├── core/                    # Core logic
│   │   ├── vllm_client.py
│   │   └── metrics.py
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── memory_service.py
│   │   └── storage_service.py
│   ├── config.py
│   └── main.py
│
├── shared/                      # Shared code
│   ├── schemas/                 # Pydantic schemas
│   │   ├── provider.py
│   │   ├── user.py
│   │   ├── request.py
│   │   └── metric.py
│   └── utils/
│       ├── auth.py
│       ├── redis.py
│       └── logging.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── load/
│
├── docker/
│   ├── router.Dockerfile
│   └── provider.Dockerfile
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── deployment.md
│
├── docker-compose.yml
├── .env.example
├── SPECIFICATION.md
├── TODO.md
└── README.md
```

---

## 🔐 보안

- **인증**: API key (SHA-256 해시 저장)
- **전송 암호화**: TLS 1.3 강제
- **Rate Limiting**: Redis 기반 token bucket
- **DDoS 방어**: Cloudflare 연동
- **데이터 암호화**: 민감 정보 AES-256 암호화

---

## 💰 가격 정책 (OpenRouter 스타일)

### Provider 단가 (투명하게 공개)
```
LLM API:
  - Input token: $1.00 per 1M tokens
  - Cached input: $0.10 per 1M tokens (90% 할인)
  - Output token: $8.00 per 1M tokens (8배)
  - Batch processing: 50% 할인

Context Memory: $0.03/GB/hour
Context Storage: $0.012/GB/hour
GP Memory: $0.009/GB/hour
GP Storage Hot: $0.001/GB/hour
GP Storage Cold: $0.0002/GB/hour (80% 할인)
```

### 사용자 단가
```
LLM API: Provider 단가 + 5% commission
Memory/Storage: Provider 단가 + 20% margin
```

### 플랫폼 수수료
- **LLM API**: 5% commission (OpenRouter와 동일)
- **Memory/Storage**: 20% margin (차별화)
- **실패한 요청**: 무료 (fallback 시)

---

## 🗓️ 로드맵

### Phase 1: MVP (Week 1-6) ✅ In Progress
- [x] 프로젝트 설정 및 구조 생성
- [x] Docker 환경 구성
- [x] Database 모델 정의 (Provider, User, Request, Workflow)
- [x] FastAPI 기본 애플리케이션
- [x] Alembic 마이그레이션 설정
- [ ] Router 기본 구현
- [ ] 단순 라우팅 및 Proxy

### Phase 2: Core Features (Week 7-14)
- [ ] Provider Agent 구현
- [ ] 고급 라우팅 전략
- [ ] Memory & Storage 서비스
- [ ] Billing Engine

### Phase 3: Production Ready (Week 15-22)
- [ ] Monitoring & Observability
- [ ] 결제 시스템 연동
- [ ] Dashboard 구현
- [ ] 보안 강화

### Phase 4: Advanced Features (Week 23+)
- [ ] Performance 최적화
- [ ] Provider SLA 모니터링
- [ ] Auto-scaling
- [ ] Multi-region 지원
- [ ] ML 기반 최적화

자세한 내용은 [TODO.md](./TODO.md) 참고.

---

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 라이선스

이 프로젝트는 Apache 2.0 라이선스를 따릅니다.

---

## 📧 문의

- **Email**: contact@example.com
- **Slack**: [Join our Slack](https://slack.example.com)
- **Issues**: [GitHub Issues](https://github.com/example/issues)

---

## 🙏 감사의 말

- [vLLM](https://github.com/vllm-project/vllm) - LLM inference engine
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [OpenRouter](https://openrouter.ai/) - Inspiration for LLM routing
