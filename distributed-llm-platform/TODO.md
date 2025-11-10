# 분산 LLM 플랫폼 구현 TODO

## 프로젝트 개요
여러 Provider가 LLM, Memory, Storage 리소스를 제공하고, 중앙 Router가 요청을 라우팅하며 metric 기반으로 정산하는 분산 플랫폼 구현.

**목표 기간**: 4-6개월
**핵심 기술**: FastAPI, PostgreSQL, Redis, vLLM, Docker

---

## Phase 1: MVP (Week 1-6) - 기본 동작 구현

### Week 1-2: 프로젝트 설정 및 기본 인프라

#### 1.1 프로젝트 초기화
- [ ] Git repository 생성
- [ ] 프로젝트 구조 설계
  ```
  distributed-llm-platform/
  ├── router/              # Central Router
  │   ├── api/
  │   ├── core/
  │   ├── models/
  │   └── services/
  ├── provider/            # Provider Agent
  │   ├── api/
  │   ├── core/
  │   └── services/
  ├── shared/              # 공통 코드
  │   ├── schemas/
  │   └── utils/
  ├── tests/
  ├── docker/
  └── docs/
  ```
- [ ] Poetry/pip-tools로 의존성 관리 설정
- [ ] Pre-commit hooks 설정 (black, isort, flake8, mypy)
- [ ] README.md 작성

#### 1.2 Database 설정
- [ ] PostgreSQL Docker 설정
- [ ] SQLAlchemy 모델 정의
  - [ ] `providers` 테이블
  - [ ] `users` 테이블
  - [ ] `requests` 테이블 (로그)
- [ ] Alembic 마이그레이션 설정
- [ ] 초기 migration 생성
- [ ] Database seeding script 작성 (테스트용 데이터)

#### 1.3 Redis 설정
- [ ] Redis Docker 설정
- [ ] Redis connection pool 구현
- [ ] Rate limiting 기본 구조 (redis-py)
- [ ] Cache utility 함수 작성

#### 1.4 Docker 환경 구성
- [ ] `docker-compose.yml` 작성
  - [ ] PostgreSQL service
  - [ ] Redis service
  - [ ] Router service
  - [ ] Provider service (template)
- [ ] Dockerfile 작성 (Router, Provider)
- [ ] `.env.example` 파일 작성
- [ ] 로컬 개발 환경 문서화

### Week 3-4: Router 기본 구현

#### 2.1 Router API 기본 구조
- [ ] FastAPI app 초기화
- [ ] CORS 설정
- [ ] Exception handling middleware
- [ ] Logging 설정 (structlog)
- [ ] Health check endpoint (`GET /health`)

#### 2.2 인증 시스템
- [ ] User API key 생성 함수
- [ ] Provider token 생성 함수
- [ ] API key 검증 dependency (`authenticate_user`)
- [ ] Provider token 검증 dependency (`authenticate_provider`)
- [ ] Rate limiting middleware (Redis 기반)

#### 2.3 Provider 관리
- [ ] Provider 등록 API
  - [ ] `POST /v1/providers/register`
  - [ ] Provider 정보 검증
  - [ ] Token 생성 및 저장
- [ ] Provider 목록 조회
  - [ ] `GET /v1/providers` (admin only)
- [ ] Provider 상태 업데이트
  - [ ] `PATCH /v1/providers/{provider_id}`
- [ ] Provider CRUD operations service 구현

#### 2.4 User 관리
- [ ] User 생성 API
  - [ ] `POST /v1/users/register`
  - [ ] Email 검증
  - [ ] API key 생성
- [ ] User 정보 조회
  - [ ] `GET /v1/users/me`
- [ ] User CRUD operations service 구현

### Week 5-6: 기본 라우팅 및 Proxy

#### 3.1 단순 라우팅 구현
- [ ] Provider registry (메모리 캐시)
- [ ] Round-robin 라우팅 알고리즘
- [ ] Provider 선택 service 구현
- [ ] Provider health 상태 추적

#### 3.2 LLM API Proxy
- [ ] Proxy endpoint 구현
  - [ ] `POST /v1/chat/completions`
  - [ ] `POST /v1/completions`
- [ ] Provider로 요청 전달 (httpx)
- [ ] Response streaming 지원
- [ ] Timeout 및 retry 로직
- [ ] Error handling (Provider 장애 시)

#### 3.3 기본 Metric 수집
- [ ] Request 로깅 (DB에 저장)
  - [ ] user_id, provider_id, model, tokens, latency
- [ ] Provider metric 수신 endpoint
  - [ ] `POST /v1/providers/{provider_id}/metrics`
- [ ] Metric DB 저장 service

#### 3.4 테스트
- [ ] Router API unit tests
- [ ] Integration tests (DB, Redis)
- [ ] E2E test (mock Provider 사용)
- [ ] Load test 준비 (Locust/K6)

---

## Phase 2: Core Features (Week 7-14) - 핵심 기능 구현

### Week 7-8: Provider Agent 구현

#### 4.1 Provider Agent 기본 구조
- [ ] FastAPI app 초기화 (Provider용)
- [ ] 설정 관리 (환경변수)
- [ ] Router 연결 설정

#### 4.2 vLLM 연동
- [ ] vLLM OpenAI-compatible server 설정
- [ ] Model loading 및 초기화
- [ ] `/v1/chat/completions` endpoint 구현
- [ ] `/v1/completions` endpoint 구현
- [ ] Streaming response 지원

#### 4.3 Provider Health Check
- [ ] Health check endpoint (`GET /health`)
- [ ] vLLM 상태 확인
- [ ] System metrics 수집 (CPU, GPU, Memory)

#### 4.4 Metric Reporting
- [ ] Metric 수집 로직
  - [ ] Request count, tokens, latency 추적
  - [ ] Memory 사용량 추적
  - [ ] Storage 사용량 추적
- [ ] 주기적 metric 전송 (Router로)
  - [ ] 60초마다 전송 (configurable)
- [ ] Metric buffer (Redis or in-memory)

#### 4.5 Provider 등록 자동화
- [ ] 시작 시 자동 Router 등록
- [ ] Heartbeat 전송 (10초마다)
- [ ] Graceful shutdown

### Week 9-10: 고급 라우팅 전략

#### 5.1 라우팅 전략 구현
- [ ] Least Latency 전략
  - [ ] Provider별 평균 latency 추적
  - [ ] 최저 latency Provider 선택
- [ ] Least Loaded 전략
  - [ ] 현재 active requests 추적
  - [ ] 부하율 계산 (current/max)
- [ ] Cost Optimized 전략
  - [ ] Request 예상 비용 계산
  - [ ] 최저 비용 Provider 선택
- [ ] Weighted 전략
  - [ ] 가중치 계산 (latency, load, cost)
  - [ ] Weighted random selection

#### 5.2 Health Check System
- [ ] Provider health checker service
- [ ] 주기적 health check (30초마다)
- [ ] Circuit breaker 패턴 구현
  - [ ] 연속 실패 시 Provider 비활성화
  - [ ] Half-open 상태 지원
- [ ] Health 상태 Redis 캐싱

#### 5.3 Load Balancing 고도화
- [ ] Sticky session 지원 (optional)
- [ ] Provider 우선순위 설정
- [ ] Geographic routing (optional)

### Week 11-12: Memory & Storage 서비스

#### 6.1 Memory Service (Provider)
- [ ] Memory allocation API
  - [ ] `POST /v1/memory/allocate`
  - [ ] Memory pool 관리
- [ ] Memory release API
  - [ ] `DELETE /v1/memory/release`
- [ ] Memory 상태 조회
  - [ ] `GET /v1/memory/status`
- [ ] 시간당 사용량 추적

#### 6.2 Storage Service (Provider)
- [ ] Storage upload API
  - [ ] `POST /v1/storage/upload`
  - [ ] Multipart upload 지원
- [ ] Storage download API
  - [ ] `GET /v1/storage/download`
  - [ ] Range request 지원
- [ ] Storage delete API
  - [ ] `DELETE /v1/storage/delete`
- [ ] Storage 상태 조회
  - [ ] `GET /v1/storage/status`
- [ ] 시간당 사용량 추적

#### 6.3 Router Memory/Storage Proxy
- [ ] Memory allocation 라우팅
  - [ ] 가용 메모리 기반 Provider 선택
- [ ] Storage upload 라우팅
  - [ ] 가용 스토리지 기반 Provider 선택
- [ ] Usage tracking (user별, provider별)

### Week 13-14: Billing Engine

#### 7.1 Provider 수익 계산
- [ ] `ProviderRevenue` 모델 정의
- [ ] LLM API 수익 계산 로직
  - [ ] Input/output tokens * rate
  - [ ] Request count * base rate
- [ ] Memory 수익 계산 로직
  - [ ] Allocated GB-hours * hourly rate
- [ ] Storage 수익 계산 로직
  - [ ] Used TB-hours * hourly rate
- [ ] 플랫폼 수수료 차감

#### 7.2 User 결제 계산
- [ ] `UserCharge` 모델 정의
- [ ] 사용량 집계 로직
- [ ] 회사 요금 적용 (Provider 단가 + 마진)
- [ ] 결제 금액 계산

#### 7.3 API Endpoints
- [ ] Provider 수익 조회
  - [ ] `GET /v1/providers/{provider_id}/revenue`
- [ ] User 사용량 조회
  - [ ] `GET /v1/usage`
- [ ] User invoice 생성
  - [ ] `GET /v1/invoices/{invoice_id}`

#### 7.4 정산 배치 Job
- [ ] Celery/RQ 설정
- [ ] 일별 정산 task
  - [ ] Provider 수익 계산 및 저장
  - [ ] User 결제 금액 계산 및 저장
- [ ] 월별 정산 task
- [ ] 정산 완료 알림

---

## Phase 3: Production Ready (Week 15-22) - 프로덕션 준비

### Week 15-16: Monitoring & Observability

#### 8.1 Metric 수집 고도화
- [ ] Prometheus exporter 구현
  - [ ] Router metrics (request rate, latency, error rate)
  - [ ] Provider metrics (health, load, revenue)
- [ ] Grafana dashboard 구성
  - [ ] System overview
  - [ ] Provider performance
  - [ ] User usage
  - [ ] Revenue analytics

#### 8.2 Logging
- [ ] Structured logging (structlog)
- [ ] Log aggregation (ELK stack or Loki)
- [ ] PII masking
- [ ] Log rotation

#### 8.3 Distributed Tracing
- [ ] OpenTelemetry 설정
- [ ] Trace context 전파 (Router → Provider)
- [ ] Jaeger 연동

#### 8.4 Alerting
- [ ] Alert rule 정의
  - [ ] Provider down
  - [ ] High error rate (> 5%)
  - [ ] High latency (P95 > 5s)
  - [ ] Payment failures
- [ ] Alert channel 설정 (Slack, PagerDuty)

### Week 17-18: 결제 시스템 연동

#### 9.1 결제 Provider 선택
- [ ] Stripe vs. PayPal vs. 국내 PG (토스페이먼츠, KG이니시스) 비교
- [ ] SDK 설치 및 설정

#### 9.2 User 결제 흐름
- [ ] Payment method 등록
  - [ ] `POST /v1/payment-methods`
- [ ] Invoice 생성 및 결제 처리
  - [ ] `POST /v1/invoices/{invoice_id}/pay`
- [ ] 결제 실패 처리
- [ ] 환불 처리

#### 9.3 Provider 정산 흐름
- [ ] Payout 생성
  - [ ] `POST /v1/providers/{provider_id}/payouts`
- [ ] 은행 계좌 연동
- [ ] 정산 승인 프로세스

#### 9.4 Webhook 처리
- [ ] Payment success webhook
- [ ] Payment failure webhook
- [ ] Payout webhook

### Week 19-20: Dashboard 구현

#### 10.1 Provider Dashboard (Frontend)
- [ ] Next.js or React 설정
- [ ] 인증 (JWT)
- [ ] Overview 페이지
  - [ ] 실시간 metric (request count, latency)
  - [ ] 수익 현황 (오늘, 이번 달)
- [ ] Revenue 페이지
  - [ ] 기간별 수익 조회
  - [ ] 차트 (시계열)
- [ ] Payout 페이지
  - [ ] 정산 내역
  - [ ] 정산 요청

#### 10.2 User Dashboard (Frontend)
- [ ] 인증 (JWT)
- [ ] Overview 페이지
  - [ ] API key 관리
  - [ ] 사용량 현황
- [ ] Usage 페이지
  - [ ] 기간별 사용량 조회
  - [ ] 차트 (request, tokens)
- [ ] Billing 페이지
  - [ ] Invoice 목록
  - [ ] Payment method 관리

#### 10.3 Admin Dashboard (Backend API)
- [ ] Provider 관리
  - [ ] `GET /v1/admin/providers`
  - [ ] `PATCH /v1/admin/providers/{id}`
- [ ] User 관리
  - [ ] `GET /v1/admin/users`
  - [ ] `PATCH /v1/admin/users/{id}`
- [ ] System metrics
  - [ ] `GET /v1/admin/metrics`

### Week 21-22: 보안 강화

#### 11.1 인증 강화
- [ ] JWT 기반 인증 (optional, API key 대신)
- [ ] Refresh token 지원
- [ ] MFA (Multi-Factor Authentication) 지원 (optional)

#### 11.2 Rate Limiting 고도화
- [ ] User별 rate limit (tier별 다른 limit)
- [ ] Provider별 rate limit
- [ ] IP 기반 rate limit
- [ ] DDoS 방어 (Cloudflare 연동)

#### 11.3 데이터 암호화
- [ ] API key DB 저장 시 암호화
- [ ] Payment info 암호화
- [ ] TLS 강제 (HTTPS only)

#### 11.4 감사 로그
- [ ] Audit log 테이블 생성
- [ ] 중요 action 로깅
  - [ ] Provider 등록/삭제
  - [ ] 결제 처리
  - [ ] 정산 처리
- [ ] Audit log 조회 API

---

## Phase 4: Advanced Features (Week 23+) - 고급 기능

### Week 23-24: Performance 최적화

#### 12.1 부하 테스트
- [ ] Locust/K6 시나리오 작성
- [ ] 100 RPS 목표 테스트
- [ ] 1000 RPS 목표 테스트
- [ ] Bottleneck 파악

#### 12.2 최적화
- [ ] Database query 최적화
  - [ ] Index 추가
  - [ ] N+1 query 제거
- [ ] Redis 캐싱 강화
  - [ ] Provider 정보 캐싱
  - [ ] User 정보 캐싱
- [ ] Connection pooling 튜닝
- [ ] Async I/O 최대 활용

#### 12.3 Caching 전략
- [ ] CDN 설정 (static assets)
- [ ] Response caching (Redis)
- [ ] Cache invalidation 로직

### Week 25-26: Provider SLA 모니터링

#### 13.1 SLA 정의
- [ ] SLA 지표 정의
  - [ ] Uptime (99.9% 목표)
  - [ ] Latency (P95 < 3s 목표)
  - [ ] Error rate (< 1% 목표)
- [ ] SLA violation 탐지 로직

#### 13.2 SLA Dashboard
- [ ] Provider별 SLA 현황 조회
- [ ] SLA violation 알림
- [ ] SLA 위반 시 패널티 (수익 차감)

#### 13.3 자동 Provider 비활성화
- [ ] SLA 위반 누적 시 자동 비활성화
- [ ] 복구 후 재활성화 로직

### Week 27-28: Auto-scaling

#### 14.1 Provider Auto-scaling
- [ ] Provider load 모니터링
- [ ] 부하 임계값 초과 시 알림
- [ ] Kubernetes HPA 연동 (optional)

#### 14.2 Router Auto-scaling
- [ ] Router instance 부하 모니터링
- [ ] Kubernetes HPA 설정
- [ ] Load balancer 설정 (Nginx/HAProxy)

### Week 29-30: Multi-region 지원

#### 15.1 Region 관리
- [ ] Region 모델 추가 (us-east, us-west, eu-west, ap-northeast 등)
- [ ] Provider region 등록
- [ ] Geographic routing 구현

#### 15.2 Cross-region Replication
- [ ] Database replication 설정
- [ ] Redis cluster 설정
- [ ] Latency 기반 region 선택

### Week 31+: ML 기반 최적화

#### 16.1 Traffic 패턴 분석
- [ ] 시간대별 traffic 패턴 분석
- [ ] User별 사용 패턴 분석
- [ ] Model popularity 분석

#### 16.2 예측 기반 라우팅
- [ ] ML model 학습 (scikit-learn or PyTorch)
  - [ ] Input: request 정보, provider 상태
  - [ ] Output: optimal provider 예측
- [ ] Model serving (MLflow or TensorFlow Serving)
- [ ] A/B testing

#### 16.3 Dynamic Pricing
- [ ] 수요 기반 가격 조정
- [ ] Provider 경쟁력 기반 가격 조정

---

## Phase 5: Marketplace (Optional, Week 32+)

### Week 32-34: Provider Marketplace

#### 17.1 Provider 프로필
- [ ] Provider 상세 정보 페이지
  - [ ] 설명, 지원 모델
  - [ ] 가격 정보
  - [ ] SLA 지표
- [ ] Provider 랭킹
  - [ ] Performance 기반 랭킹
  - [ ] 리뷰 기반 랭킹

#### 17.2 리뷰 시스템
- [ ] User review 작성
  - [ ] `POST /v1/providers/{id}/reviews`
- [ ] Review 조회
- [ ] 평균 별점 계산

#### 17.3 Provider Discovery
- [ ] Provider 검색 (모델, 가격, region)
- [ ] Provider 추천 시스템

---

## Testing Strategy

### Unit Tests
- [ ] Router services (100% coverage 목표)
- [ ] Provider agent services
- [ ] Billing engine
- [ ] Authentication

### Integration Tests
- [ ] Database operations
- [ ] Redis operations
- [ ] API endpoints (Router ↔ Provider)

### E2E Tests
- [ ] User registration → API call → Billing
- [ ] Provider registration → Metric reporting → Revenue
- [ ] Payment flow

### Load Tests
- [ ] 100 RPS (baseline)
- [ ] 1000 RPS (target)
- [ ] 10000 RPS (stretch goal)

### Security Tests
- [ ] Penetration testing
- [ ] SQL injection 방어 확인
- [ ] XSS 방어 확인
- [ ] Rate limiting 동작 확인

---

## Deployment Checklist

### Infrastructure
- [ ] Kubernetes cluster 설정 (or Docker Swarm)
- [ ] Load balancer 설정 (Nginx/HAProxy)
- [ ] Database backup 자동화
- [ ] Redis persistence 설정
- [ ] SSL/TLS 인증서 설정 (Let's Encrypt)

### CI/CD
- [ ] GitHub Actions or GitLab CI 설정
- [ ] Docker image build pipeline
- [ ] Automated testing
- [ ] Blue-green deployment
- [ ] Rollback 전략

### Monitoring
- [ ] Prometheus + Grafana
- [ ] ELK stack (logging)
- [ ] Jaeger (tracing)
- [ ] PagerDuty (alerting)

### Documentation
- [ ] API documentation (Swagger/OpenAPI)
- [ ] Architecture diagram
- [ ] Runbook (운영 가이드)
- [ ] Troubleshooting guide

---

## Risk Management

### Technical Risks
- [ ] **Provider 장애**: Circuit breaker 및 retry 로직으로 대응
- [ ] **Database 부하**: Read replica 및 caching으로 대응
- [ ] **네트워크 지연**: Timeout 및 geographic routing으로 대응
- [ ] **Security 취약점**: 정기 보안 감사 및 패치

### Business Risks
- [ ] **Provider 이탈**: 경쟁력 있는 정산 조건 제공
- [ ] **사용자 이탈**: 안정적인 서비스 및 합리적인 가격
- [ ] **법적 리스크**: 이용약관, 개인정보처리방침 작성

---

## Success Metrics (KPI)

### Technical KPIs
- [ ] **Uptime**: 99.9% 이상
- [ ] **Latency**: P95 < 3초
- [ ] **Error rate**: < 1%
- [ ] **Throughput**: 1000 RPS 이상

### Business KPIs
- [ ] **Active Providers**: 10개 이상
- [ ] **Active Users**: 100명 이상
- [ ] **Monthly Revenue**: $10,000 이상
- [ ] **Provider 만족도**: 4.5/5.0 이상
- [ ] **User 만족도**: 4.5/5.0 이상

---

## Next Steps (시작 전 체크리스트)

1. [ ] 팀 구성 (Backend, Frontend, DevOps, PM)
2. [ ] 기술 스택 최종 확정
3. [ ] 개발 환경 구축 (Git, CI/CD, Staging)
4. [ ] Sprint 계획 (2주 단위 sprint)
5. [ ] 첫 번째 Provider 파트너 확보
6. [ ] 베타 테스터 모집

**시작일**: [날짜]
**예상 완료일**: [날짜 + 6개월]

---

## 참고

- [SPECIFICATION.md](./SPECIFICATION.md) - 상세 기술 스펙
- [Architecture Diagram](./docs/architecture.png)
- [API Documentation](./docs/api.md)
