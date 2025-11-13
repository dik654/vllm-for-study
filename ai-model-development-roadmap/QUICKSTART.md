# 🚀 빠른 시작 가이드

이 가이드는 5분 만에 학습을 시작할 수 있도록 도와줍니다.

## 📋 사전 준비

### 필수 요구사항
- Python 3.8 이상
- 10GB 이상의 디스크 공간
- (권장) NVIDIA GPU with 8GB+ VRAM

### 하드웨어별 권장사항

| 하드웨어 | Phase 0-2 | Phase 3-4 | Phase 5-6 |
|---------|-----------|-----------|-----------|
| CPU Only | ✅ 가능 | ⚠️ 느림 | ✅ 가능 |
| GPU 8GB | ✅ 최적 | ✅ 가능 | ✅ 최적 |
| GPU 16GB+ | ✅ 최적 | ✅ 최적 | ✅ 최적 |

---

## 1️⃣ 설치 (5분)

### 방법 A: 자동 설치 (권장)

```bash
cd ai-model-development-roadmap/setup
chmod +x install.sh
./install.sh
```

### 방법 B: 수동 설치

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r setup/requirements.txt
```

### 설치 확인

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

예상 출력:
```
PyTorch: 2.x.x
CUDA: True (또는 False)
Transformers: 4.x.x
```

---

## 2️⃣ 첫 번째 실습 (10분)

### Hello World: HuggingFace 모델 로딩

```python
# test_setup.py
from transformers import pipeline

# 감성 분석 파이프라인
classifier = pipeline("sentiment-analysis")

# 테스트
result = classifier("I love learning AI!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

print("✅ Setup 완료! 학습을 시작할 준비가 되었습니다!")
```

실행:
```bash
python test_setup.py
```

---

## 3️⃣ 수학적 기초 자가 진단 (3분) ⭐ **중요!**

### 📝 다음 문제를 **지금 당장** 풀어보세요:

**1. 선형대수**: $\begin{bmatrix}1&2\\3&4\end{bmatrix} \times \begin{bmatrix}5\\6\end{bmatrix} = ?$

**2. 미적분**: $f(x,y) = x^2 + 3xy$일 때, $\frac{\partial f}{\partial x} = ?$

**3. 확률**: 동전을 3번 던져서 앞면이 정확히 2번 나올 확률은?

### 🎯 결과 해석

**모두 30초 내 풀림**: ✅ Phase 0부터 바로 시작!
**1-2개 막힘**: ⚠️ Phase -1을 **1주**로 빠르게!
**전부 막힘**: 🚨 Phase -1부터 **2주**로 천천히!

> **"수학 없이 AI를 하는 것은 악보 없이 피아노를 치는 것과 같습니다."**
>
> 논문의 수식, Backpropagation 원리, Attention 메커니즘을 **진짜로** 이해하려면 수학이 필수입니다!

---

## 4️⃣ 학습 경로 선택

### 🎯 완전 정복 경로 (14주, 추천)
수학 기초 + 모든 Phase를 순서대로 진행합니다.

**수학 필요하면:**
```bash
cd phase-1-math-foundations
```
→ [Phase -1 README](./phase-1-math-foundations/README.md) 시작

**수학 준비됐으면:**
```bash
cd phase0-huggingface-ecosystem
```
→ [Phase 0 README](./phase0-huggingface-ecosystem/README.md) 시작

### ⚡ 빠른 학습 경로 (6주)
핵심만 빠르게 학습합니다.

**Week 1-2**: Phase 0-1 (HuggingFace + Transformer)
**Week 3-4**: Phase 2 or 3 (BERT/GPT or Diffusion)
**Week 5-6**: Phase 5-6 (최신 기법 + 프로젝트)

### 🎨 관심 분야별 경로

#### NLP 중심
Phase 0 → Phase 1 → Phase 2 → Phase 5 → Phase 6

#### Vision 중심
Phase 0 → Phase 1 → Phase 3 → Phase 4 → Phase 5 → Phase 6

#### 프로덕션 중심
Phase 0 → Phase 5 → Phase 6

---

## 4️⃣ 일일 학습 루틴

### 평일 (2-3시간)

```
09:00-09:30  이론 학습 (문서 읽기)
09:30-10:30  코드 구현
10:30-11:00  실습 과제
11:00-11:30  복습 및 정리
```

### 주말 (4-6시간)

```
10:00-11:00  주간 복습
11:00-13:00  프로젝트 작업
14:00-16:00  심화 실습
16:00-17:00  블로그 작성 / 정리
```

---

## 5️⃣ 학습 팁

### ✅ DO
- ✅ 매일 조금씩이라도 꾸준히
- ✅ 코드를 직접 타이핑 (복붙 금지!)
- ✅ 이해 안 되면 더 간단한 예제로
- ✅ 배운 내용 블로그에 정리
- ✅ 커뮤니티에 질문하기

### ❌ DON'T
- ❌ 한 번에 몰아서 하기
- ❌ 이해 없이 다음 단계로
- ❌ 에러 회피하기
- ❌ 혼자 고민만 하기
- ❌ 완벽주의 (80%면 넘어가기)

---

## 6️⃣ 도움받기

### 문제 해결 순서

1. **에러 메시지 읽기**
   - 90%는 메시지에 답이 있음

2. **문서 확인**
   - 각 Phase의 README 참고
   - 공식 문서 확인

3. **GitHub Issues 검색**
   - 비슷한 문제 찾기

4. **질문하기**
   - 에러 메시지 포함
   - 환경 정보 포함 (Python, GPU 등)
   - 시도한 해결책 설명

### 유용한 리소스

- **공식 문서**: https://huggingface.co/docs
- **논문**: https://arxiv.org
- **커뮤니티**: HuggingFace Forums, Reddit r/MachineLearning
- **Q&A**: Stack Overflow

---

## 7️⃣ 진행상황 추적

[PROGRESS.md](./PROGRESS.md) 파일을 복사하여 사용하세요:

```bash
cp PROGRESS.md MY_PROGRESS.md
```

매일 체크리스트를 업데이트하면서 성취감을 느껴보세요!

---

## 🎯 첫 주 목표

### Day 1
- [ ] 환경 설정 완료
- [ ] Phase 0 시작
- [ ] Model Card 작성

### Day 2-3
- [ ] Safetensors 실습
- [ ] Quantization 실험

### Day 4-5
- [ ] 토크나이저 이해
- [ ] Config 파일 분석

### Day 6-7
- [ ] 벤치마크 실습
- [ ] 훈련 파이프라인 구축

---

## 💡 동기부여

### 이미 이 여정을 완료한 사람들

> "12주 전만 해도 Transformer가 뭔지 몰랐는데, 이제는 직접 구현하고 논문도 읽어요!"
> - 김OO, 주니어 개발자 → ML Engineer

> "Phase 6 프로젝트가 포트폴리오가 되어 원하던 회사에 취업했습니다."
> - 이OO, 취준생 → AI Researcher

> "코드를 보면 이제 어떻게 동작하는지 바로 이해가 돼요. 자신감이 생겼습니다!"
> - 박OO, 백엔드 개발자 → MLOps Engineer

### 당신도 할 수 있습니다!

3개월 후, 당신은:
- ✅ 최신 논문을 읽고 구현할 수 있습니다
- ✅ Custom AI 모델을 설계할 수 있습니다
- ✅ 프로덕션에 모델을 배포할 수 있습니다
- ✅ AI 엔지니어/연구원으로 커리어를 시작할 수 있습니다

---

## 🚀 시작하기

준비되셨나요?

```bash
cd phase0-huggingface-ecosystem
cat README.md
```

**Let's become an AI Hero together! 💪**

---

## 📞 문의

- Issues: GitHub Issues 탭
- 피드백: Pull Requests 환영
- 커뮤니티: Discussions 탭

**Happy Learning! 🎉**
