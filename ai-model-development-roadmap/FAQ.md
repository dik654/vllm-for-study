# ❓ Frequently Asked Questions (FAQ)

AI 모델 개발 로드맵에 대해 자주 묻는 질문들입니다.

---

## 🎯 시작하기

### Q: 이 로드맵은 누구를 위한 것인가요?

**A**: 다음과 같은 분들을 위한 것입니다:
- **초보자**: 프로그래밍은 할 줄 알지만 AI/ML은 처음
- **중급자**: 기본 ML은 알지만 Transformer, LLM은 모름
- **고급자**: 최신 기술 (GQA, LoRA, Speculative Decoding) 학습 원함
- **연구자/엔지니어**: 프로덕션 레벨 구현 필요

### Q: 사전 지식이 필요한가요?

**A**: 최소 요구사항:
- ✅ **필수**: Python 프로그래밍 (for loop, class, function)
- ✅ **필수**: 기본 수학 (고등학교 수준 - 행렬 곱셈, 미분 개념)
- ⚠️ **권장**: Linear Algebra, Calculus 기초
- ⚠️ **권장**: PyTorch 또는 TensorFlow 경험
- ❌ **불필요**: 고급 수학, PhD 수준 지식

**완전 초보라면?** → Phase -1 (수학 기초)부터 시작하세요!

### Q: 완료하는 데 얼마나 걸리나요?

**A**: 투자 시간에 따라 다릅니다:

```
풀타임 학습 (주 40시간):
  - 기본만: 2-3개월
  - 전체: 4-6개월

파트타임 (주 10-15시간):
  - 기본만: 4-6개월
  - 전체: 8-12개월

주말만 (주 5시간):
  - 기본만: 8-12개월
  - 전체: 18-24개월
```

**팁**: 매일 조금씩이 주말 몰아서보다 효과적!

---

## 💻 기술적 질문

### Q: GPU가 필요한가요?

**A**: Phase에 따라 다릅니다:

**불필요 (CPU 충분)**:
- Phase -1: 수학 기초
- Phase 0: HuggingFace 기초
- Phase 1: Transformer 이해 (작은 모델)

**권장 (하지만 Colab 무료 GPU로 가능)**:
- Phase 2: BERT/GPT 훈련
- Phase 3: Diffusion Models
- Phase 4: GANs

**필수 (자체 GPU 또는 Cloud 필요)**:
- Phase 5: Large model fine-tuning
- Phase 6: 실전 프로젝트

**대안**:
1. **Google Colab** (무료 T4 GPU, 제한적)
2. **Kaggle Notebooks** (무료 P100 GPU)
3. **Cloud GPU** (AWS, GCP, Lambda Labs)
   - 비용: 시간당 $0.5-$3
4. **RTX 3060/3070** (가성비 좋음, ~$400-600)

### Q: 어떤 GPU를 사야 하나요?

**A**: 예산별 추천:

```
$500 이하:
  → RTX 3060 (12GB VRAM)
  - Phase 1-4 충분
  - Small model fine-tuning 가능

$600-1000:
  → RTX 3070/4070 Ti (16GB VRAM)
  - Phase 1-5 대부분 가능
  - Medium model fine-tuning

$1000-2000:
  → RTX 4090 (24GB VRAM)
  - 거의 모든 작업 가능
  - 7B model fine-tuning with QLoRA

$2000+:
  → A5000/A6000 또는 Cloud
  - Professional level
  - 13B+ model fine-tuning
```

**초보자라면?** → GPU 없이 시작, 나중에 필요할 때 구매/대여

### Q: Mac M1/M2/M3로 가능한가요?

**A**: **가능합니다!** (제한적)

**장점**:
- PyTorch MPS backend 지원
- Unified memory (RAM = VRAM)
- 전력 효율 좋음

**단점**:
- 일부 CUDA 전용 라이브러리 안됨
- FlashAttention, vLLM 등 불가
- 큰 모델은 느림

**추천**:
- Phase 1-3: 충분히 가능
- Phase 4-5: 가능하지만 느림
- 프로덕션: Cloud GPU 권장

---

## 📚 학습 방법

### Q: 어떤 순서로 공부해야 하나요?

**A**: 3가지 경로 중 선택:

**경로 1: 탄탄한 기초 (추천)**
```
Phase -1 (수학) → Phase 0 → Phase 1 → Phase 2 → ...
```
- 장점: 깊은 이해
- 단점: 시간 오래 걸림
- 적합: 시간 여유 있고 수학 약한 사람

**경로 2: 빠른 실전 (실용적)**
```
Phase 0 → Phase 1 (기본만) → Phase 2 → 프로젝트
```
- 장점: 빠르게 결과물
- 단점: 이론 부족할 수 있음
- 적합: 빨리 프로젝트 시작하고 싶은 사람

**경로 3: 맞춤형 (목표 중심)**
```
관심 있는 Phase만 선택적으로
```
- 예: LLM만 → Phase 1.5, 2.4, 5
- 예: Vision만 → Phase 1.7, 3
- 적합: 특정 목표가 명확한 사람

### Q: 막히면 어떻게 하나요?

**A**: 단계별 접근:

**Level 1: 문서 다시 읽기**
- 천천히, 코드 라인별로
- 손으로 그림 그려보기
- 작은 예제 만들어보기

**Level 2: 다른 리소스 찾기**
- YouTube 검색 (The Illustrated Transformer 등)
- 블로그 포스트
- 관련 논문 (논문이 때로는 더 명확)

**Level 3: 커뮤니티에 질문**
- GitHub Issues에 질문
- Reddit r/MachineLearning
- Stack Overflow
- HuggingFace Forums

**Level 4: 잠시 쉬기**
- 산책하기 🚶
- 자기 💤
- 다음날 다시 보면 보임!

**중요**: 막히는 것은 정상입니다! 모두가 겪습니다.

### Q: 수학이 너무 어려워요

**A**: 실용적 접근:

**전략 1: Just-in-time 학습**
- 필요할 때 해당 수학만 배우기
- 예: Attention 배울 때 → 행렬 곱셈만

**전략 2: 코드 먼저, 수학 나중**
- 일단 코드로 구현
- 작동하는 것 확인
- 나중에 수학적 이해

**전략 3: 시각화 활용**
- 3Blue1Brown YouTube
- distill.pub 논문들
- 직접 그림 그리기

**핵심 수학만 집중**:
- Matrix multiplication (행렬 곱셈)
- Derivatives (미분)
- Softmax
- 나머지는 선택적!

---

## 🎯 프로젝트 관련

### Q: 어떤 프로젝트를 해야 하나요?

**A**: 관심사와 수준에 따라:

**초보자 프로젝트** (Phase 1-2 후):
- Shakespeare 스타일 텍스트 생성
- 간단한 챗봇 (GPT-2 fine-tuning)
- 감정 분석 (BERT fine-tuning)
- 이미지 분류 (ViT)

**중급자 프로젝트** (Phase 3-4 후):
- Custom 도메인 챗봇 (의료, 법률 등)
- 코드 생성 모델
- 요약/번역 모델
- 이미지 생성 (Stable Diffusion fine-tuning)

**고급자 프로젝트** (Phase 5-6 후):
- 새로운 아키텍처 제안
- Efficient inference 시스템
- Multi-modal model
- 논문 재현 + 개선

**프로젝트 선택 팁**:
1. **관심사**: 좋아하는 분야 선택
2. **실용성**: 실제 사용할 수 있는 것
3. **적절한 난이도**: 너무 쉽지도 어렵지도 않게
4. **포트폴리오**: 취업/이직에 도움될 것

### Q: GitHub에 올려도 되나요?

**A**: **절대 환영!** 🎉

**오픈소스 권장사항**:
1. **README.md 작성**
   - 프로젝트 설명
   - 설치 방법
   - 사용 예제

2. **라이선스 추가**
   - MIT License (가장 자유로움)
   - Apache 2.0
   - GPL (상업적 제한)

3. **코드 정리**
   - 주석 달기
   - 함수/변수명 명확하게
   - Requirements.txt 포함

4. **데모 포함**
   - Colab notebook
   - HuggingFace Space
   - GIF/동영상

**참고**: 이 로드맵 자체도 오픈소스입니다!

---

## 💼 커리어 관련

### Q: 취업에 도움이 되나요?

**A**: **매우 도움됩니다!**

**이 로드맵 완료 후 지원 가능한 포지션**:
- ML Engineer
- AI Researcher
- LLM Engineer
- Computer Vision Engineer
- Data Scientist (AI 특화)

**포트폴리오 구성**:
1. **GitHub**:
   - 3-5개 프로젝트
   - Clean code
   - 좋은 documentation

2. **블로그**:
   - 배운 내용 정리
   - 프로젝트 설명
   - 튜토리얼 작성

3. **논문**:
   - 재현 (reproduction)
   - 개선 (improvement)
   - arXiv 제출 (선택)

4. **오픈소스 기여**:
   - HuggingFace Transformers
   - PyTorch
   - 다른 ML 라이브러리

### Q: 학위가 필요한가요?

**A**: **필요 없습니다!** (대부분의 경우)

**포지션별**:

**ML Engineer**: 학위 불필요
- 중요: 구현 능력, 프로젝트 경험
- Portfolio > Degree

**AI Researcher**: 석사/박사 권장
- 논문 작성 능력 중요
- Academic background 도움됨

**Startup**: 학위 거의 무관
- Ability to ship 가장 중요
- 실력 증명하면 됨

**대기업 (Google, Meta)**: 학위 도움됨
- 하지만 exceptional portfolio면 가능
- 최근 no-degree hire 증가 추세

**핵심**: **실력이 가장 중요!**

---

## 🤖 기술 세부사항

### Q: PyTorch vs TensorFlow?

**A**: **PyTorch 추천** (이 로드맵 기준)

**이유**:
- 🔬 연구: 압도적 점유율
- 📖 학습: 더 직관적 (Pythonic)
- 🚀 최신 기술: 먼저 PyTorch로 나옴
- 👥 커뮤니티: 활발함

**TensorFlow는?**:
- 프로덕션 배포에 강함 (TensorFlow Serving)
- 모바일 (TensorFlow Lite)
- JavaScript (TensorFlow.js)

**결론**: PyTorch로 배우고, 필요시 TensorFlow 추가 학습

### Q: Transformer, BERT, GPT 차이는?

**A**: 계층적 관계:

```
Transformer (Architecture)
  ├─ Encoder Only → BERT
  │   └─ Use: Classification, NER, Q&A
  │
  ├─ Decoder Only → GPT
  │   └─ Use: Text generation, Chat
  │
  └─ Encoder-Decoder → T5, BART
      └─ Use: Translation, Summarization
```

**간단히**:
- **Transformer**: 전체 아키텍처 (Attention + FFN + ...)
- **BERT**: Encoder만 사용 (양방향)
- **GPT**: Decoder만 사용 (단방향)

### Q: Fine-tuning vs Pre-training 차이는?

**A**:

**Pre-training** (사전 학습):
- 목적: 일반적 언어 이해 학습
- 데이터: 대규모 (수십~수백 GB)
- 시간: 수주~수개월
- 비용: 수십만~수백만 달러
- 예: GPT-3 pre-training

**Fine-tuning** (미세 조정):
- 목적: 특정 task 적응
- 데이터: 작음 (수 MB~GB)
- 시간: 수 시간~수일
- 비용: 수십~수백 달러
- 예: GPT-3 → ChatGPT (instruction tuning)

**대부분의 경우**: Pre-trained 모델을 다운받아 fine-tuning

### Q: LoRA가 뭔가요?

**A**: **효율적 fine-tuning 기법**

**문제**: 7B 모델 fine-tuning = 84GB 메모리 필요
**해결**: LoRA = 0.1% 파라미터만 학습, 메모리 6GB

**수학**:
```
Full FT:  W_new = W_old + ΔW (모든 weight 변경)
LoRA:     W_new = W_old + BA (작은 행렬로 근사)
```

**장점**:
- 메모리 90% 절감
- 속도 5배 빠름
- 성능 98% 유지

**사용 사례**: 작은 GPU로 대형 모델 fine-tuning

자세한 내용: `phase5-modern-techniques/05-efficient-finetuning.md`

---

## 📖 리소스 관련

### Q: 추천 책이 있나요?

**A**: 수준별 추천:

**입문**:
- "Deep Learning" by Ian Goodfellow (무료 온라인)
- "Hands-On Machine Learning" by Aurélien Géron

**중급**:
- "Speech and Language Processing" by Jurafsky & Martin
- "Dive into Deep Learning" (d2l.ai, 무료)

**고급**:
- 논문 직접 읽기 (Papers with Code)
- arXiv daily

### Q: 온라인 강의 추천해주세요

**A**:

**무료**:
- **Fast.ai** - 실용적, 코드 중심
- **Stanford CS224N** - NLP 이론
- **MIT 6.S191** - DL 입문
- **3Blue1Brown** - 시각적 설명

**유료** (선택적):
- **Coursera Deep Learning Specialization** (Andrew Ng)
- **Udacity Deep Learning Nanodegree**

**이 로드맵만으로도 충분합니다!**

### Q: 논문을 어떻게 읽나요?

**A**: 효율적 논문 읽기 3-pass:

**Pass 1 (5분): 훑어보기**
- Abstract, Introduction, Conclusion만
- 뭐에 관한 논문인지 파악
- 읽을 가치 있는지 판단

**Pass 2 (30분): 이해하기**
- 그림, 표, 수식 중심으로
- Method section 읽기
- 핵심 아이디어 파악

**Pass 3 (2-3시간): 완전 이해**
- 처음부터 끝까지 상세히
- 수식 유도, 코드 구현
- 재현 가능한 수준

**팁**: 모든 논문을 Pass 3까지 읽을 필요 없음!

---

## 🎯 동기부여

### Q: 너무 어려워서 포기하고 싶어요

**A**: **정상입니다!** 모두가 느낌니다.

**이것을 기억하세요**:
1. **어려움 = 성장**
   - 쉬우면 배우는 게 없음
   - 버티면 실력 향상

2. **작은 성공 축하**
   - Attention 구현 완료 ✅
   - 첫 loss 감소 ✅
   - 의미있는 text 생성 ✅

3. **휴식도 중요**
   - 하루 쉬기
   - 산책하기
   - 다른 취미

4. **커뮤니티 활용**
   - 혼자가 아닙니다
   - 같이 공부하는 사람 찾기
   - Discord, Slack 커뮤니티

**명언**:
> "The master has failed more times than the beginner has even tried."
> - 마스터는 초보자가 시도한 횟수보다 더 많이 실패했다.

### Q: 나이가 많은데 시작해도 될까요?

**A**: **물론입니다!**

- AI는 계속 진화하는 분야
- 나이보다 **열정**과 **끈기**가 중요
- 많은 성공 사례들 (30대, 40대, 50대 career change)

**장점**:
- 인생 경험 → 더 나은 문제 정의
- Domain knowledge → Unique perspective
- Maturity → 더 나은 학습 전략

**단점**:
- 체력? → 꾸준히 하면 됨
- 시간? → 효율적으로 사용

**핵심**: 시작이 늦은 게 아니라, 시작하지 않는 게 문제!

---

## 💬 기타

### Q: 이 로드맵을 만든 이유는?

**A**: 다음을 위해:
1. 체계적 학습 경로 제공
2. 최신 기술까지 포함 (GQA, LoRA 등)
3. 이론 + 실습 균형
4. 한글 설명으로 접근성 향상

### Q: 기여할 수 있나요?

**A**: **환영합니다!** 🎉

**기여 방법**:
1. **Typo 수정**: PR 보내기
2. **코드 개선**: Better implementation 제안
3. **번역**: 영어 버전 작성
4. **새 내용**: 빠진 주제 추가
5. **피드백**: Issues에 의견 남기기

### Q: 로드맵이 업데이트되나요?

**A**: **계속 업데이트됩니다!**

**최근 추가**:
- ✅ Vision Transformers (ViT)
- ✅ LoRA/QLoRA 상세 가이드
- ✅ Speculative Decoding
- ✅ State Space Models (Mamba)
- ✅ Korean comments in code

**계획 중**:
- Multi-modal models (CLIP, LLaVA)
- RLHF 상세 가이드
- Mixture of Experts
- Long-context models

---

## 🚀 마지막 조언

**기억하세요**:
1. **완벽보다 진행**
2. **매일 조금씩**
3. **즐기세요!**
4. **커뮤니티 활용**
5. **포기하지 마세요**

**You can do this!** 💪

더 궁금한 점이 있다면:
- GitHub Issues에 질문
- [QUICK-START.md](QUICK-START.md)에서 빠른 시작
- [NEXT-STEPS.md](NEXT-STEPS.md)에서 상세 계획

**Happy Learning!** 🎉
