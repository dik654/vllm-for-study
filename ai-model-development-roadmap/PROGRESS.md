# 🎯 학습 진행상황 체크리스트

이 파일을 복사하여 자신의 진행상황을 추적하세요!

## 📅 시작일: ____________
## 📅 목표 종료일: ____________

---

## Phase -1: 수학적 기초 (2주, 선택) ⏱️ 예상: 80시간

### Week 1: 핵심 수학

#### Day 1-2: 선형대수 기초
- [ ] 벡터 연산 (내적, 외적)
- [ ] 행렬 곱셈 밑바닥 구현
- [ ] Transpose, Inverse 이해
- [ ] NumPy로 실습
- **완료일: ____________**

#### Day 3-4: 선형대수 고급
- [ ] Eigenvalues & Eigenvectors
- [ ] SVD 구현
- [ ] PCA 구현
- [ ] LoRA 수학적 유도
- **완료일: ____________**

#### Day 5-6: 미적분 & 최적화
- [ ] Derivatives 손으로 계산
- [ ] Chain rule 연습
- [ ] Backpropagation 유도
- [ ] Gradient descent 구현
- [ ] Adam optimizer 이해
- **완료일: ____________**

### Week 2: AI 수학

#### Day 7-8: 확률 & 통계
- [ ] Gaussian distribution
- [ ] Bayes' theorem
- [ ] MLE 개념
- [ ] Sampling 기법
- **완료일: ____________**

#### Day 9-10: 정보이론
- [ ] Entropy 계산
- [ ] Cross entropy 이해
- [ ] KL divergence 구현
- [ ] Loss functions와의 연결
- **완료일: ____________**

#### Day 11-14: 통합 실습
- [ ] 2-layer NN 밑바닥 구현
- [ ] MNIST 훈련
- [ ] Gradient checking
- [ ] Overfitting test
- **완료일: ____________**

**Phase -1 총 시간: ______ 시간**

---

## Phase 0: HuggingFace 생태계 (1주) ⏱️ 예상: 40시간

### Day 1-2: 핵심 용어
- [ ] Model Card 작성
- [ ] Safetensors vs Pickle 실습
- [ ] ONNX, TorchScript 변환
- [ ] Quantization (FP16, INT8, INT4)
- [ ] LoRA, QLoRA 실습
- **완료일: ____________**

### Day 3-4: 모델 허브 구조
- [ ] config.json 분석
- [ ] tokenizer_config.json 이해
- [ ] generation_config.json 실습
- [ ] Custom config 생성
- [ ] Sharded model 로딩
- **완료일: ____________**

### Day 5-6: 벤치마크 & 데이터셋
- [ ] Perplexity 계산
- [ ] BLEU, ROUGE 구현
- [ ] MMLU 벤치마크 실행
- [ ] HumanEval 테스트
- [ ] Custom 평가 파이프라인
- **완료일: ____________**

### Day 7: 훈련 파이프라인
- [ ] Trainer API vs Native PyTorch
- [ ] DeepSpeed 설정
- [ ] FSDP 이해
- [ ] W&B 연동
- [ ] Flash Attention 사용
- **완료일: ____________**

**Phase 0 총 시간: ______ 시간**

---

## Phase 1: Transformer 완전 구현 (2주) ⏱️ 예상: 80시간

### Week 1: 기초 구현

#### Day 1-2: Attention 메커니즘
- [ ] NumPy로 Attention 구현
- [ ] PyTorch로 효율적 구현
- [ ] Attention 시각화
- [ ] 계산 복잡도 분석
- **완료일: ____________**

#### Day 3-4: Positional Encoding
- [ ] Sinusoidal PE 구현
- [ ] Learned PE 실험
- [ ] RoPE 구현
- [ ] ALiBi 구현
- **완료일: ____________**

### Week 2: 전체 모델

#### Day 5-6: Transformer 블록
- [ ] EncoderBlock 구현
- [ ] DecoderBlock 구현
- [ ] Layer Norm (Pre-LN vs Post-LN)
- [ ] 전체 Transformer 조립
- **완료일: ____________**

#### Day 7: 훈련 파이프라인
- [ ] Teacher forcing 구현
- [ ] Learning rate scheduling
- [ ] Label smoothing
- [ ] Beam search 디코딩
- [ ] 간단한 번역 모델 훈련
- **완료일: ____________**

**Phase 1 총 시간: ______ 시간**

---

## Phase 2: BERT/GPT 마스터 (2주) ⏱️ 예상: 80시간

### Week 3: 토크나이저

#### Day 1-3: BPE 구현
- [ ] BPE 알고리즘 직접 구현
- [ ] Vocab 구축
- [ ] Encoding/Decoding
- [ ] Special tokens 처리
- [ ] Custom 토크나이저 훈련
- **완료일: ____________**

### Week 4: Pre-training

#### Day 4-5: BERT
- [ ] MLM 구현
- [ ] NSP 구현
- [ ] BERT 모델 구조
- [ ] Pre-training
- [ ] Fine-tuning for classification
- **완료일: ____________**

#### Day 6-7: GPT
- [ ] Causal LM objective
- [ ] GPT 아키텍처
- [ ] Greedy, Top-k, Top-p sampling
- [ ] Temperature scaling
- [ ] Few-shot prompting
- **완료일: ____________**

**Phase 2 총 시간: ______ 시간**

---

## Phase 3: Diffusion Models 정복 (3주) ⏱️ 예상: 120시간

### Week 5: 수학적 기초

#### Day 1-3: 확률론 & Forward Process
- [ ] Gaussian 샘플링
- [ ] KL divergence 구현
- [ ] Reparameterization trick
- [ ] Noise scheduling
- [ ] q(x_t|x_0) 구현
- **완료일: ____________**

### Week 6: DDPM

#### Day 4-6: U-Net 구현
- [ ] Residual blocks
- [ ] Self-attention layers
- [ ] Time embedding
- [ ] Skip connections
- [ ] 전체 U-Net 조립
- **완료일: ____________**

#### Day 7-9: Training
- [ ] Noise prediction objective
- [ ] Training loop
- [ ] EMA 구현
- [ ] MNIST 생성 성공
- **완료일: ____________**

### Week 7: Sampling & Conditioning

#### Day 10-12: Sampling 알고리즘
- [ ] DDPM sampling
- [ ] DDIM 구현
- [ ] Classifier guidance
- [ ] Classifier-free guidance
- **완료일: ____________**

#### Day 13-14: Conditional Generation
- [ ] Class conditioning
- [ ] Text conditioning (CLIP)
- [ ] ControlNet 이해
- [ ] Inpainting 구현
- **완료일: ____________**

**Phase 3 총 시간: ______ 시간**

---

## Phase 3.5: Video & Audio Generation (1.5주) ⏱️ 예상: 60시간

### Week 7.5: Video Generation

#### Day 1-3: Video Diffusion
- [ ] Temporal layers 이해
- [ ] 3D attention 구현
- [ ] Stable Video Diffusion 실습
- [ ] AnimateDiff 실행
- [ ] Image-to-video 생성
- **완료일: ____________**

### Week 8: Audio & Music

#### Day 4-6: Audio Generation
- [ ] Waveform vs Spectrogram 이해
- [ ] EnCodec tokenization
- [ ] MusicGen 실습 (텍스트 → 음악)
- [ ] AudioLDM 실습 (사운드 이펙트)
- [ ] Stable Audio 실행
- **완료일: ____________**

#### Day 7-8: Multi-modal
- [ ] ImageBind 이해
- [ ] Cross-modal retrieval
- [ ] Audio → Video 생성
- [ ] Multi-modal embedding 시각화
- [ ] ControlNet 실습
- **완료일: ____________**

**Phase 3.5 총 시간: ______ 시간**

---

## Phase 4: GAN 마스터리 (3주) ⏱️ 예상: 120시간

### Week 8: Vanilla GAN

#### Day 1-3: 기본 GAN
- [ ] Generator/Discriminator 구현
- [ ] BCE loss vs Wasserstein loss
- [ ] Mode collapse 경험
- [ ] MNIST GAN 훈련
- **완료일: ____________**

#### Day 4-5: 안정화 기법
- [ ] Spectral normalization
- [ ] Gradient penalty (WGAN-GP)
- [ ] Instance noise
- [ ] Label smoothing
- **완료일: ____________**

### Week 9: Progressive GAN

#### Day 6-8: Progressive Growing
- [ ] Resolution ladder 구현
- [ ] Smooth fade-in
- [ ] Minibatch statistics
- [ ] Pixel normalization
- **완료일: ____________**

#### Day 9-10: 평가 지표
- [ ] Inception Score 구현
- [ ] FID 계산
- [ ] Precision & Recall
- [ ] PPL 측정
- **완료일: ____________**

### Week 10: StyleGAN

#### Day 11-13: Style-based Generator
- [ ] Mapping network (Z→W)
- [ ] AdaIN 구현
- [ ] Style mixing
- [ ] Noise injection
- [ ] Path length regularization
- **완료일: ____________**

#### Day 14: StyleGAN2/3 개선
- [ ] Weight demodulation
- [ ] Lazy regularization
- [ ] Alias-free generator
- **완료일: ____________**

**Phase 4 총 시간: ______ 시간**

---

## Phase 5: 최신 기술 통합 (2주) ⏱️ 예상: 80시간

### Week 11: 효율화 기법

#### Day 1-3: PEFT
- [ ] LoRA 밑바닥 구현
- [ ] QLoRA 설정
- [ ] Prefix tuning
- [ ] Adapter layers
- [ ] 성능 비교 실험
- **완료일: ____________**

#### Day 4-5: Quantization
- [ ] INT8 quantization
- [ ] GPTQ 구현
- [ ] AWQ 이해
- [ ] 4-bit quantization (bitsandbytes)
- **완료일: ____________**

### Week 12: 프로덕션

#### Day 6-7: 모델 최적화
- [ ] ONNX 변환
- [ ] TensorRT 최적화
- [ ] Model pruning
- [ ] Knowledge distillation
- [ ] torch.compile() 사용
- **완료일: ____________**

#### Day 8-9: 배포
- [ ] TorchServe 설정
- [ ] FastAPI로 REST API
- [ ] Streaming generation
- [ ] Batch inference
- [ ] Model versioning (DVC/MLflow)
- **완료일: ____________**

**Phase 5 총 시간: ______ 시간**

---

## Phase 5.5: LLM Alignment & Advanced Architectures (2주) ⏱️ 예상: 80시간

### Week 1: LLM Alignment

#### Day 1-2: Instruction Tuning
- [ ] SFT (Supervised Fine-Tuning) 이해
- [ ] Instruction dataset 구축
- [ ] Alpaca 스타일 훈련
- [ ] Few-shot prompting
- **완료일: ____________**

#### Day 3-5: RLHF
- [ ] Reward Model 훈련
- [ ] PPO 알고리즘 이해
- [ ] RLHF 파이프라인 구현
- [ ] Preference dataset 분석
- **완료일: ____________**

#### Day 6-7: DPO
- [ ] DPO vs RLHF 비교
- [ ] DPO 구현
- [ ] Preference optimization
- [ ] 성능 평가
- **완료일: ____________**

### Week 2: Advanced Architectures

#### Day 8-10: Mixture of Experts
- [ ] MoE 아키텍처 이해
- [ ] Router 구현
- [ ] Expert parallelism
- [ ] DeepSeek-V3, Mixtral 분석
- **완료일: ____________**

#### Day 11-12: Constitutional AI
- [ ] Self-critique 메커니즘
- [ ] Constitutional prompts
- [ ] Claude 접근 방식 이해
- **완료일: ____________**

#### Day 13-14: 통합 프로젝트
- [ ] Mini ChatGPT 구현
- [ ] SFT → RLHF/DPO 파이프라인
- [ ] 성능 벤치마크
- **완료일: ____________**

**Phase 5.5 총 시간: ______ 시간**

---

## Phase 5.7: Hardware & Systems (1.5주) ⏱️ 예상: 60시간

### Week 1: GPU Programming

#### Day 1-2: GPU Architecture
- [ ] GPU vs CPU 구조 이해
- [ ] CUDA Core vs Tensor Core
- [ ] Memory hierarchy 분석
- [ ] Compute/Memory bound 구분
- **완료일: ____________**

#### Day 3-5: CUDA Programming
- [ ] 첫 CUDA kernel 작성
- [ ] Memory coalescing
- [ ] Shared memory 활용
- [ ] Matrix multiplication 최적화
- [ ] Kernel fusion 이해
- **완료일: ____________**

#### Day 6-7: Advanced Optimization
- [ ] CUDA profiling (Nsight)
- [ ] Attention kernel 최적화
- [ ] FlashAttention 분석
- [ ] 2-3배 속도 향상 달성
- **완료일: ____________**

### Week 2: AI Accelerators

#### Day 8-9: NPU/TPU
- [ ] Google TPU 아키텍처
- [ ] Systolic array 이해
- [ ] Apple Neural Engine
- [ ] NPU 설계 철학
- **완료일: ____________**

#### Day 10-11: Triton Programming
- [ ] Triton 문법 학습
- [ ] Custom operator 작성
- [ ] PyTorch 통합
- [ ] 성능 비교 (CUDA vs Triton)
- **완료일: ____________**

**Phase 5.7 총 시간: ______ 시간**

---

## Phase 6: 실전 프로젝트 (4주) ⏱️ 예상: 160시간

### Week 13: 문제 정의 및 연구

#### Day 1-2: 문제 선정
- [ ] 문제 정의
- [ ] 성공 기준 설정
- [ ] 기존 솔루션 조사
- **완료일: ____________**

#### Day 3-5: 문헌 조사
- [ ] 논문 20개 읽기
- [ ] SOTA 파악
- [ ] Gap analysis
- **완료일: ____________**

#### Day 6-7: 데이터셋
- [ ] 데이터 수집/선정
- [ ] 데이터 전처리
- [ ] Train/Val/Test split
- [ ] 데이터 분석
- **완료일: ____________**

### Week 14: 아키텍처 설계

#### Day 8-10: 설계
- [ ] Baseline 선정
- [ ] 새로운 아이디어 설계
- [ ] 이론적 근거 정리
- [ ] 설계 문서 작성
- **완료일: ____________**

#### Day 11-14: 구현 준비
- [ ] Ablation study 계획
- [ ] Hyperparameter space 정의
- [ ] 코드 구조 설계
- [ ] 개발 환경 설정
- **완료일: ____________**

### Week 15: 모델 구현

#### Day 15-17: 핵심 구현
- [ ] Custom 아키텍처 구현
- [ ] Forward/Backward 검증
- [ ] Unit tests 작성
- **완료일: ____________**

#### Day 18-19: 훈련 파이프라인
- [ ] 데이터 로더
- [ ] 훈련 루프
- [ ] Logging 설정
- [ ] Checkpoint 관리
- **완료일: ____________**

#### Day 20-21: Baseline
- [ ] Baseline 훈련
- [ ] 초기 성능 확인
- [ ] 디버깅
- **완료일: ____________**

### Week 16: 실험 및 분석

#### Day 22-24: Ablation Study
- [ ] 컴포넌트 기여도 측정
- [ ] 실험 결과 정리
- **완료일: ____________**

#### Day 25-26: Hyperparameter Tuning
- [ ] Learning rate 최적화
- [ ] Batch size 조정
- [ ] Regularization 튜닝
- [ ] 최적 설정 도출
- **완료일: ____________**

#### Day 27-28: 최종 평가
- [ ] Test set 평가
- [ ] Baseline 대비 분석
- [ ] Error analysis
- [ ] Efficiency 분석
- [ ] 결과 시각화
- **완료일: ____________**

**Phase 6 총 시간: ______ 시간**

---

## 📊 전체 진행 통계

### 시간 투자
- **총 계획 시간**: 960시간
- **실제 투자 시간**: ______ 시간
- **일일 평균**: ______ 시간
- **주간 평균**: ______ 시간

### 완료율
- **Phase -1** (선택): _____%
- **Phase 0**: _____%
- **Phase 1**: _____%
- **Phase 2**: _____%
- **Phase 3**: _____%
- **Phase 3.5**: _____%
- **Phase 4**: _____%
- **Phase 5**: _____%
- **Phase 5.5**: _____%
- **Phase 5.7**: _____%
- **Phase 6**: _____%
- **전체**: _____%

### 프로젝트 산출물
- [ ] GitHub Repository 생성
- [ ] HuggingFace Hub에 모델 업로드
- [ ] 블로그 포스트 작성
- [ ] 논문/보고서 작성
- [ ] 데모 애플리케이션 배포

---

## 🏆 마일스톤

### Phase 완료
- [ ] Phase -1 완료 (선택) - 날짜: ____________
- [ ] Phase 0 완료 - 날짜: ____________
- [ ] Phase 1 완료 - 날짜: ____________
- [ ] Phase 2 완료 - 날짜: ____________
- [ ] Phase 3 완료 - 날짜: ____________
- [ ] Phase 3.5 완료 - 날짜: ____________
- [ ] Phase 4 완료 - 날짜: ____________
- [ ] Phase 5 완료 - 날짜: ____________
- [ ] Phase 5.5 완료 - 날짜: ____________
- [ ] Phase 5.7 완료 - 날짜: ____________
- [ ] Phase 6 완료 - 날짜: ____________

### 기술 마스터
- [ ] Transformer를 2시간 내 구현 가능
- [ ] 최신 논문을 하루 만에 구현 가능
- [ ] 프로덕션 레벨 모델 배포 가능
- [ ] 모델 디버깅 및 최적화 능숙
- [ ] Custom architecture 설계 가능

### 커뮤니티 기여
- [ ] GitHub stars 10+
- [ ] 블로그 방문자 100+
- [ ] 논문/프로젝트 인용 1+
- [ ] 오픈소스 기여 1+

---

## 💭 회고

### 가장 어려웠던 부분
```
여기에 작성하세요...
```

### 가장 재미있었던 부분
```
여기에 작성하세요...
```

### 예상 밖의 배움
```
여기에 작성하세요...
```

### 앞으로의 계획
```
여기에 작성하세요...
```

---

## 🎉 축하합니다!

모든 Phase를 완료하셨다면, 당신은 이제 진정한 **AI Hero**입니다! 🦸

**완료 날짜**: ____________

**다음 목표**:
1. ______________________________
2. ______________________________
3. ______________________________
