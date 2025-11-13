# Phase 0: HuggingFace 생태계 완벽 이해

## 🎯 학습 목표

HuggingFace는 현대 AI 개발의 중심 허브입니다. 이 Phase에서는:

1. ✅ HuggingFace Hub의 모든 파일과 용어를 완벽히 이해
2. ✅ 모델 포맷과 최적화 기법 마스터
3. ✅ 벤치마크와 평가 지표 실습
4. ✅ 훈련 파이프라인 구축 능력

## 📚 학습 내용

### Day 1-2: [핵심 용어 마스터](./01-core-terminology.md)
- Model Card와 문서화 표준
- Safetensors vs Pickle 포맷
- 모델 변환 (GGUF, ONNX, TorchScript)
- Quantization 기법 (fp16/bf16/int8/int4)
- Parameter-Efficient Fine-Tuning (LoRA, QLoRA, PEFT)

### Day 3-4: [모델 허브 구조](./02-model-hub-structure.md)
- config.json 완벽 분석
- tokenizer_config.json 파라미터
- generation_config.json 이해
- Sharded 모델 로딩 메커니즘

### Day 5-6: [벤치마크와 데이터셋](./03-benchmarks-datasets.md)
- 평가 지표 (Perplexity, BLEU, ROUGE, FID 등)
- 주요 벤치마크 (MMLU, HumanEval, COCO)
- 데이터셋 로딩 및 전처리
- Custom 평가 파이프라인 구축

### Day 7: [훈련 파이프라인](./04-training-pipeline.md)
- Trainer API vs Native PyTorch
- 분산 훈련 (DeepSpeed, FSDP)
- 메모리 최적화 (Gradient Checkpointing, Flash Attention)
- 로깅 및 모니터링 (W&B, TensorBoard)

## 🛠️ 실습 프로젝트

### 프로젝트 1: 모델 분석 도구
HuggingFace Hub의 모델을 자동으로 분석하는 CLI 도구 구축

### 프로젝트 2: Custom Benchmark
자신만의 평가 벤치마크 설계 및 구현

### 프로젝트 3: 훈련 템플릿
재사용 가능한 훈련 파이프라인 템플릿 구축

## 📈 학습 진도

- [ ] Day 1-2 완료
- [ ] Day 3-4 완료
- [ ] Day 5-6 완료
- [ ] Day 7 완료
- [ ] 실습 프로젝트 완료

## ⏭️ 다음 단계

Phase 0을 완료했다면 [Phase 1: Transformer 구현](../phase1-transformer/)으로 진행하세요!
