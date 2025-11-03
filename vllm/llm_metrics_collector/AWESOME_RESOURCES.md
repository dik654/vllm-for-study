# Awesome LLM Metrics & Observability Resources

LLM 메트릭 수집 및 관찰성(Observability)을 위한 큐레이션된 리소스 목록입니다.

---

## 📋 목차

1. [표준 및 스펙](#1-표준-및-스펙)
2. [오픈소스 관찰성 플랫폼](#2-오픈소스-관찰성-플랫폼)
3. [평가 프레임워크](#3-평가-프레임워크)
4. [품질 메트릭 도구](#4-품질-메트릭-도구)
5. [안전성 도구](#5-안전성-도구)
6. [RAG 평가](#6-rag-평가)
7. [상용 플랫폼](#7-상용-플랫폼)
8. [유틸리티 라이브러리](#8-유틸리티-라이브러리)

---

## 1. 표준 및 스펙

### OpenTelemetry Semantic Conventions for GenAI
⭐⭐⭐ **2025년 산업 표준으로 자리잡는 중**

**공식 문서**:
- 메인 스펙: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- GenAI 메트릭: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/
- 블로그: https://opentelemetry.io/blog/2024/otel-generative-ai/
- 2025 AI Agent 관찰성: https://opentelemetry.io/blog/2025/ai-agent-observability/

**GitHub**:
- Semantic Conventions 이슈: https://github.com/open-telemetry/semantic-conventions/issues/2664

**커버하는 내용**:
- Span attributes (gen_ai.system, gen_ai.request.model, etc.)
- Event attributes (gen_ai.prompt, gen_ai.completion)
- Metrics (token usage, latency, TTFT, TPOT)
- Agent-specific conventions (tasks, actions, memory, teams)

---

### OpenLLMetry
OpenTelemetry 기반 LLM 관찰성

**GitHub**: https://github.com/traceloop/openllmetry
⭐ 1.8k stars

**특징**:
- LangChain, LlamaIndex, OpenAI, Anthropic, Bedrock 통합
- 자동 instrumentation
- OpenTelemetry 표준 준수

**설치**:
```bash
pip install openllmetry
```

**사용법**:
```python
from traceloop.sdk import Traceloop
Traceloop.init()

# Your LLM calls are automatically traced
```

**문서**: https://www.traceloop.com/docs/openllmetry/

---

## 2. 오픈소스 관찰성 플랫폼

### Phoenix (by Arize AI)
⭐⭐⭐ **강력 추천! 완전 오픈소스**

**GitHub**: https://github.com/Arize-ai/phoenix
⭐ 4.2k stars

**특징**:
- 완전 오픈소스 (ELv2 License)
- LLM traces & spans 시각화
- 내장 hallucination detection
- 임베딩 drift 모니터링
- RAG 평가 도구
- 프롬프트 실험 플레이그라운드

**설치**:
```bash
pip install arize-phoenix
```

**빠른 시작**:
```python
import phoenix as px

# Launch Phoenix UI
px.launch_app()

# Instrument your LLM framework
from phoenix.trace.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument()
```

**문서**: https://docs.arize.com/phoenix/

**통합**:
- LangChain
- LlamaIndex
- OpenAI
- Anthropic
- Bedrock
- vLLM (가능)

---

### LangFuse
오픈소스 LangSmith 대안

**GitHub**: https://github.com/langfuse/langfuse
⭐ 6.4k stars

**특징**:
- Self-hosted 가능 (Docker)
- 프롬프트 관리 & 버전 관리
- 비용 추적
- 사용자 피드백 수집
- 데이터셋 & 평가
- 플레이그라운드

**설치**:
```bash
# Self-hosted
docker run -d --name langfuse \
  -p 3000:3000 \
  langfuse/langfuse:latest

# Client
pip install langfuse
```

**사용법**:
```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-..."
)

trace = langfuse.trace(name="my-llm-app")
generation = trace.generation(
    name="generation",
    model="gpt-4",
    input={"prompt": "..."},
    output={"completion": "..."}
)
```

**문서**: https://langfuse.com/docs
**데모**: https://cloud.langfuse.com/

---

### TruLens (by TruEra)
LLM 앱 평가 및 추적

**GitHub**: https://github.com/truera/trulens
⭐ 2.3k stars

**특징**:
- LangChain, LlamaIndex 완벽 통합
- Feedback functions (hallucination, relevance, toxicity)
- 자동 평가 파이프라인
- 대시보드 UI
- 레코드 기반 앱 평가

**설치**:
```bash
pip install trulens-eval
```

**예시**:
```python
from trulens_eval import TruChain, Feedback, Tru
from trulens_eval.feedback.provider import OpenAI

provider = OpenAI()

# Define feedback functions
f_groundedness = Feedback(provider.groundedness_measure)
    .on_output()
    .on_input()
    .aggregate(np.mean)

# Wrap your chain
tru_chain = TruChain(
    your_langchain,
    app_id='my-app',
    feedbacks=[f_groundedness]
)

# Use as normal
response = tru_chain("What is LangChain?")

# View results
tru.run_dashboard()
```

**문서**: https://www.trulens.org/

---

### LangWatch
실시간 LLM 모니터링

**GitHub**: https://github.com/langwatch/langwatch
⭐ 500+ stars

**특징**:
- 실시간 모니터링
- 오류 추적
- 비용 분석
- 토큰 사용량 추적

**문서**: https://langwatch.ai/docs

---

### OpenLIT
OpenTelemetry 기반 LLM 관찰성

**GitHub**: https://github.com/openlit/openlit
⭐ 3.3k stars

**특징**:
- 원클릭 통합 (1줄 코드)
- GPU 모니터링
- 비용 추적
- 프롬프트 관리

**설치**:
```bash
pip install openlit
```

**사용법**:
```python
import openlit
openlit.init()

# That's it! All LLM calls are now traced
```

---

## 3. 평가 프레임워크

### DeepEval
LLM 평가 프레임워크

**GitHub**: https://github.com/confident-ai/deepeval
⭐ 3.5k stars

**특징**:
- 14+ 내장 평가 메트릭
- Pytest 통합
- 합성 데이터셋 생성
- CI/CD 통합

**설치**:
```bash
pip install deepeval
```

**메트릭**:
```python
from deepeval import evaluate
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is LangChain?",
    actual_output="LangChain is a blockchain for languages",
    context=["LangChain is a framework for building LLM apps"]
)

metrics = [
    HallucinationMetric(threshold=0.5),
    AnswerRelevancyMetric(threshold=0.7)
]

evaluate(test_cases=[test_case], metrics=metrics)
```

**문서**: https://docs.confident-ai.com/

---

### PromptTools
프롬프트 실험 도구

**GitHub**: https://github.com/hegelai/prompttools
⭐ 2.7k stars

**특징**:
- 프롬프트 A/B 테스트
- 여러 모델 동시 평가
- 자동 벤치마킹

**설치**:
```bash
pip install prompttools
```

---

### Evaluate (HuggingFace)
다목적 ML 평가 라이브러리

**GitHub**: https://github.com/huggingface/evaluate
⭐ 2k stars

**특징**:
- 50+ 평가 메트릭
- ROUGE, BLEU, BERTScore 등
- 모델 카드 생성

**설치**:
```bash
pip install evaluate
```

**사용법**:
```python
import evaluate

# ROUGE
rouge = evaluate.load('rouge')
results = rouge.compute(predictions=predictions, references=references)

# BERTScore
bertscore = evaluate.load("bertscore")
results = bertscore.compute(predictions=predictions, references=references, lang="en")
```

**문서**: https://huggingface.co/docs/evaluate/

---

### RAGAS (RAG Assessment)
RAG 시스템 평가

**GitHub**: https://github.com/explodinggradients/ragas
⭐ 8k stars

**특징**:
- RAG 전용 메트릭
- Faithfulness, Answer Relevancy, Context Precision, Context Recall
- 합성 테스트 데이터 생성

**설치**:
```bash
pip install ragas
```

**메트릭**:
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
)

print(results)
```

**문서**: https://docs.ragas.io/

---

## 4. 품질 메트릭 도구

### BERTScore
의미적 유사도 평가

**GitHub**: https://github.com/Tiiiger/bert_score
⭐ 1.6k stars

**특징**:
- BERT 임베딩 기반
- 의미적 유사도 측정
- 언어별 최적화

**설치**:
```bash
pip install bert-score
```

**사용법**:
```python
from bert_score import score

P, R, F1 = score(
    candidates,
    references,
    lang="en",
    verbose=True
)
```

**논문**: https://arxiv.org/abs/1904.09675

---

### ROUGE
요약 평가 메트릭

**GitHub**: https://github.com/google-research/google-research/tree/master/rouge

**Python 구현**:
```bash
pip install rouge-score
```

**사용법**:
```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score('reference text', 'hypothesis text')

print(scores)
# {'rouge1': Score(precision=..., recall=..., fmeasure=...)}
```

---

### SelfCheckGPT
환각(Hallucination) 감지

**GitHub**: https://github.com/potsawee/selfcheckgpt
⭐ 800+ stars

**방법론**:
- 동일한 프롬프트로 여러 번 생성
- 샘플 간 일관성 측정
- 일관성 낮으면 → 환각 가능성 높음

**설치**:
```bash
pip install selfcheckgpt
```

**사용법**:
```python
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

selfcheck_nli = SelfCheckNLI(device='cuda')

# Generate multiple samples
samples = [generate(prompt) for _ in range(5)]

# Check consistency
scores = selfcheck_nli.predict(
    sentences=[response],  # sentences to check
    sampled_passages=samples
)
```

**논문**: https://arxiv.org/abs/2303.08896

---

### Hallucination Detection - Awesome List
환각 감지 연구 논문 및 도구 모음

**GitHub**: https://github.com/EdinburghNLP/awesome-hallucination-detection
⭐ 1.2k stars

**포함 내용**:
- 최신 연구 논문
- 감지 방법론
- 벤치마크 데이터셋
- 평가 도구

**카테고리**:
- Faithfulness (사실성)
- Factuality (정확성)
- Attribution (출처 추적)
- Consistency (일관성)

---

### Hallucinations Leaderboard
LLM 환각 벤치마크

**HuggingFace**: https://huggingface.co/blog/leaderboard-hallucinations

**데이터셋**:
- TruthfulQA
- HaluEval
- FactualityPrompts

---

## 5. 안전성 도구

### Detoxify
독성 감지

**GitHub**: https://github.com/unitaryai/detoxify
⭐ 900+ stars

**특징**:
- BERT 기반 분류기
- 6가지 독성 카테고리
- 다국어 지원

**설치**:
```bash
pip install detoxify
```

**사용법**:
```python
from detoxify import Detoxify

model = Detoxify('original')
results = model.predict('your text here')

print(results)
# {
#     'toxicity': 0.01,
#     'severe_toxicity': 0.001,
#     'obscene': 0.02,
#     'threat': 0.001,
#     'insult': 0.03,
#     'identity_attack': 0.001
# }
```

**모델**:
- `original`: 영어
- `unbiased`: 편향 제거 버전
- `multilingual`: 다국어

---

### Presidio (by Microsoft)
PII (개인정보) 감지 및 익명화

**GitHub**: https://github.com/microsoft/presidio
⭐ 3.8k stars

**구성요소**:
- `presidio-analyzer`: PII 감지
- `presidio-anonymizer`: PII 익명화

**설치**:
```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

**사용법**:
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Analyze
analyzer = AnalyzerEngine()
results = analyzer.analyze(
    text="John's phone is 212-555-5555 and email is john@example.com",
    language='en'
)

print(results)
# [type: PHONE_NUMBER, start: 18, end: 30, score: 0.75]
# [type: EMAIL_ADDRESS, start: 44, end: 62, score: 0.95]

# Anonymize
anonymizer = AnonymizerEngine()
anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
print(anonymized.text)
# "John's phone is <PHONE_NUMBER> and email is <EMAIL_ADDRESS>"
```

**엔티티 타입**:
- CREDIT_CARD, CRYPTO, DATE_TIME, EMAIL_ADDRESS
- IBAN_CODE, IP_ADDRESS, PERSON, PHONE_NUMBER
- MEDICAL_LICENSE, URL, 등

**문서**: https://microsoft.github.io/presidio/

---

### LangKit (by WhyLabs)
프롬프트 인젝션 및 LLM 보안

**GitHub**: https://github.com/whylabs/langkit
⭐ 500+ stars

**특징**:
- Prompt injection 감지
- Jailbreak 시도 감지
- Toxicity, sentiment 분석
- Regex 패턴 매칭

**설치**:
```bash
pip install langkit
```

**사용법**:
```python
import langkit

# Extract metrics
from langkit import extract
metrics = extract({"prompt": "Ignore previous instructions..."})

print(metrics)
# {
#     'prompt.injection_score': 0.85,
#     'prompt.sentiment': -0.3,
#     'prompt.toxicity': 0.1
# }
```

---

### NeMo Guardrails (by NVIDIA)
LLM 가드레일 프레임워크

**GitHub**: https://github.com/NVIDIA/NeMo-Guardrails
⭐ 4.3k stars

**특징**:
- 입출력 가드레일
- 토픽 제한
- 사실성 검증
- Jailbreak 방지

**설치**:
```bash
pip install nemoguardrails
```

**문서**: https://docs.nvidia.com/nemo/guardrails/

---

### Guardrails AI
구조화된 LLM 출력 검증

**GitHub**: https://github.com/guardrails-ai/guardrails
⭐ 4.2k stars

**특징**:
- 출력 구조 검증
- 스키마 강제
- 재시도 로직

**설치**:
```bash
pip install guardrails-ai
```

---

## 6. RAG 평가

### RAGAS
(위 "평가 프레임워크" 섹션 참조)

**GitHub**: https://github.com/explodinggradients/ragas
⭐ 8k stars

---

### LlamaIndex Evaluation
RAG 파이프라인 평가

**GitHub**: https://github.com/run-llama/llama_index
⭐ 37k stars

**특징**:
- 내장 평가 모듈
- Retrieval 평가
- Response 평가

**문서**: https://docs.llamaindex.ai/en/stable/module_guides/evaluating/

**예시**:
```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator
)

faithfulness_evaluator = FaithfulnessEvaluator()
relevancy_evaluator = RelevancyEvaluator()

# Evaluate
eval_result = faithfulness_evaluator.evaluate_response(
    query=query,
    response=response
)
```

---

### Context Precision/Recall
RAG 검색 품질 평가

**구현**: RAGAS에 포함

**메트릭**:
- **Context Precision**: 검색된 문서의 순서가 적절한가?
- **Context Recall**: 필요한 정보를 모두 검색했는가?

---

## 7. 상용 플랫폼

### LangSmith (by LangChain)
**URL**: https://smith.langchain.com/

**특징**:
- LangChain 완벽 통합
- 프롬프트 플레이그라운드
- 데이터셋 & 평가
- 디버깅 UI

**가격**:
- Free: 5K traces/월
- Plus: $39/월
- Enterprise: 커스텀

**문서**: https://docs.smith.langchain.com/

---

### Weights & Biases (W&B)
**GitHub**: https://github.com/wandb/wandb
⭐ 9.2k stars

**URL**: https://wandb.ai/

**특징**:
- MLOps + LLMOps 통합
- 실험 추적
- 프롬프트 관리
- 모델 레지스트리

**설치**:
```bash
pip install wandb
```

**Prompts 문서**: https://docs.wandb.ai/guides/prompts

---

### Arize (Enterprise)
**URL**: https://arize.com/

**특징**:
- ML + LLM 통합 모니터링
- 임베딩 drift 감지
- Root cause analysis
- 엔터프라이즈급 확장성

---

### WhyLabs
**URL**: https://whylabs.ai/

**특징**:
- LangKit 통합
- Compliance 중심
- 데이터 품질 모니터링
- 이상 탐지

---

### Helicone
**GitHub**: https://github.com/Helicone/helicone
⭐ 1.4k stars

**URL**: https://www.helicone.ai/

**특징**:
- OpenAI 프록시
- 비용 추적
- 캐싱
- 레이트 리밋

---

## 8. 유틸리티 라이브러리

### tiktoken (by OpenAI)
토큰 카운팅

**GitHub**: https://github.com/openai/tiktoken
⭐ 12.5k stars

**설치**:
```bash
pip install tiktoken
```

**사용법**:
```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")
tokens = encoding.encode("Hello, world!")
print(len(tokens))  # 3
```

---

### LiteLLM
통합 LLM API

**GitHub**: https://github.com/BerriAI/litellm
⭐ 14k stars

**특징**:
- 100+ LLM 지원 (OpenAI, Anthropic, Bedrock, Azure, 등)
- 통합 API
- 비용 추적
- 프록시 서버

**설치**:
```bash
pip install litellm
```

**사용법**:
```python
from litellm import completion

# OpenAI
response = completion(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])

# Anthropic (same interface)
response = completion(model="claude-3-opus-20240229", messages=[...])

# Bedrock (same interface)
response = completion(model="bedrock/anthropic.claude-v2", messages=[...])
```

---

### LMQL
언어 모델 쿼리 언어

**GitHub**: https://github.com/eth-sri/lmql
⭐ 3.6k stars

**특징**:
- SQL-like 구문
- 구조화된 출력
- 제약 조건

**설치**:
```bash
pip install lmql
```

---

### Instructor
구조화된 LLM 출력

**GitHub**: https://github.com/jxnl/instructor
⭐ 8k stars

**특징**:
- Pydantic 모델로 출력 구조화
- 자동 재시도
- 검증

**설치**:
```bash
pip install instructor
```

**사용법**:
```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())

class User(BaseModel):
    name: str
    age: int

user = client.chat.completions.create(
    model="gpt-4",
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30 years old"}]
)

print(user)  # User(name='John', age=30)
```

---

## 9. 벤치마크 & 리더보드

### HELM (Holistic Evaluation of Language Models)
**GitHub**: https://github.com/stanford-crfm/helm
⭐ 2k stars

**URL**: https://crfm.stanford.edu/helm/

**특징**:
- 포괄적인 LLM 벤치마크
- 42+ 시나리오
- 다차원 평가

---

### OpenLLM Leaderboard
**HuggingFace**: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

**벤치마크**:
- MMLU, GSM8K, TruthfulQA, Winogrande, ARC

---

### MTEB (Massive Text Embedding Benchmark)
**GitHub**: https://github.com/embeddings-benchmark/mteb
⭐ 2k stars

**URL**: https://huggingface.co/spaces/mteb/leaderboard

**특징**:
- 임베딩 모델 벤치마크
- 58개 데이터셋
- 8개 태스크

---

## 10. 추가 리소스

### Papers with Code
**URL**: https://paperswithcode.com/

**카테고리**:
- Natural Language Processing
- Question Answering
- Language Modeling

---

### LLM Paper 모음
**Awesome LLM**: https://github.com/Hannibal046/Awesome-LLM
⭐ 19k stars

---

### Prompt Engineering Guide
**GitHub**: https://github.com/dair-ai/Prompt-Engineering-Guide
⭐ 50k stars

**URL**: https://www.promptingguide.ai/

---

## 📊 빠른 선택 가이드

### 오픈소스 관찰성 플랫폼이 필요하다면?
1. **Phoenix** - 가장 완전한 기능
2. **LangFuse** - Self-hosted, 프롬프트 관리
3. **TruLens** - LangChain/LlamaIndex 사용 시

### 평가가 필요하다면?
1. **DeepEval** - 종합적인 평가
2. **RAGAS** - RAG 시스템 전용
3. **TruLens** - 기존 프레임워크 통합

### Hallucination 감지가 필요하다면?
1. **SelfCheckGPT** - 일관성 기반
2. **TruLens** - Groundedness feedback
3. **Phoenix** - 내장 도구

### 안전성 체크가 필요하다면?
1. **Detoxify** - 독성
2. **Presidio** - PII
3. **LangKit** - Prompt injection
4. **NeMo Guardrails** - 종합 가드레일

### RAG 평가가 필요하다면?
1. **RAGAS** - 전용 메트릭
2. **LlamaIndex Evaluation** - 통합 솔루션
3. **TruLens** - 상세 분석

---

## 🔗 종합 추천

### 시작하는 경우
1. **OpenLLMetry** - 자동 tracing
2. **Phoenix** - 무료 대시보드
3. **DeepEval** - 빠른 평가

### 프로덕션 환경
1. **LangFuse** (Self-hosted) - 비용 효율
2. **Phoenix** - 모니터링
3. **Detoxify + Presidio** - 안전성

### 엔터프라이즈
1. **LangSmith** 또는 **Arize** - 상용
2. **NeMo Guardrails** - 보안
3. **커스텀 평가 파이프라인**

---

## 📝 기여

이 문서에 추가하고 싶은 도구나 리소스가 있다면 PR을 보내주세요!

**마지막 업데이트**: 2025-11-03
