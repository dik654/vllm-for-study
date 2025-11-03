# LLM 사용량 평가를 위한 추가 개념 및 리소스

현재 구현: 기본적인 **운영 메트릭** (토큰, 레이턴시, 비용)
추가 필요: **품질 메트릭**, **안전성 메트릭**, **비즈니스 메트릭**, **표준화**

---

## 1. 산업 표준 및 프레임워크 🌐

### 1.1 OpenTelemetry Semantic Conventions for GenAI ⭐⭐⭐
**가장 중요! 2025년 표준으로 자리잡는 중**

**공식 문서**:
- https://opentelemetry.io/docs/specs/semconv/gen-ai/
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/

**주요 내용**:
```yaml
# Span Attributes (추적)
gen_ai.system: "openai"
gen_ai.request.model: "gpt-4"
gen_ai.request.max_tokens: 100
gen_ai.request.temperature: 0.7
gen_ai.usage.input_tokens: 50
gen_ai.usage.output_tokens: 30

# Event Attributes (입출력)
gen_ai.prompt: "user prompt"
gen_ai.completion: "model response"

# Metrics (메트릭)
gen_ai.client.token.usage          # 토큰 사용량
gen_ai.client.operation.duration   # 요청 시간
gen_ai.server.time_per_output_token # TPOT
gen_ai.server.time_to_first_token   # TTFT
```

**우리 시스템과의 통합**:
- 우리의 `RequestMetrics` 필드명을 OpenTelemetry 규칙에 맞추기
- Span/Trace context 추가
- 표준 attribute 네이밍 채택

**참고 레포**:
- https://github.com/traceloop/openllmetry (OpenLLMetry - OTel 기반)

---

### 1.2 GenAI Agentic Systems Conventions (2025 신규)
**에이전트 시스템을 위한 표준**

**주요 개념**:
```yaml
# Agent 메트릭
gen_ai.agent.task.name: "web_search"
gen_ai.agent.action.type: "tool_call"
gen_ai.agent.memory.type: "short_term"

# Multi-agent 메트릭
gen_ai.team.id: "research_team"
gen_ai.artifact.type: "document"
```

**우리가 추가해야 할 것**:
- Agent task 추적
- Tool call 메트릭
- Multi-turn conversation 추적

---

## 2. 품질 평가 메트릭 📊

### 2.1 전통적인 NLP 메트릭

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
**용도**: 요약, 텍스트 생성 평가
```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference, hypothesis)
# rouge1: unigram overlap
# rouge2: bigram overlap
# rougeL: longest common subsequence
```

**레포**: https://github.com/google-research/google-research/tree/master/rouge

**한계**: 표면적 유사도만 측정, 의미적 정확성 무시

---

#### BLEU (Bilingual Evaluation Understudy)
**용도**: 번역, 텍스트 생성 평가
```python
from nltk.translate.bleu_score import sentence_bleu

reference = [['the', 'cat', 'is', 'on', 'the', 'mat']]
candidate = ['the', 'cat', 'sat', 'on', 'the', 'mat']
score = sentence_bleu(reference, candidate)
```

**한계**: 동일한 문제 - 의미보다 단어 매칭

---

#### BERTScore
**용도**: 의미적 유사도 평가 (임베딩 기반)
```python
from bert_score import score

P, R, F1 = score(candidates, references, lang="en", verbose=True)
# Precision, Recall, F1 기반
```

**레포**: https://github.com/Tiiiger/bert_score

**장점**: 의미적 유사도 측정
**한계**: 사실성(factuality)과 의미 유사성은 다름

---

### 2.2 LLM 특화 평가 메트릭

#### 2.2.1 Hallucination Detection (환각 탐지) ⭐⭐⭐
**매우 중요! 생산 환경에서 필수**

**방법론**:

1. **SelfCheckGPT**
   ```python
   # 동일한 프롬프트로 여러 번 생성 후 일관성 확인
   responses = [generate(prompt) for _ in range(5)]
   consistency_score = calculate_consistency(responses)
   ```
   - 레포: https://github.com/potsawee/selfcheckgpt

2. **Factual Consistency Metrics**
   - **FactKB**: Knowledge Base 기반 사실성 검증
   - **QuestEval**: 질문 생성 → 답변 비교
   - **SUMMAC**: NLI 모델 기반 일관성 체크

3. **LLM-as-a-Judge**
   ```python
   prompt = f"""
   Given this context: {context}
   And this response: {response}
   Is the response factually correct? Answer YES or NO and explain.
   """
   judgment = judge_llm(prompt)
   ```

**레포**:
- https://github.com/EdinburghNLP/awesome-hallucination-detection (종합 리스트)
- https://huggingface.co/blog/leaderboard-hallucinations (리더보드)

**우리가 추가해야 할 메트릭**:
```python
@dataclass
class RequestMetrics:
    # 기존 필드들...

    # 품질 메트릭 추가
    hallucination_score: Optional[float] = None  # 0.0-1.0
    factuality_score: Optional[float] = None
    relevance_score: Optional[float] = None
    coherence_score: Optional[float] = None
```

---

#### 2.2.2 RAG 전용 메트릭
**RAG 시스템 사용 시 필수**

**메트릭**:
1. **Context Relevance**: 검색된 문서가 질문과 관련 있는가?
2. **Answer Relevance**: 답변이 질문과 관련 있는가?
3. **Faithfulness/Groundedness**: 답변이 컨텍스트에 기반하는가?
4. **Context Precision**: 검색 순서가 적절한가?

**레포**:
- https://github.com/explodinggradients/ragas (RAGAS - RAG Assessment)
- https://github.com/run-llama/llama_index (LlamaIndex - built-in evaluation)

**예시**:
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
```

---

## 3. 안전성 메트릭 🛡️

### 3.1 Toxicity (독성) 감지
```python
from detoxify import Detoxify

model = Detoxify('original')
results = model.predict("your text here")
# Returns: toxicity, severe_toxicity, obscene, threat, insult, identity_attack
```

**레포**: https://github.com/unitaryai/detoxify

---

### 3.2 Bias (편향) 감지
**도구**:
- https://github.com/huggingface/evaluate (HuggingFace Evaluate)
- https://github.com/microsoft/responsible-ai-toolbox

---

### 3.3 PII (개인정보) 감지
```python
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()
results = analyzer.analyze(text="John's phone is 212-555-5555", language='en')
```

**레포**: https://github.com/microsoft/presidio

---

### 3.4 Prompt Injection 감지
**도구**:
- https://github.com/whylabs/langkit (LangKit by WhyLabs)
  - Prompt injection score
  - Jailbreak detection
  - Toxicity, sentiment, regex patterns

**우리가 추가해야 할 메트릭**:
```python
@dataclass
class RequestMetrics:
    # 안전성 메트릭
    toxicity_score: Optional[float] = None
    pii_detected: bool = False
    pii_types: List[str] = field(default_factory=list)
    prompt_injection_score: Optional[float] = None
    content_safety_flags: List[str] = field(default_factory=list)
```

---

## 4. 비즈니스 메트릭 📈

### 4.1 사용자 피드백 메트릭
```python
@dataclass
class RequestMetrics:
    # 사용자 피드백
    user_rating: Optional[int] = None  # 1-5 stars
    thumbs_up: Optional[bool] = None
    user_feedback_text: Optional[str] = None
    user_edited_output: bool = False

    # 비즈니스 메트릭
    session_id: Optional[str] = None
    conversion: bool = False  # 목표 달성 여부
    retention_7d: Optional[bool] = None
    churn_risk_score: Optional[float] = None
```

---

### 4.2 A/B 테스트 메트릭
```python
@dataclass
class RequestMetrics:
    experiment_id: Optional[str] = None
    variant: Optional[str] = None  # "control" or "treatment"

    # Winner 판단 기준
    user_preference: Optional[str] = None
    task_success: bool = False
```

---

## 5. LLMOps 플랫폼 및 도구 🛠️

### 5.1 오픈소스 도구

#### Phoenix (by Arize AI) ⭐⭐⭐
**강력 추천! 완전 오픈소스**

**레포**: https://github.com/Arize-ai/phoenix

**기능**:
- LLM traces & spans 시각화
- 내장 hallucination detection
- 임베딩 drift 모니터링
- RAG 평가
- 프롬프트 실험

**통합 예시**:
```python
import phoenix as px
from openinference.instrumentation import using_session

px.launch_app()

with using_session(session_id="my-session"):
    # Your LLM calls here
    response = llm.generate(prompt)
```

---

#### TruLens (by TruEra)
**레포**: https://github.com/truera/trulens

**특징**:
- LangChain, LlamaIndex 통합
- 자동 평가 파이프라인
- Feedback functions (hallucination, relevance, 등)

```python
from trulens_eval import TruChain, Feedback, Tru

# Define feedback functions
f_groundedness = Feedback(provider.groundedness_measure)
f_qa_relevance = Feedback(provider.relevance)

# Wrap your chain
tru_chain = TruChain(
    chain,
    app_id='my-app',
    feedbacks=[f_groundedness, f_qa_relevance]
)
```

---

#### LangFuse (오픈소스 LangSmith 대안)
**레포**: https://github.com/langfuse/langfuse

**특징**:
- Self-hosted 가능
- 비용 추적
- 프롬프트 관리
- 사용자 피드백

---

### 5.2 상용 플랫폼

#### LangSmith (by LangChain)
- **가격**: Free tier (5K traces/month), Paid ($39+/month)
- **강점**: LangChain 완벽 통합, 디버깅 UI
- **URL**: https://smith.langchain.com/

#### Weights & Biases (W&B)
- **레포**: https://github.com/wandb/wandb
- **특징**: MLOps + LLMOps 통합
- **Prompts**: https://docs.wandb.ai/guides/prompts

#### Arize (Enterprise)
- **URL**: https://arize.com/
- **특징**: ML + LLM 통합 모니터링, 임베딩 drift

#### WhyLabs
- **URL**: https://whylabs.ai/
- **특징**: LangKit 통합, compliance 중심

---

## 6. 추가로 고려해야 할 고급 개념 🚀

### 6.1 Token Attribution
**누가 비용을 발생시켰는가?**
```python
@dataclass
class RequestMetrics:
    # 시스템 vs 사용자 토큰 구분
    system_prompt_tokens: int = 0
    user_prompt_tokens: int = 0
    tool_call_tokens: int = 0

    # 멀티턴 대화에서 누적
    conversation_id: Optional[str] = None
    turn_number: int = 0
    cumulative_tokens: int = 0
```

---

### 6.2 Prompt Optimization 메트릭
```python
@dataclass
class PromptMetrics:
    prompt_template_id: str
    prompt_version: str

    # 최적화 지표
    avg_tokens_used: float
    avg_latency: float
    success_rate: float
    user_satisfaction: float

    # A/B 테스트
    is_winning_variant: bool
```

---

### 6.3 Model Drift Detection
**모델 출력이 시간에 따라 변하는가?**
```python
# 임베딩 기반 drift detection
from scipy.spatial.distance import cosine

def calculate_drift(current_embeddings, baseline_embeddings):
    return cosine(current_embeddings.mean(axis=0),
                  baseline_embeddings.mean(axis=0))
```

**도구**: Phoenix, Arize, WhyLabs

---

### 6.4 Fine-tuning 메트릭
```python
@dataclass
class FineTuningMetrics:
    base_model: str
    fine_tuned_model: str
    training_tokens: int
    training_cost: float

    # 성능 개선
    accuracy_improvement: float
    latency_improvement: float
    cost_per_request_improvement: float
```

---

## 7. 우리 시스템에 추가할 우선순위 🎯

### Priority 1 (즉시 추가)
1. **OpenTelemetry 호환성**
   - Span/Trace context
   - 표준 attribute 네이밍

2. **Hallucination Score**
   - SelfCheckGPT 또는 LLM-as-a-judge

3. **User Feedback**
   - Thumbs up/down
   - Ratings

---

### Priority 2 (다음 단계)
1. **RAG 메트릭** (RAG 사용 시)
   - Context relevance
   - Faithfulness

2. **안전성 메트릭**
   - Toxicity detection
   - PII detection

3. **A/B Testing 지원**
   - Experiment tracking
   - Variant comparison

---

### Priority 3 (고급 기능)
1. **Phoenix/TruLens 통합**
2. **Embedding drift monitoring**
3. **Custom evaluation pipelines**

---

## 8. 통합 예시 코드

```python
from vllm.llm_metrics_collector import RequestMetrics, VLLMMetricsCollector
from opentelemetry import trace
from trulens_eval import Feedback

class EnhancedVLLMCollector(VLLMMetricsCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer = trace.get_tracer(__name__)
        self.hallucination_detector = SelfCheckGPT()
        self.toxicity_detector = Detoxify()

    def collect_with_quality_metrics(
        self,
        finished_stats,
        prompt: str,
        response: str,
        context: Optional[str] = None,
        **kwargs
    ) -> RequestMetrics:
        # 1. 기본 메트릭 수집
        metrics = self.collect(finished_stats=finished_stats, **kwargs)

        # 2. OpenTelemetry Span
        with self.tracer.start_as_current_span("llm.request") as span:
            span.set_attribute("gen_ai.system", "vllm")
            span.set_attribute("gen_ai.request.model", self.model_name)
            span.set_attribute("gen_ai.usage.input_tokens", metrics.prompt_tokens)

        # 3. Hallucination detection
        if context:
            metrics.hallucination_score = self.hallucination_detector.check(
                response, context
            )

        # 4. Toxicity detection
        toxicity_results = self.toxicity_detector.predict(response)
        metrics.toxicity_score = toxicity_results['toxicity']

        # 5. RAG metrics (if applicable)
        if context:
            metrics.context_relevance = calculate_relevance(prompt, context)
            metrics.faithfulness = calculate_faithfulness(response, context)

        return metrics
```

---

## 9. 참고 리소스 총정리 📚

### 표준 & 스펙
- ⭐ OpenTelemetry GenAI: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- GenAI Metrics: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/

### 평가 프레임워크
- ⭐ RAGAS (RAG): https://github.com/explodinggradients/ragas
- TruLens: https://github.com/truera/trulens
- DeepEval: https://github.com/confident-ai/deepeval
- PromptTools: https://github.com/hegelai/prompttools

### 오픈소스 플랫폼
- ⭐ Phoenix: https://github.com/Arize-ai/phoenix
- LangFuse: https://github.com/langfuse/langfuse
- OpenLLMetry: https://github.com/traceloop/openllmetry

### Hallucination Detection
- ⭐ Awesome List: https://github.com/EdinburghNLP/awesome-hallucination-detection
- SelfCheckGPT: https://github.com/potsawee/selfcheckgpt

### 안전성
- Detoxify: https://github.com/unitaryai/detoxify
- Presidio (PII): https://github.com/microsoft/presidio
- LangKit: https://github.com/whylabs/langkit

### 평가 메트릭
- BERTScore: https://github.com/Tiiiger/bert_score
- ROUGE: https://github.com/google-research/google-research/tree/master/rouge

---

## 10. 결론 및 다음 단계

### 현재 우리 시스템의 커버리지
✅ 운영 메트릭 (토큰, 레이턴시, 비용) - **완료**
❌ 품질 메트릭 (정확성, 관련성) - **미구현**
❌ 안전성 메트릭 (독성, PII) - **미구현**
❌ 비즈니스 메트릭 (만족도, 전환율) - **미구현**
⚠️  OpenTelemetry 호환성 - **부분 구현**

### 추천 구현 순서
1. **Phase 6**: OpenTelemetry 통합 (표준 준수)
2. **Phase 7**: Hallucination detection (품질)
3. **Phase 8**: User feedback 시스템 (비즈니스)
4. **Phase 9**: Phoenix/TruLens 통합 (오픈소스 도구 활용)
5. **Phase 10**: 안전성 메트릭 (Toxicity, PII)

이 문서를 TODO.md와 함께 보시면서 다음 단계를 계획하시면 됩니다!
