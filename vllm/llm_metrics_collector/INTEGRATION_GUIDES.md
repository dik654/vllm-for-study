# LLM Metrics Collector 통합 가이드

vLLM Metrics Collector를 다양한 오픈소스 도구 및 플랫폼과 통합하는 상세 가이드입니다.

---

## 📋 목차

1. [OpenTelemetry 통합](#1-opentelemetry-통합)
2. [Phoenix (Arize AI) 통합](#2-phoenix-arize-ai-통합)
3. [LangFuse 통합](#3-langfuse-통합)
4. [TruLens 통합](#4-trulens-통합)
5. [품질 메트릭 추가](#5-품질-메트릭-추가)
6. [안전성 체크 추가](#6-안전성-체크-추가)
7. [RAG 평가 통합](#7-rag-평가-통합)
8. [데이터베이스 저장소 구현](#8-데이터베이스-저장소-구현)

---

## 1. OpenTelemetry 통합

### 1.1 개요

OpenTelemetry Semantic Conventions for GenAI를 준수하여 표준화된 메트릭을 수집합니다.

**공식 스펙**: https://opentelemetry.io/docs/specs/semconv/gen-ai/

### 1.2 설치

```bash
pip install opentelemetry-api opentelemetry-sdk
pip install opentelemetry-exporter-otlp
```

### 1.3 구현

#### 1.3.1 RequestMetrics에 OTel 속성 추가

```python
# vllm/llm_metrics_collector/models/request_metrics.py

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

@dataclass
class RequestMetrics:
    # 기존 필드들...

    # OpenTelemetry 추가
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    def to_otel_attributes(self) -> Dict[str, Any]:
        """Convert to OpenTelemetry span attributes.

        Follows: https://opentelemetry.io/docs/specs/semconv/gen-ai/
        """
        return {
            # System
            "gen_ai.system": "vllm",
            "gen_ai.request.model": self.model_name,

            # Request parameters
            "gen_ai.request.max_tokens": self.max_tokens,
            "gen_ai.request.temperature": self.temperature,
            "gen_ai.request.top_p": self.top_p,

            # Token usage
            "gen_ai.usage.input_tokens": self.prompt_tokens,
            "gen_ai.usage.output_tokens": self.completion_tokens,

            # Response
            "gen_ai.response.finish_reasons": [self.finish_reason.value],

            # Server metrics (if available)
            "gen_ai.server.time_per_output_token": self.time_per_output_token,
            "gen_ai.server.time_to_first_token": self.time_to_first_token,
        }
```

#### 1.3.2 Tracer 통합

```python
# vllm/llm_metrics_collector/collectors/otel_collector.py

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

class OpenTelemetryCollector(VLLMMetricsCollector):
    """VLLMMetricsCollector with OpenTelemetry integration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set up OpenTelemetry
        provider = TracerProvider()
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        self.tracer = trace.get_tracer(__name__)

    def collect_with_tracing(
        self,
        finished_stats,
        request_id: str,
        **kwargs
    ) -> RequestMetrics:
        """Collect metrics and create OpenTelemetry span."""

        with self.tracer.start_as_current_span("gen_ai.request") as span:
            # Collect basic metrics
            metrics = self.collect(
                finished_stats=finished_stats,
                request_id=request_id,
                **kwargs
            )

            # Add OTel attributes
            otel_attrs = metrics.to_otel_attributes()
            for key, value in otel_attrs.items():
                span.set_attribute(key, value)

            # Add trace context to metrics
            span_context = span.get_span_context()
            metrics.trace_id = format(span_context.trace_id, '032x')
            metrics.span_id = format(span_context.span_id, '016x')

            # Set span status based on finish reason
            if metrics.is_successful():
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR))

            return metrics
```

#### 1.3.3 사용 예시

```python
from vllm.llm_metrics_collector.collectors.otel_collector import OpenTelemetryCollector

collector = OpenTelemetryCollector(
    model_name="gpt-3.5-turbo",
    engine_id="engine-1"
)

# 요청 처리 시
metrics = collector.collect_with_tracing(
    finished_stats=vllm_stats,
    request_id="req-123",
    user_id="user-456"
)

# trace_id로 분산 추적 가능
print(f"Trace ID: {metrics.trace_id}")
```

### 1.4 메트릭 내보내기

```python
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# Set up metrics
reader = PeriodicExportingMetricReader(OTLPMetricExporter())
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)

meter = metrics.get_meter(__name__)

# Create instruments
token_counter = meter.create_counter(
    "gen_ai.client.token.usage",
    description="Number of tokens used",
    unit="token"
)

duration_histogram = meter.create_histogram(
    "gen_ai.client.operation.duration",
    description="Duration of LLM operations",
    unit="s"
)

# Record metrics
token_counter.add(metrics.total_tokens, {
    "gen_ai.token.type": "input",
    "gen_ai.request.model": metrics.model_name
})

duration_histogram.record(metrics.e2e_latency, {
    "gen_ai.operation.name": "completion",
    "gen_ai.request.model": metrics.model_name
})
```

---

## 2. Phoenix (Arize AI) 통합

### 2.1 개요

Phoenix는 완전 오픈소스 LLM 관찰성 플랫폼입니다.

**GitHub**: https://github.com/Arize-ai/phoenix
**문서**: https://docs.arize.com/phoenix/

### 2.2 설치

```bash
pip install arize-phoenix
```

### 2.3 기본 설정

```python
# vllm/llm_metrics_collector/integrations/phoenix_integration.py

import phoenix as px
from phoenix.trace import SpanKind
from phoenix.trace.openai import OpenAIInstrumentor

class PhoenixIntegration:
    """Phoenix integration for vLLM metrics."""

    def __init__(self, launch_app: bool = True):
        """Initialize Phoenix.

        Args:
            launch_app: Whether to launch Phoenix UI
        """
        if launch_app:
            self.session = px.launch_app()
            print(f"Phoenix UI: http://localhost:6006")

        # Auto-instrument (optional)
        # OpenAIInstrumentor().instrument()

    def log_metrics(self, metrics: RequestMetrics):
        """Log metrics to Phoenix.

        Args:
            metrics: RequestMetrics to log
        """
        # Phoenix automatically captures instrumented calls
        # For custom logging:
        px.Client().log_evaluations(
            dataframe=self._metrics_to_dataframe([metrics])
        )

    def _metrics_to_dataframe(self, metrics_list: List[RequestMetrics]):
        """Convert metrics to pandas DataFrame for Phoenix."""
        import pandas as pd

        data = []
        for m in metrics_list:
            data.append({
                'request_id': m.request_id,
                'model': m.model_name,
                'prompt_tokens': m.prompt_tokens,
                'completion_tokens': m.completion_tokens,
                'latency': m.e2e_latency,
                'ttft': m.time_to_first_token,
                'cost': m.estimated_cost,
                'finish_reason': m.finish_reason.value,
            })

        return pd.DataFrame(data)
```

### 2.4 자동 계측 (Auto-instrumentation)

```python
# vllm/llm_metrics_collector/collectors/phoenix_collector.py

import phoenix as px
from phoenix.trace.openinference import OpenInferenceSpanKind
from openinference.instrumentation import using_attributes

class PhoenixCollector(VLLMMetricsCollector):
    """Collector with Phoenix tracing."""

    def __init__(self, *args, phoenix_endpoint: str = None, **kwargs):
        super().__init__(*args, **kwargs)

        # Launch Phoenix
        if phoenix_endpoint:
            px.launch_app(host=phoenix_endpoint)
        else:
            px.launch_app()

    def collect_with_phoenix(
        self,
        finished_stats,
        prompt: str,
        response: str,
        **kwargs
    ) -> RequestMetrics:
        """Collect metrics with Phoenix tracing."""

        with using_attributes(
            session_id=kwargs.get('session_id'),
            user_id=kwargs.get('user_id'),
            metadata={
                'model': self.model_name,
                'engine': self.engine_id,
            },
            tags=['vllm', 'production'],
            prompt_template=kwargs.get('prompt_template'),
            prompt_template_version=kwargs.get('prompt_version'),
        ):
            # Collect metrics
            metrics = self.collect(
                finished_stats=finished_stats,
                **kwargs
            )

            # Phoenix automatically captures this in the context

        return metrics
```

### 2.5 환각 감지 통합

```python
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    run_evals
)

class PhoenixQualityCollector(PhoenixCollector):
    """Collector with Phoenix quality evaluations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize evaluators
        self.hallucination_eval = HallucinationEvaluator()
        self.qa_eval = QAEvaluator()

    def collect_with_quality_checks(
        self,
        finished_stats,
        prompt: str,
        response: str,
        context: Optional[str] = None,
        **kwargs
    ) -> RequestMetrics:
        """Collect with quality evaluations."""

        # Basic collection
        metrics = self.collect_with_phoenix(
            finished_stats=finished_stats,
            prompt=prompt,
            response=response,
            **kwargs
        )

        # Run evaluations
        if context:
            # Hallucination check
            hallucination_result = self.hallucination_eval.evaluate(
                input=prompt,
                output=response,
                context=context
            )
            metrics.hallucination_score = hallucination_result.score

            # QA relevance
            qa_result = self.qa_eval.evaluate(
                input=prompt,
                output=response
            )
            metrics.relevance_score = qa_result.score

        return metrics
```

### 2.6 사용 예시

```python
from vllm.llm_metrics_collector.integrations.phoenix_integration import PhoenixIntegration
from vllm.llm_metrics_collector.collectors.phoenix_collector import PhoenixQualityCollector

# 1. Phoenix 시작
phoenix = PhoenixIntegration(launch_app=True)
# UI available at http://localhost:6006

# 2. Collector 생성
collector = PhoenixQualityCollector(
    model_name="gpt-3.5-turbo",
    engine_id="engine-1"
)

# 3. 메트릭 수집
metrics = collector.collect_with_quality_checks(
    finished_stats=vllm_stats,
    prompt="What is LangChain?",
    response="LangChain is a framework...",
    context="LangChain documentation says...",
    request_id="req-123"
)

print(f"Hallucination score: {metrics.hallucination_score}")
print(f"Relevance score: {metrics.relevance_score}")

# 4. Phoenix UI에서 결과 확인
# http://localhost:6006
```

---

## 3. LangFuse 통합

### 3.1 개요

LangFuse는 오픈소스 LLM 엔지니어링 플랫폼입니다.

**GitHub**: https://github.com/langfuse/langfuse
**문서**: https://langfuse.com/docs

### 3.2 설치

```bash
pip install langfuse
```

### 3.3 Self-hosted 설정 (선택사항)

```bash
# Docker Compose
docker run -d --name langfuse \
  -p 3000:3000 \
  -e DATABASE_URL="postgresql://..." \
  langfuse/langfuse:latest
```

### 3.4 구현

```python
# vllm/llm_metrics_collector/integrations/langfuse_integration.py

from langfuse import Langfuse
from langfuse.model import CreateScore

class LangFuseIntegration:
    """LangFuse integration for vLLM metrics."""

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com"
    ):
        """Initialize LangFuse client.

        Args:
            public_key: LangFuse public key
            secret_key: LangFuse secret key
            host: LangFuse host (for self-hosted)
        """
        self.client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )

    def create_trace(self, metrics: RequestMetrics, session_id: str = None):
        """Create a trace in LangFuse.

        Args:
            metrics: RequestMetrics to log
            session_id: Optional session ID for grouping

        Returns:
            Trace ID
        """
        trace = self.client.trace(
            name=f"vllm-request-{metrics.request_id}",
            user_id=metrics.user_id,
            session_id=session_id,
            metadata={
                'model': metrics.model_name,
                'engine_id': metrics.engine_id,
            },
            tags=['vllm', 'production'],
        )

        # Add generation
        generation = trace.generation(
            name="completion",
            model=metrics.model_name,
            model_parameters={
                'max_tokens': metrics.max_tokens,
                'temperature': metrics.temperature,
                'top_p': metrics.top_p,
            },
            usage={
                'input': metrics.prompt_tokens,
                'output': metrics.completion_tokens,
                'total': metrics.total_tokens,
            },
            metadata={
                'finish_reason': metrics.finish_reason.value,
                'cached_tokens': metrics.cached_tokens,
            },
        )

        # Add scores
        if metrics.estimated_cost > 0:
            generation.score(
                name="cost",
                value=metrics.estimated_cost,
                comment=f"${metrics.estimated_cost:.4f}"
            )

        generation.score(
            name="latency",
            value=metrics.e2e_latency,
            comment=f"{metrics.e2e_latency:.3f}s"
        )

        if metrics.hallucination_score is not None:
            generation.score(
                name="hallucination",
                value=1.0 - metrics.hallucination_score,  # Higher is better
                comment="Hallucination check"
            )

        return trace.id

    def add_user_feedback(
        self,
        trace_id: str,
        rating: int,
        comment: str = None
    ):
        """Add user feedback to a trace.

        Args:
            trace_id: Trace ID
            rating: 1-5 rating
            comment: Optional feedback text
        """
        self.client.score(
            trace_id=trace_id,
            name="user_feedback",
            value=rating / 5.0,  # Normalize to 0-1
            comment=comment
        )
```

### 3.5 Collector 통합

```python
# vllm/llm_metrics_collector/collectors/langfuse_collector.py

class LangFuseCollector(VLLMMetricsCollector):
    """Collector with LangFuse integration."""

    def __init__(
        self,
        *args,
        langfuse_public_key: str,
        langfuse_secret_key: str,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.langfuse = LangFuseIntegration(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key
        )

    def collect_and_trace(
        self,
        finished_stats,
        session_id: str = None,
        **kwargs
    ) -> Tuple[RequestMetrics, str]:
        """Collect metrics and create LangFuse trace.

        Returns:
            Tuple of (metrics, trace_id)
        """
        # Collect metrics
        metrics = self.collect(
            finished_stats=finished_stats,
            **kwargs
        )

        # Create trace
        trace_id = self.langfuse.create_trace(
            metrics=metrics,
            session_id=session_id
        )

        return metrics, trace_id
```

### 3.6 사용 예시

```python
from vllm.llm_metrics_collector.collectors.langfuse_collector import LangFuseCollector

# 1. Collector 생성
collector = LangFuseCollector(
    model_name="gpt-3.5-turbo",
    engine_id="engine-1",
    langfuse_public_key="pk-...",
    langfuse_secret_key="sk-..."
)

# 2. 메트릭 수집 및 추적
metrics, trace_id = collector.collect_and_trace(
    finished_stats=vllm_stats,
    request_id="req-123",
    user_id="user-456",
    session_id="session-789"
)

print(f"Trace ID: {trace_id}")
print(f"Cost: ${metrics.estimated_cost:.4f}")

# 3. 나중에 사용자 피드백 추가
collector.langfuse.add_user_feedback(
    trace_id=trace_id,
    rating=5,
    comment="Great response!"
)

# 4. LangFuse UI에서 확인
# https://cloud.langfuse.com/ (또는 self-hosted URL)
```

---

## 4. TruLens 통합

### 4.1 개요

TruLens는 LLM 앱 평가 및 추적 프레임워크입니다.

**GitHub**: https://github.com/truera/trulens
**문서**: https://www.trulens.org/

### 4.2 설치

```bash
pip install trulens-eval
```

### 4.3 구현

```python
# vllm/llm_metrics_collector/integrations/trulens_integration.py

from trulens_eval import Tru, Feedback, Select
from trulens_eval.feedback import Groundedness, GroundTruthAgreement
from trulens_eval.feedback.provider.openai import OpenAI as OpenAIProvider
import numpy as np

class TruLensIntegration:
    """TruLens integration for vLLM metrics."""

    def __init__(self, database_url: str = "sqlite:///trulens.db"):
        """Initialize TruLens.

        Args:
            database_url: Database URL for storing results
        """
        self.tru = Tru(database_url=database_url)
        self.provider = OpenAIProvider()

        # Define feedback functions
        self.feedbacks = self._create_feedback_functions()

    def _create_feedback_functions(self) -> List[Feedback]:
        """Create feedback functions for evaluation."""

        feedbacks = []

        # Groundedness (hallucination check)
        groundedness = Groundedness(groundedness_provider=self.provider)
        f_groundedness = Feedback(
            groundedness.groundedness_measure_with_cot_reasons,
            name="Groundedness"
        ).on(
            Select.RecordCalls.retrieve.rets[:].page_content  # context
        ).on_output()
        feedbacks.append(f_groundedness)

        # Answer relevance
        f_qa_relevance = Feedback(
            self.provider.relevance,
            name="Answer Relevance"
        ).on_input().on_output()
        feedbacks.append(f_qa_relevance)

        # Context relevance
        f_context_relevance = Feedback(
            self.provider.context_relevance,
            name="Context Relevance"
        ).on_input().on(
            Select.RecordCalls.retrieve.rets[:].page_content
        ).aggregate(np.mean)
        feedbacks.append(f_context_relevance)

        return feedbacks

    def evaluate_metrics(
        self,
        metrics: RequestMetrics,
        prompt: str,
        response: str,
        context: List[str] = None
    ) -> Dict[str, float]:
        """Evaluate metrics with TruLens feedback functions.

        Args:
            metrics: RequestMetrics
            prompt: Input prompt
            response: Model response
            context: Retrieved context (for RAG)

        Returns:
            Dictionary of feedback scores
        """
        record = {
            'input': prompt,
            'output': response,
            'context': context or []
        }

        results = {}
        for feedback in self.feedbacks:
            try:
                score = feedback.run(record)
                results[feedback.name] = score
            except Exception as e:
                print(f"Error in {feedback.name}: {e}")
                results[feedback.name] = None

        return results
```

### 4.4 Collector 통합

```python
# vllm/llm_metrics_collector/collectors/trulens_collector.py

class TruLensCollector(VLLMMetricsCollector):
    """Collector with TruLens evaluation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trulens = TruLensIntegration()

    def collect_with_evaluation(
        self,
        finished_stats,
        prompt: str,
        response: str,
        context: List[str] = None,
        **kwargs
    ) -> RequestMetrics:
        """Collect metrics with TruLens evaluation.

        Args:
            finished_stats: vLLM stats
            prompt: Input prompt
            response: Model response
            context: Retrieved context for RAG
            **kwargs: Additional collector arguments

        Returns:
            RequestMetrics with evaluation scores
        """
        # Basic collection
        metrics = self.collect(
            finished_stats=finished_stats,
            **kwargs
        )

        # Run evaluations
        eval_results = self.trulens.evaluate_metrics(
            metrics=metrics,
            prompt=prompt,
            response=response,
            context=context
        )

        # Add to metrics
        metrics.groundedness_score = eval_results.get('Groundedness')
        metrics.relevance_score = eval_results.get('Answer Relevance')
        metrics.context_relevance_score = eval_results.get('Context Relevance')

        return metrics
```

### 4.5 사용 예시

```python
from vllm.llm_metrics_collector.collectors.trulens_collector import TruLensCollector

# 1. Collector 생성
collector = TruLensCollector(
    model_name="gpt-3.5-turbo",
    engine_id="engine-1"
)

# 2. RAG 요청 평가
prompt = "What is LangChain?"
context = [
    "LangChain is a framework for building LLM applications.",
    "It provides tools for prompt management, chains, and agents."
]
response = "LangChain is a framework for developing applications powered by language models."

metrics = collector.collect_with_evaluation(
    finished_stats=vllm_stats,
    prompt=prompt,
    response=response,
    context=context,
    request_id="req-123"
)

print(f"Groundedness: {metrics.groundedness_score:.2f}")
print(f"Relevance: {metrics.relevance_score:.2f}")
print(f"Context Relevance: {metrics.context_relevance_score:.2f}")

# 3. TruLens 대시보드 시작
from trulens_eval import Tru
tru = Tru()
tru.run_dashboard()
# UI at http://localhost:8501
```

---

## 5. 품질 메트릭 추가

### 5.1 Hallucination Detection (SelfCheckGPT)

```python
# vllm/llm_metrics_collector/quality/hallucination.py

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

class HallucinationDetector:
    """Hallucination detection using SelfCheckGPT."""

    def __init__(self, device: str = 'cuda'):
        self.detector = SelfCheckNLI(device=device)

    def detect(
        self,
        response: str,
        samples: List[str]
    ) -> float:
        """Detect hallucinations.

        Args:
            response: Response to check
            samples: Multiple sampled responses

        Returns:
            Hallucination score (0-1, higher = more hallucination)
        """
        scores = self.detector.predict(
            sentences=[response],
            sampled_passages=samples
        )

        return float(scores[0])
```

### 5.2 통합

```python
# vllm/llm_metrics_collector/collectors/quality_collector.py

class QualityCollector(VLLMMetricsCollector):
    """Collector with quality checks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hallucination_detector = HallucinationDetector()

    def collect_with_quality(
        self,
        finished_stats,
        response: str,
        samples: List[str] = None,
        **kwargs
    ) -> RequestMetrics:
        """Collect with quality metrics."""

        metrics = self.collect(finished_stats=finished_stats, **kwargs)

        # Hallucination detection
        if samples and len(samples) >= 3:
            metrics.hallucination_score = self.hallucination_detector.detect(
                response, samples
            )

        return metrics
```

---

## 6. 안전성 체크 추가

### 6.1 Toxicity Detection

```python
# vllm/llm_metrics_collector/safety/toxicity.py

from detoxify import Detoxify

class ToxicityDetector:
    """Toxicity detection."""

    def __init__(self, model: str = 'original'):
        self.model = Detoxify(model)

    def detect(self, text: str) -> Dict[str, float]:
        """Detect toxicity in text.

        Returns:
            Dictionary with toxicity scores
        """
        return self.model.predict(text)
```

### 6.2 PII Detection

```python
# vllm/llm_metrics_collector/safety/pii.py

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class PIIDetector:
    """PII detection and anonymization."""

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def detect(self, text: str, language: str = 'en') -> List[Dict]:
        """Detect PII in text.

        Returns:
            List of detected PII entities
        """
        results = self.analyzer.analyze(text=text, language=language)

        return [
            {
                'type': result.entity_type,
                'start': result.start,
                'end': result.end,
                'score': result.score
            }
            for result in results
        ]

    def anonymize(self, text: str, language: str = 'en') -> str:
        """Anonymize PII in text."""
        analyzer_results = self.analyzer.analyze(text=text, language=language)
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=analyzer_results)
        return anonymized.text
```

### 6.3 통합

```python
# vllm/llm_metrics_collector/collectors/safety_collector.py

class SafetyCollector(VLLMMetricsCollector):
    """Collector with safety checks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toxicity_detector = ToxicityDetector()
        self.pii_detector = PIIDetector()

    def collect_with_safety(
        self,
        finished_stats,
        prompt: str,
        response: str,
        **kwargs
    ) -> RequestMetrics:
        """Collect with safety metrics."""

        metrics = self.collect(finished_stats=finished_stats, **kwargs)

        # Toxicity check
        toxicity = self.toxicity_detector.detect(response)
        metrics.toxicity_score = toxicity['toxicity']

        # PII check
        pii_entities = self.pii_detector.detect(response)
        metrics.pii_detected = len(pii_entities) > 0
        metrics.pii_types = [e['type'] for e in pii_entities]

        return metrics
```

---

## 7. RAG 평가 통합

### 7.1 RAGAS 통합

```python
# vllm/llm_metrics_collector/rag/ragas_evaluator.py

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

class RAGASEvaluator:
    """RAGAS evaluation for RAG systems."""

    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate RAG response.

        Returns:
            Dictionary of RAGAS scores
        """
        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [contexts],
        }

        if ground_truth:
            data['ground_truth'] = [ground_truth]

        dataset = Dataset.from_dict(data)
        results = evaluate(dataset, metrics=self.metrics)

        return results
```

### 7.2 통합

```python
# vllm/llm_metrics_collector/collectors/rag_collector.py

class RAGCollector(VLLMMetricsCollector):
    """Collector for RAG systems."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ragas = RAGASEvaluator()

    def collect_with_rag_eval(
        self,
        finished_stats,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> RequestMetrics:
        """Collect with RAG evaluation."""

        metrics = self.collect(finished_stats=finished_stats, **kwargs)

        # RAGAS evaluation
        rag_scores = self.ragas.evaluate(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )

        metrics.faithfulness_score = rag_scores.get('faithfulness')
        metrics.answer_relevancy_score = rag_scores.get('answer_relevancy')
        metrics.context_precision_score = rag_scores.get('context_precision')
        metrics.context_recall_score = rag_scores.get('context_recall')

        return metrics
```

---

## 8. 데이터베이스 저장소 구현

### 8.1 PostgreSQL 백엔드

```python
# vllm/llm_metrics_collector/storage/postgres_storage.py

import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Any, Optional

class PostgreSQLStorageBackend(StorageBackend):
    """PostgreSQL storage backend."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str
    ):
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        self._create_table()

    def _create_table(self):
        """Create metrics table if not exists."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS llm_request_metrics (
                    id BIGSERIAL PRIMARY KEY,
                    request_id VARCHAR(255) UNIQUE NOT NULL,
                    user_id VARCHAR(255),
                    organization_id VARCHAR(255),
                    model_name VARCHAR(255) NOT NULL,
                    arrival_time TIMESTAMP WITH TIME ZONE NOT NULL,

                    -- Tokens
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cached_tokens INTEGER DEFAULT 0,

                    -- Timing (seconds)
                    queued_time REAL,
                    prefill_time REAL,
                    decode_time REAL,
                    inference_time REAL,
                    e2e_latency REAL,
                    time_to_first_token REAL,
                    time_per_output_token REAL,

                    -- Resources
                    kv_cache_blocks_used INTEGER,
                    gpu_compute_time REAL,

                    -- Cost
                    estimated_cost REAL,
                    input_cost REAL,
                    output_cost REAL,

                    -- Quality
                    hallucination_score REAL,
                    relevance_score REAL,
                    toxicity_score REAL,

                    -- Metadata
                    finish_reason VARCHAR(50),
                    error_code VARCHAR(50),

                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_user_id ON llm_request_metrics(user_id);
                CREATE INDEX IF NOT EXISTS idx_arrival_time ON llm_request_metrics(arrival_time);
                CREATE INDEX IF NOT EXISTS idx_model_name ON llm_request_metrics(model_name);
            """)
            self.conn.commit()

    def save(self, metrics: RequestMetrics) -> None:
        """Save single metrics."""
        self.save_batch([metrics])

    def save_batch(self, metrics_list: List[RequestMetrics]) -> None:
        """Batch save metrics."""
        with self.conn.cursor() as cur:
            values = [
                (
                    m.request_id,
                    m.user_id,
                    m.organization_id,
                    m.model_name,
                    datetime.fromtimestamp(m.arrival_time),
                    m.prompt_tokens,
                    m.completion_tokens,
                    m.total_tokens,
                    m.cached_tokens,
                    m.queued_time,
                    m.prefill_time,
                    m.decode_time,
                    m.inference_time,
                    m.e2e_latency,
                    m.time_to_first_token,
                    m.time_per_output_token,
                    m.kv_cache_blocks_used,
                    m.gpu_compute_time,
                    m.estimated_cost,
                    m.input_cost,
                    m.output_cost,
                    m.hallucination_score,
                    m.relevance_score,
                    m.toxicity_score,
                    m.finish_reason.value,
                    m.error_code,
                )
                for m in metrics_list
            ]

            execute_values(
                cur,
                """
                INSERT INTO llm_request_metrics (
                    request_id, user_id, organization_id, model_name, arrival_time,
                    prompt_tokens, completion_tokens, total_tokens, cached_tokens,
                    queued_time, prefill_time, decode_time, inference_time, e2e_latency,
                    time_to_first_token, time_per_output_token,
                    kv_cache_blocks_used, gpu_compute_time,
                    estimated_cost, input_cost, output_cost,
                    hallucination_score, relevance_score, toxicity_score,
                    finish_reason, error_code
                ) VALUES %s
                ON CONFLICT (request_id) DO NOTHING
                """,
                values
            )
            self.conn.commit()

    def load(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[RequestMetrics]:
        """Load metrics with filters."""
        # Implementation similar to MemoryStorageBackend
        # but using SQL WHERE clauses
        pass

    def close(self):
        """Close database connection."""
        self.conn.close()
```

---

## 9. 종합 예시

모든 통합을 결합한 프로덕션 예시:

```python
# production_collector.py

from vllm.llm_metrics_collector import (
    FileStorageBackend,
    TokenBasedPricing,
)
from vllm.llm_metrics_collector.collectors.otel_collector import OpenTelemetryCollector
from vllm.llm_metrics_collector.integrations.phoenix_integration import PhoenixIntegration
from vllm.llm_metrics_collector.safety.toxicity import ToxicityDetector
from vllm.llm_metrics_collector.safety.pii import PIIDetector

class ProductionCollector:
    """Production-ready collector with all integrations."""

    def __init__(self):
        # Storage
        self.storage = FileStorageBackend(
            base_dir="/var/log/vllm/metrics",
            partition_by="day",
            compress=True
        )

        # Pricing
        self.pricing = TokenBasedPricing(
            input_price_per_1k=0.0005,
            output_price_per_1k=0.0015,
            cache_discount_per_1k=0.00025
        )

        # Collector with OpenTelemetry
        self.collector = OpenTelemetryCollector(
            model_name="gpt-3.5-turbo",
            engine_id="prod-engine-1"
        )

        # Phoenix for observability
        self.phoenix = PhoenixIntegration(launch_app=True)

        # Safety checks
        self.toxicity = ToxicityDetector()
        self.pii = PIIDetector()

    def process_request(
        self,
        finished_stats,
        request_id: str,
        user_id: str,
        prompt: str,
        response: str
    ):
        """Process a completed request with full metrics."""

        # 1. Collect基本 metrics with OpenTelemetry
        metrics = self.collector.collect_with_tracing(
            finished_stats=finished_stats,
            request_id=request_id,
            user_id=user_id
        )

        # 2. Add pricing
        metrics = self.collector.enrich(metrics, pricing_calculator=self.pricing)

        # 3. Safety checks
        toxicity = self.toxicity.detect(response)
        metrics.toxicity_score = toxicity['toxicity']

        pii_entities = self.pii.detect(response)
        metrics.pii_detected = len(pii_entities) > 0
        metrics.pii_types = [e['type'] for e in pii_entities]

        # 4. Save to file storage
        self.storage.save(metrics)

        # 5. Log to Phoenix
        self.phoenix.log_metrics(metrics)

        return metrics

# Usage
collector = ProductionCollector()

# On request completion
metrics = collector.process_request(
    finished_stats=vllm_stats,
    request_id="req-123",
    user_id="user-456",
    prompt="What is AI?",
    response="AI stands for Artificial Intelligence..."
)

print(f"Request processed: {metrics.request_id}")
print(f"Cost: ${metrics.estimated_cost:.4f}")
print(f"Toxicity: {metrics.toxicity_score:.3f}")
print(f"PII detected: {metrics.pii_detected}")
```

---

## 📝 다음 단계

각 통합은 독립적으로 구현 가능합니다. 우선순위는:

1. **OpenTelemetry** - 표준 준수
2. **Phoenix** - 무료 관찰성
3. **품질/안전성 메트릭** - 프로덕션 필수
4. **RAG 평가** - RAG 사용 시
5. **PostgreSQL** - 대규모 배포

자세한 내용은 각 도구의 공식 문서를 참조하세요!
