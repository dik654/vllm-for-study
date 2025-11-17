# Workflow Examples

이 디렉토리에는 분산 LLM 플랫폼의 Workflow UI 시스템을 사용한 예시 워크플로우가 포함되어 있습니다.

## 예시 워크플로우 목록

### 1. Simple Chat (`01-simple-chat.json`)
**카테고리**: Chatbot
**난이도**: ⭐ Beginner
**예상 비용**: ~$0.001 per execution

**설명**:
가장 기본적인 워크플로우로, 사용자 입력을 받아 LLM으로 응답을 생성합니다.

**노드 구성**:
- Input → LLM Inference → Output

**사용 예시**:
```bash
curl -X POST https://api.distributed-llm.com/v1/workflows/wf_simple_chat_001/execute \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "data": "안녕하세요, 오늘 날씨가 어떤가요?"
    }
  }'
```

---

### 2. Multimodal Content Pipeline (`02-multimodal-pipeline.json`)
**카테고리**: Content Creation
**난이도**: ⭐⭐⭐ Advanced
**예상 비용**: ~$0.15 per execution

**설명**:
텍스트 프롬프트에서 시작하여 이미지, 비디오, 음성까지 생성하는 완전한 멀티모달 콘텐츠 파이프라인입니다.

**노드 구성**:
1. Input (Story Prompt)
2. LLM (Detailed Description)
3. Image Generation
4. Video Generation (from Image)
5. Audio/Speech Generation (Narration)
6. Storage (Save Results)
7. Output

**생성되는 콘텐츠**:
- ✅ 상세한 텍스트 설명 (~300 words)
- ✅ 고해상도 이미지 (1024×1024)
- ✅ 4초 비디오 (24 FPS)
- ✅ 음성 나레이션

**병렬 실행**: ✅ Enabled (Image + Audio 동시 생성)

---

### 3. AI Agent Debate (`03-ai-debate.json`)
**카테고리**: Research
**난이도**: ⭐⭐⭐ Advanced
**예상 비용**: ~$0.025 per execution

**설명**:
여러 AI 에이전트(낙관론자, 비관론자, 실용주의자)가 특정 주제에 대해 토론하고, LLM이 최종 합의를 도출합니다.

**노드 구성**:
1. Input (Question)
2. Memory Allocation (Debate Context)
3. Agent Communication (3 agents: optimist, pessimist, pragmatist)
4. LLM (Summarize Consensus)
5. Conditional (Quality Check)
6. Output (Success or Retry)

**특징**:
- **Multi-agent protocol**: Debate 모드
- **Context Memory**: 4GB, 2시간 TTL
- **Quality Control**: 합의 결과의 품질을 확인하여 재시도 가능

**Fork 수**: 87
**실행 횟수**: 234

---

### 4. Blog Post Generator (`04-blog-post-generator.json`)
**카테고리**: Content Creation
**난이도**: ⭐⭐ Intermediate
**예상 비용**: ~$0.08 per execution

**설명**:
주제를 입력받아 블로그 개요를 생성하고, 전체 글을 작성한 뒤, 관련 이미지 3개를 자동으로 생성합니다.

**노드 구성**:
1. Input (Blog Topic)
2. LLM (Generate Outline)
3. LLM (Write Full Article)
4. LLM (Generate Image Prompts)
5. Loop (3 iterations)
6. Image Generation (inside Loop)
7. Storage (Save Images)
8. Output (Article + Image URLs)

**생성되는 콘텐츠**:
- ✅ 블로그 개요 (5 sections)
- ✅ 전체 글 (1000-1500 words)
- ✅ 관련 이미지 3개 (1024×768 each)

**Fork 수**: 342
**실행 횟수**: 1,289

---

### 5. Voice Assistant (`05-voice-assistant.json`)
**카테고리**: Voice Assistant
**난이도**: ⭐⭐⭐ Advanced
**예상 비용**: ~$0.012 per execution

**설명**:
음성 입력을 받아 처리하고 음성으로 응답하는 완전한 음성 어시스턴트 워크플로우입니다.

**노드 구성**:
1. Input (Voice File)
2. Memory (Conversation Context)
3. LLM (Understand Request)
4. Conditional (Intent Type)
5. LLM (Answer) OR LLM (Casual Chat)
6. Audio/Speech (Text-to-Speech)
7. Storage (Save Response Audio)
8. Output (Voice Response URL)

**특징**:
- **Intent Classification**: Question vs Conversation
- **Context Memory**: 1GB, 12시간 TTL
- **Multi-path Execution**: 조건부 분기로 다른 응답 생성

**Fork 수**: 98
**실행 횟수**: 467

---

## 워크플로우 사용 방법

### 1. REST API로 실행

```bash
# 워크플로우 생성
curl -X POST https://api.distributed-llm.com/v1/workflows \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @01-simple-chat.json

# 워크플로우 실행
curl -X POST https://api.distributed-llm.com/v1/workflows/{workflow_id}/execute \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "data": "Your input here"
    }
  }'

# 실행 상태 확인
curl -X GET https://api.distributed-llm.com/v1/executions/{execution_id} \
  -H "Authorization: Bearer YOUR_API_KEY"

# 실행 결과 조회
curl -X GET https://api.distributed-llm.com/v1/executions/{execution_id}/result \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 2. JavaScript SDK로 실행

```javascript
import { WorkflowClient, Workflow } from '@distributed-llm/workflow-sdk';

const client = new WorkflowClient({ apiKey: 'YOUR_API_KEY' });

// JSON 파일에서 워크플로우 로드
const workflowJson = require('./01-simple-chat.json');
const workflow = Workflow.fromJSON(workflowJson);

// 워크플로우 실행
const execution = await client.executeWorkflow(workflow, {
  input: { data: '안녕하세요' }
});

// 결과 대기
const result = await execution.waitForCompletion();
console.log(result.output);

// 또는 스트리밍 실행
await client.executeWorkflowStream(workflow, {
  input: { data: '안녕하세요' },
  onProgress: (nodeId, output) => {
    console.log(`Node ${nodeId} completed:`, output);
  },
  onComplete: (result) => {
    console.log('Final result:', result);
  }
});
```

### 3. Python SDK로 실행

```python
from distributed_llm.workflow import WorkflowClient, Workflow
import json

client = WorkflowClient(api_key='YOUR_API_KEY')

# JSON 파일에서 워크플로우 로드
with open('01-simple-chat.json') as f:
    workflow_dict = json.load(f)

workflow = Workflow.from_dict(workflow_dict)

# 워크플로우 실행
execution = client.execute_workflow(workflow, input={'data': '안녕하세요'})

# 결과 대기
result = execution.wait_for_completion()
print(result.output)

# 또는 비동기 스트리밍
async for event in client.execute_workflow_stream(workflow, input={'data': '안녕하세요'}):
    if event.type == 'node_complete':
        print(f"Node {event.node_id}: {event.output}")
    elif event.type == 'complete':
        print(f"Final result: {event.output}")
```

---

## 워크플로우 수정 및 커스터마이징

### 노드 파라미터 변경

각 워크플로우 JSON 파일에서 노드의 `parameters` 필드를 수정하여 동작을 커스터마이징할 수 있습니다.

**예시: LLM 모델 변경**

```json
{
  "id": "node_llm_1",
  "type": "llm.inference",
  "parameters": {
    "model": "llama-3-8b",        // llama-3-70b에서 변경
    "temperature": 0.5,           // 0.7에서 변경 (더 일관성 있는 응답)
    "max_tokens": 200,            // 500에서 변경 (더 짧은 응답)
    "enable_cache": true
  }
}
```

### 새 노드 추가

워크플로우에 새 노드를 추가하려면:

1. `nodes` 배열에 새 노드 추가
2. `connections` 배열에 연결 추가

**예시: 번역 노드 추가**

```json
{
  "nodes": [
    // ... 기존 노드들 ...
    {
      "id": "node_translate",
      "type": "llm.inference",
      "name": "Translate to Korean",
      "position": { "x": 950, "y": 200 },
      "parameters": {
        "model": "llama-3-70b",
        "system_prompt": "Translate the following text to Korean. Only output the translation.",
        "temperature": 0.3,
        "max_tokens": 500
      },
      "enabled": true
    }
  ],
  "connections": [
    // ... 기존 연결들 ...
    {
      "id": "conn_new",
      "source_node_id": "node_llm_1",
      "source_output": "response",
      "target_node_id": "node_translate",
      "target_input": "prompt",
      "enabled": true
    }
  ]
}
```

---

## 비용 최적화 팁

### 1. 캐싱 활성화
LLM 노드에서 `enable_cache: true`를 설정하면 동일한 프롬프트에 대해 90% 할인:

```json
{
  "parameters": {
    "enable_cache": true  // 캐시 활성화
  }
}
```

### 2. 작은 모델 사용
간단한 작업에는 작은 모델 사용:
- `llama-3-8b` (빠르고 저렴) vs `llama-3-70b` (강력하지만 비쌈)

### 3. 병렬 실행
독립적인 노드들을 병렬로 실행하여 시간 절약:

```json
{
  "settings": {
    "parallel_execution": true
  }
}
```

### 4. 토큰 제한
불필요하게 긴 응답을 방지:

```json
{
  "parameters": {
    "max_tokens": 300  // 필요한 만큼만 설정
  }
}
```

---

## 문제 해결

### 워크플로우 실행 실패

1. **Validation Error**: 노드 연결이 올바른지 확인
   - 모든 입력이 연결되어 있는지 확인
   - 순환 참조(cycle) 없는지 확인

2. **Node Execution Error**: 노드 파라미터 확인
   - Required 파라미터가 모두 설정되어 있는지
   - 파라미터 값이 허용 범위 내인지

3. **Timeout**: 실행 시간 초과
   - `settings.timeout_seconds` 증가
   - 또는 워크플로우 단순화

### 비용 초과

1. 실행 전 비용 추정 API 사용:
```bash
curl -X POST https://api.distributed-llm.com/v1/workflows/{id}/estimate \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{ "input": {...} }'
```

2. UI에서 "Estimated Cost" 확인

---

## 더 많은 예시

더 많은 워크플로우 예시는 다음에서 확인할 수 있습니다:

- [Workflow Templates Marketplace](https://distributed-llm.com/workflows/templates)
- [Community Workflows](https://github.com/distributed-llm-platform/community-workflows)

---

## 기여하기

자신만의 워크플로우를 공유하고 싶으신가요?

1. 워크플로우를 Public으로 설정
2. Fork 및 Star 기능 활성화
3. 자세한 설명과 태그 추가
4. [Pull Request](https://github.com/distributed-llm-platform/examples) 제출

---

## 라이선스

MIT License

## 문의

- 문서: [WORKFLOW_SPECIFICATION.md](../../WORKFLOW_SPECIFICATION.md)
- API 문서: [SPECIFICATION.md](../../SPECIFICATION.md)
- 이슈: [GitHub Issues](https://github.com/distributed-llm-platform/issues)
