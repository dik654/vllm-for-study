# Workflow UI Specification - AI 노드 기반 워크플로우 시스템

## 1. 개요

### 1.1 목적
사용자가 n8n처럼 노드를 연결하여 AI 서비스 워크플로우를 시각적으로 구성하고, 생성된 워크플로우를 JSON으로 변환하여 SDK를 통해 프로그래밍 방식으로 사용할 수 있는 시스템.

### 1.2 핵심 기능
- **Visual Workflow Editor**: 드래그 앤 드롭 방식의 노드 기반 UI
- **다양한 AI 서비스 노드**: LLM, Image/Video/Audio Generation, Agent-to-Agent
- **하드웨어 리소스 노드**: Memory, Storage allocation
- **Workflow JSON Export**: 워크플로우를 JSON으로 변환
- **Multi-language SDK**: JavaScript, Java, Python SDK 제공
- **Workflow Execution**: 워크플로우 실행 및 모니터링

### 1.3 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     Workflow Frontend                        │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │  Node Palette  │  │  Canvas Editor │  │  Properties    │ │
│  │  (노드 목록)    │  │  (연결 편집기)   │  │  (노드 설정)     │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │  JSON Export   │  │  Execution     │  │  Version Ctrl  │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ REST API
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Router Backend (FastAPI)                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │ Workflow API   │  │ Execution Eng  │  │  Node Registry │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │ LLM     │          │ Image   │          │ Agent   │
   │ Service │          │ Service │          │ Service │
   └─────────┘          └─────────┘          └─────────┘
```

---

## 2. 노드 타입 정의

### 2.1 노드 카테고리

#### 2.1.1 AI Service Nodes (AI 서비스 노드)

**1. LLM Inference Node (LLM 추론)**
```json
{
  "type": "llm.inference",
  "category": "ai_service",
  "name": "LLM Chat Completion",
  "description": "LLM을 사용한 텍스트 생성",
  "icon": "brain",
  "color": "#4F46E5",
  "inputs": [
    {
      "name": "prompt",
      "type": "string",
      "required": true,
      "description": "사용자 프롬프트"
    },
    {
      "name": "system_prompt",
      "type": "string",
      "required": false,
      "description": "시스템 프롬프트"
    },
    {
      "name": "context",
      "type": "array",
      "required": false,
      "description": "이전 대화 컨텍스트"
    }
  ],
  "outputs": [
    {
      "name": "response",
      "type": "string",
      "description": "LLM 응답"
    },
    {
      "name": "tokens_used",
      "type": "object",
      "description": "토큰 사용량"
    }
  ],
  "parameters": {
    "model": {
      "type": "select",
      "options": ["llama-3-70b", "llama-3-8b", "gpt-4", "claude-3"],
      "default": "llama-3-70b",
      "required": true
    },
    "temperature": {
      "type": "number",
      "min": 0.0,
      "max": 2.0,
      "default": 0.7,
      "description": "생성 다양성"
    },
    "max_tokens": {
      "type": "number",
      "min": 1,
      "max": 4096,
      "default": 1000,
      "description": "최대 생성 토큰 수"
    },
    "enable_cache": {
      "type": "boolean",
      "default": true,
      "description": "프롬프트 캐싱 활성화"
    }
  }
}
```

**2. Image Generation Node (이미지 생성)**
```json
{
  "type": "image.generation",
  "category": "ai_service",
  "name": "Image Generation",
  "description": "텍스트에서 이미지 생성 (Stable Diffusion, DALL-E)",
  "icon": "image",
  "color": "#EC4899",
  "inputs": [
    {
      "name": "prompt",
      "type": "string",
      "required": true,
      "description": "이미지 생성 프롬프트"
    },
    {
      "name": "negative_prompt",
      "type": "string",
      "required": false,
      "description": "부정적 프롬프트"
    }
  ],
  "outputs": [
    {
      "name": "image_url",
      "type": "string",
      "description": "생성된 이미지 URL"
    },
    {
      "name": "image_data",
      "type": "binary",
      "description": "이미지 바이너리 데이터"
    }
  ],
  "parameters": {
    "model": {
      "type": "select",
      "options": ["stable-diffusion-xl", "stable-diffusion-v1-5", "dall-e-3"],
      "default": "stable-diffusion-xl",
      "required": true
    },
    "width": {
      "type": "number",
      "options": [512, 768, 1024],
      "default": 1024
    },
    "height": {
      "type": "number",
      "options": [512, 768, 1024],
      "default": 1024
    },
    "steps": {
      "type": "number",
      "min": 1,
      "max": 150,
      "default": 50,
      "description": "생성 스텝 수"
    },
    "guidance_scale": {
      "type": "number",
      "min": 1.0,
      "max": 20.0,
      "default": 7.5,
      "description": "프롬프트 가이던스 강도"
    }
  }
}
```

**3. Video Generation Node (동영상 생성)**
```json
{
  "type": "video.generation",
  "category": "ai_service",
  "name": "Video Generation",
  "description": "텍스트 또는 이미지에서 동영상 생성",
  "icon": "video",
  "color": "#F59E0B",
  "inputs": [
    {
      "name": "prompt",
      "type": "string",
      "required": false,
      "description": "비디오 생성 프롬프트"
    },
    {
      "name": "init_image",
      "type": "binary",
      "required": false,
      "description": "초기 이미지 (Image-to-Video)"
    }
  ],
  "outputs": [
    {
      "name": "video_url",
      "type": "string",
      "description": "생성된 비디오 URL"
    },
    {
      "name": "video_data",
      "type": "binary",
      "description": "비디오 바이너리 데이터"
    },
    {
      "name": "duration",
      "type": "number",
      "description": "비디오 길이 (초)"
    }
  ],
  "parameters": {
    "model": {
      "type": "select",
      "options": ["runway-gen2", "stable-video-diffusion", "pika-1.0"],
      "default": "stable-video-diffusion",
      "required": true
    },
    "duration": {
      "type": "number",
      "min": 2,
      "max": 30,
      "default": 4,
      "description": "생성 길이 (초)"
    },
    "fps": {
      "type": "number",
      "options": [24, 30, 60],
      "default": 24,
      "description": "프레임 레이트"
    },
    "motion_scale": {
      "type": "number",
      "min": 0.0,
      "max": 2.0,
      "default": 1.0,
      "description": "모션 강도"
    }
  }
}
```

**4. Audio/Speech Generation Node (음성 생성)**
```json
{
  "type": "audio.generation",
  "category": "ai_service",
  "name": "Speech Synthesis",
  "description": "텍스트를 음성으로 변환 (TTS)",
  "icon": "microphone",
  "color": "#10B981",
  "inputs": [
    {
      "name": "text",
      "type": "string",
      "required": true,
      "description": "변환할 텍스트"
    },
    {
      "name": "voice_sample",
      "type": "binary",
      "required": false,
      "description": "음성 클론용 샘플"
    }
  ],
  "outputs": [
    {
      "name": "audio_url",
      "type": "string",
      "description": "생성된 오디오 URL"
    },
    {
      "name": "audio_data",
      "type": "binary",
      "description": "오디오 바이너리 데이터"
    },
    {
      "name": "duration",
      "type": "number",
      "description": "오디오 길이 (초)"
    }
  ],
  "parameters": {
    "model": {
      "type": "select",
      "options": ["elevenlabs", "openai-tts", "bark", "xtts-v2"],
      "default": "xtts-v2",
      "required": true
    },
    "voice": {
      "type": "select",
      "options": ["male-1", "female-1", "child-1", "custom"],
      "default": "female-1"
    },
    "language": {
      "type": "select",
      "options": ["en", "ko", "ja", "zh", "es", "fr"],
      "default": "en"
    },
    "speed": {
      "type": "number",
      "min": 0.5,
      "max": 2.0,
      "default": 1.0,
      "description": "재생 속도"
    }
  }
}
```

**5. Agent-to-Agent Communication Node**
```json
{
  "type": "agent.communication",
  "category": "ai_service",
  "name": "Agent Communication",
  "description": "여러 AI 에이전트 간 통신",
  "icon": "users",
  "color": "#8B5CF6",
  "inputs": [
    {
      "name": "message",
      "type": "string",
      "required": true,
      "description": "다른 에이전트에게 전달할 메시지"
    },
    {
      "name": "agent_context",
      "type": "object",
      "required": false,
      "description": "에이전트 컨텍스트"
    }
  ],
  "outputs": [
    {
      "name": "responses",
      "type": "array",
      "description": "모든 에이전트의 응답"
    },
    {
      "name": "consensus",
      "type": "string",
      "description": "에이전트들의 합의 결과"
    }
  ],
  "parameters": {
    "agent_ids": {
      "type": "array",
      "items": "string",
      "required": true,
      "description": "통신할 에이전트 ID 목록"
    },
    "protocol": {
      "type": "select",
      "options": ["sequential", "parallel", "voting", "debate"],
      "default": "sequential",
      "description": "통신 프로토콜"
    },
    "timeout": {
      "type": "number",
      "default": 30,
      "description": "타임아웃 (초)"
    }
  }
}
```

#### 2.1.2 Resource Nodes (하드웨어 리소스 노드)

**6. Memory Allocation Node (메모리 할당)**
```json
{
  "type": "resource.memory",
  "category": "resource",
  "name": "Allocate Memory",
  "description": "메모리 리소스 할당",
  "icon": "cpu",
  "color": "#6366F1",
  "inputs": [
    {
      "name": "trigger",
      "type": "any",
      "required": false,
      "description": "할당 트리거"
    }
  ],
  "outputs": [
    {
      "name": "memory_id",
      "type": "string",
      "description": "할당된 메모리 ID"
    },
    {
      "name": "status",
      "type": "object",
      "description": "할당 상태"
    }
  ],
  "parameters": {
    "memory_type": {
      "type": "select",
      "options": ["context_memory", "gp_memory"],
      "default": "context_memory",
      "required": true,
      "description": "메모리 타입"
    },
    "size_gb": {
      "type": "number",
      "min": 0.5,
      "max": 128,
      "default": 2.0,
      "description": "할당 크기 (GB)"
    },
    "ttl_hours": {
      "type": "number",
      "min": 1,
      "max": 720,
      "default": 24,
      "description": "유지 시간 (시간)"
    }
  }
}
```

**7. Storage Allocation Node (스토리지 할당)**
```json
{
  "type": "resource.storage",
  "category": "resource",
  "name": "Allocate Storage",
  "description": "스토리지 리소스 할당",
  "icon": "database",
  "color": "#14B8A6",
  "inputs": [
    {
      "name": "data",
      "type": "any",
      "required": false,
      "description": "저장할 데이터"
    }
  ],
  "outputs": [
    {
      "name": "storage_id",
      "type": "string",
      "description": "할당된 스토리지 ID"
    },
    {
      "name": "storage_url",
      "type": "string",
      "description": "스토리지 접근 URL"
    }
  ],
  "parameters": {
    "storage_type": {
      "type": "select",
      "options": ["context_storage", "gp_storage"],
      "default": "gp_storage",
      "required": true,
      "description": "스토리지 타입"
    },
    "tier": {
      "type": "select",
      "options": ["hot", "cold"],
      "default": "hot",
      "description": "스토리지 티어"
    },
    "size_gb": {
      "type": "number",
      "min": 1,
      "max": 1000,
      "default": 10,
      "description": "할당 크기 (GB)"
    }
  }
}
```

#### 2.1.3 Utility Nodes (유틸리티 노드)

**8. Input Node (입력)**
```json
{
  "type": "utility.input",
  "category": "utility",
  "name": "Workflow Input",
  "description": "워크플로우 시작 입력",
  "icon": "arrow-down-circle",
  "color": "#64748B",
  "inputs": [],
  "outputs": [
    {
      "name": "data",
      "type": "any",
      "description": "입력 데이터"
    }
  ],
  "parameters": {
    "input_type": {
      "type": "select",
      "options": ["text", "json", "file", "webhook"],
      "default": "text"
    },
    "default_value": {
      "type": "any",
      "required": false,
      "description": "기본값"
    }
  }
}
```

**9. Output Node (출력)**
```json
{
  "type": "utility.output",
  "category": "utility",
  "name": "Workflow Output",
  "description": "워크플로우 결과 출력",
  "icon": "arrow-up-circle",
  "color": "#64748B",
  "inputs": [
    {
      "name": "data",
      "type": "any",
      "required": true,
      "description": "출력할 데이터"
    }
  ],
  "outputs": [],
  "parameters": {
    "output_format": {
      "type": "select",
      "options": ["json", "text", "file", "webhook"],
      "default": "json"
    },
    "destination": {
      "type": "string",
      "required": false,
      "description": "출력 대상"
    }
  }
}
```

**10. Conditional Node (조건 분기)**
```json
{
  "type": "utility.condition",
  "category": "utility",
  "name": "If-Else",
  "description": "조건에 따라 다른 경로 실행",
  "icon": "git-branch",
  "color": "#64748B",
  "inputs": [
    {
      "name": "value",
      "type": "any",
      "required": true,
      "description": "비교할 값"
    }
  ],
  "outputs": [
    {
      "name": "true",
      "type": "any",
      "description": "조건이 참일 때"
    },
    {
      "name": "false",
      "type": "any",
      "description": "조건이 거짓일 때"
    }
  ],
  "parameters": {
    "condition": {
      "type": "string",
      "required": true,
      "description": "조건 표현식 (예: value > 100)"
    }
  }
}
```

**11. Loop Node (반복)**
```json
{
  "type": "utility.loop",
  "category": "utility",
  "name": "Loop",
  "description": "배열의 각 항목에 대해 반복 실행",
  "icon": "refresh-cw",
  "color": "#64748B",
  "inputs": [
    {
      "name": "items",
      "type": "array",
      "required": true,
      "description": "반복할 배열"
    }
  ],
  "outputs": [
    {
      "name": "item",
      "type": "any",
      "description": "현재 항목"
    },
    {
      "name": "results",
      "type": "array",
      "description": "모든 반복 결과"
    }
  ],
  "parameters": {
    "max_iterations": {
      "type": "number",
      "default": 100,
      "description": "최대 반복 횟수"
    }
  }
}
```

---

## 3. Workflow JSON Schema

### 3.1 Workflow Structure

```typescript
interface Workflow {
  id: string;                      // 워크플로우 ID
  name: string;                    // 워크플로우 이름
  description?: string;            // 설명
  version: string;                 // 버전 (semantic versioning)
  created_at: string;              // 생성 시간 (ISO 8601)
  updated_at: string;              // 수정 시간 (ISO 8601)
  user_id: string;                 // 소유자 ID

  nodes: WorkflowNode[];           // 노드 목록
  connections: Connection[];       // 연결 목록

  metadata: WorkflowMetadata;      // 메타데이터
  settings: WorkflowSettings;      // 워크플로우 설정
}

interface WorkflowNode {
  id: string;                      // 노드 인스턴스 ID (uuid)
  type: string;                    // 노드 타입 (예: "llm.inference")
  name: string;                    // 노드 이름 (사용자 정의)
  position: {                      // 캔버스 상 위치
    x: number;
    y: number;
  };
  parameters: Record<string, any>; // 노드 파라미터
  enabled: boolean;                // 활성화 여부
  notes?: string;                  // 노트
}

interface Connection {
  id: string;                      // 연결 ID
  source_node_id: string;          // 출발 노드 ID
  source_output: string;           // 출발 출력 이름
  target_node_id: string;          // 도착 노드 ID
  target_input: string;            // 도착 입력 이름
  enabled: boolean;                // 활성화 여부
}

interface WorkflowMetadata {
  tags: string[];                  // 태그
  category: string;                // 카테고리
  is_public: boolean;              // 공개 여부
  fork_count: number;              // 포크 수
  execution_count: number;         // 실행 횟수
}

interface WorkflowSettings {
  timeout_seconds: number;         // 최대 실행 시간
  max_retries: number;             // 최대 재시도 횟수
  error_handling: "stop" | "continue" | "rollback";  // 에러 처리 방식
  parallel_execution: boolean;     // 병렬 실행 가능 여부
  webhook_url?: string;            // 완료 시 웹훅 URL
}
```

### 3.2 예시 Workflow JSON

**예시 1: 간단한 LLM 요청**
```json
{
  "id": "wf_001",
  "name": "Simple Chat",
  "description": "사용자 입력을 받아 LLM으로 응답 생성",
  "version": "1.0.0",
  "created_at": "2025-11-17T10:00:00Z",
  "updated_at": "2025-11-17T10:00:00Z",
  "user_id": "user_123",

  "nodes": [
    {
      "id": "node_input_1",
      "type": "utility.input",
      "name": "User Input",
      "position": { "x": 100, "y": 200 },
      "parameters": {
        "input_type": "text",
        "default_value": "안녕하세요"
      },
      "enabled": true
    },
    {
      "id": "node_llm_1",
      "type": "llm.inference",
      "name": "LLM Chat",
      "position": { "x": 400, "y": 200 },
      "parameters": {
        "model": "llama-3-70b",
        "temperature": 0.7,
        "max_tokens": 500,
        "enable_cache": true
      },
      "enabled": true
    },
    {
      "id": "node_output_1",
      "type": "utility.output",
      "name": "Response",
      "position": { "x": 700, "y": 200 },
      "parameters": {
        "output_format": "json"
      },
      "enabled": true
    }
  ],

  "connections": [
    {
      "id": "conn_1",
      "source_node_id": "node_input_1",
      "source_output": "data",
      "target_node_id": "node_llm_1",
      "target_input": "prompt",
      "enabled": true
    },
    {
      "id": "conn_2",
      "source_node_id": "node_llm_1",
      "source_output": "response",
      "target_node_id": "node_output_1",
      "target_input": "data",
      "enabled": true
    }
  ],

  "metadata": {
    "tags": ["llm", "chat", "simple"],
    "category": "chatbot",
    "is_public": false,
    "fork_count": 0,
    "execution_count": 0
  },

  "settings": {
    "timeout_seconds": 300,
    "max_retries": 3,
    "error_handling": "stop",
    "parallel_execution": false
  }
}
```

**예시 2: 멀티모달 콘텐츠 생성 파이프라인**
```json
{
  "id": "wf_002",
  "name": "Multimodal Content Pipeline",
  "description": "텍스트 → 이미지 → 비디오 → 음성 생성 파이프라인",
  "version": "1.0.0",
  "created_at": "2025-11-17T11:00:00Z",
  "updated_at": "2025-11-17T11:00:00Z",
  "user_id": "user_123",

  "nodes": [
    {
      "id": "node_input_1",
      "type": "utility.input",
      "name": "Story Prompt",
      "position": { "x": 100, "y": 300 },
      "parameters": {
        "input_type": "text",
        "default_value": "A cat exploring a futuristic city"
      },
      "enabled": true
    },
    {
      "id": "node_llm_1",
      "type": "llm.inference",
      "name": "Generate Description",
      "position": { "x": 350, "y": 300 },
      "parameters": {
        "model": "llama-3-70b",
        "system_prompt": "You are a creative writer. Expand the prompt into a detailed visual description.",
        "temperature": 0.9,
        "max_tokens": 300
      },
      "enabled": true
    },
    {
      "id": "node_image_1",
      "type": "image.generation",
      "name": "Generate Image",
      "position": { "x": 600, "y": 200 },
      "parameters": {
        "model": "stable-diffusion-xl",
        "width": 1024,
        "height": 1024,
        "steps": 50,
        "guidance_scale": 7.5
      },
      "enabled": true
    },
    {
      "id": "node_video_1",
      "type": "video.generation",
      "name": "Generate Video",
      "position": { "x": 850, "y": 200 },
      "parameters": {
        "model": "stable-video-diffusion",
        "duration": 4,
        "fps": 24,
        "motion_scale": 1.2
      },
      "enabled": true
    },
    {
      "id": "node_audio_1",
      "type": "audio.generation",
      "name": "Generate Narration",
      "position": { "x": 600, "y": 400 },
      "parameters": {
        "model": "xtts-v2",
        "voice": "female-1",
        "language": "en",
        "speed": 1.0
      },
      "enabled": true
    },
    {
      "id": "node_storage_1",
      "type": "resource.storage",
      "name": "Save Results",
      "position": { "x": 1100, "y": 300 },
      "parameters": {
        "storage_type": "gp_storage",
        "tier": "hot",
        "size_gb": 5
      },
      "enabled": true
    },
    {
      "id": "node_output_1",
      "type": "utility.output",
      "name": "Final Output",
      "position": { "x": 1350, "y": 300 },
      "parameters": {
        "output_format": "json"
      },
      "enabled": true
    }
  ],

  "connections": [
    {
      "id": "conn_1",
      "source_node_id": "node_input_1",
      "source_output": "data",
      "target_node_id": "node_llm_1",
      "target_input": "prompt",
      "enabled": true
    },
    {
      "id": "conn_2",
      "source_node_id": "node_llm_1",
      "source_output": "response",
      "target_node_id": "node_image_1",
      "target_input": "prompt",
      "enabled": true
    },
    {
      "id": "conn_3",
      "source_node_id": "node_image_1",
      "source_output": "image_data",
      "target_node_id": "node_video_1",
      "target_input": "init_image",
      "enabled": true
    },
    {
      "id": "conn_4",
      "source_node_id": "node_llm_1",
      "source_output": "response",
      "target_node_id": "node_audio_1",
      "target_input": "text",
      "enabled": true
    },
    {
      "id": "conn_5",
      "source_node_id": "node_video_1",
      "source_output": "video_data",
      "target_node_id": "node_storage_1",
      "target_input": "data",
      "enabled": true
    },
    {
      "id": "conn_6",
      "source_node_id": "node_storage_1",
      "source_output": "storage_url",
      "target_node_id": "node_output_1",
      "target_input": "data",
      "enabled": true
    }
  ],

  "metadata": {
    "tags": ["multimodal", "content-generation", "pipeline"],
    "category": "content_creation",
    "is_public": true,
    "fork_count": 0,
    "execution_count": 0
  },

  "settings": {
    "timeout_seconds": 600,
    "max_retries": 2,
    "error_handling": "continue",
    "parallel_execution": true
  }
}
```

**예시 3: Agent-to-Agent Debate**
```json
{
  "id": "wf_003",
  "name": "AI Debate",
  "description": "여러 AI 에이전트가 토론하여 최선의 답을 도출",
  "version": "1.0.0",
  "created_at": "2025-11-17T12:00:00Z",
  "updated_at": "2025-11-17T12:00:00Z",
  "user_id": "user_456",

  "nodes": [
    {
      "id": "node_input_1",
      "type": "utility.input",
      "name": "Question",
      "position": { "x": 100, "y": 300 },
      "parameters": {
        "input_type": "text",
        "default_value": "What is the best way to solve climate change?"
      },
      "enabled": true
    },
    {
      "id": "node_memory_1",
      "type": "resource.memory",
      "name": "Debate Memory",
      "position": { "x": 350, "y": 200 },
      "parameters": {
        "memory_type": "context_memory",
        "size_gb": 4.0,
        "ttl_hours": 2
      },
      "enabled": true
    },
    {
      "id": "node_agent_1",
      "type": "agent.communication",
      "name": "AI Debate",
      "position": { "x": 600, "y": 300 },
      "parameters": {
        "agent_ids": ["agent_optimist", "agent_pessimist", "agent_pragmatist"],
        "protocol": "debate",
        "timeout": 60
      },
      "enabled": true
    },
    {
      "id": "node_llm_summarize",
      "type": "llm.inference",
      "name": "Summarize Consensus",
      "position": { "x": 850, "y": 300 },
      "parameters": {
        "model": "llama-3-70b",
        "system_prompt": "Summarize the key points from the debate.",
        "temperature": 0.5,
        "max_tokens": 500
      },
      "enabled": true
    },
    {
      "id": "node_output_1",
      "type": "utility.output",
      "name": "Final Answer",
      "position": { "x": 1100, "y": 300 },
      "parameters": {
        "output_format": "json"
      },
      "enabled": true
    }
  ],

  "connections": [
    {
      "id": "conn_1",
      "source_node_id": "node_input_1",
      "source_output": "data",
      "target_node_id": "node_agent_1",
      "target_input": "message",
      "enabled": true
    },
    {
      "id": "conn_2",
      "source_node_id": "node_memory_1",
      "source_output": "memory_id",
      "target_node_id": "node_agent_1",
      "target_input": "agent_context",
      "enabled": true
    },
    {
      "id": "conn_3",
      "source_node_id": "node_agent_1",
      "source_output": "responses",
      "target_node_id": "node_llm_summarize",
      "target_input": "prompt",
      "enabled": true
    },
    {
      "id": "conn_4",
      "source_node_id": "node_llm_summarize",
      "source_output": "response",
      "target_node_id": "node_output_1",
      "target_input": "data",
      "enabled": true
    }
  ],

  "metadata": {
    "tags": ["agent", "debate", "multi-agent"],
    "category": "research",
    "is_public": true,
    "fork_count": 5,
    "execution_count": 23
  },

  "settings": {
    "timeout_seconds": 180,
    "max_retries": 1,
    "error_handling": "stop",
    "parallel_execution": false
  }
}
```

---

## 4. Frontend UI 설계

### 4.1 기술 스택

```yaml
Framework: Next.js 14 (React 18)
UI Library:
  - React Flow (노드 기반 에디터)
  - Tailwind CSS (스타일링)
  - shadcn/ui (UI 컴포넌트)
State Management: Zustand or Jotai
API Client: TanStack Query (React Query)
Code Editor: Monaco Editor (JSON 편집용)
Icons: Lucide React
```

### 4.2 주요 컴포넌트

#### 4.2.1 Workflow Canvas (WorkflowCanvas.tsx)

```typescript
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  MiniMap,
  Panel
} from 'reactflow';
import 'reactflow/dist/style.css';

interface WorkflowCanvasProps {
  workflow: Workflow;
  onNodesChange: (nodes: Node[]) => void;
  onEdgesChange: (edges: Edge[]) => void;
  onConnect: (connection: Connection) => void;
}

export function WorkflowCanvas({
  workflow,
  onNodesChange,
  onEdgesChange,
  onConnect
}: WorkflowCanvasProps) {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  // workflow.nodes를 ReactFlow Node로 변환
  useEffect(() => {
    const flowNodes = workflow.nodes.map(node => ({
      id: node.id,
      type: node.type,
      position: node.position,
      data: {
        label: node.name,
        parameters: node.parameters,
        enabled: node.enabled
      }
    }));
    setNodes(flowNodes);

    const flowEdges = workflow.connections.map(conn => ({
      id: conn.id,
      source: conn.source_node_id,
      sourceHandle: conn.source_output,
      target: conn.target_node_id,
      targetHandle: conn.target_input,
      animated: conn.enabled
    }));
    setEdges(flowEdges);
  }, [workflow]);

  return (
    <div className="h-screen w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
      >
        <Background />
        <Controls />
        <MiniMap />
        <Panel position="top-right">
          <ExecuteButton workflowId={workflow.id} />
          <ExportButton workflow={workflow} />
        </Panel>
      </ReactFlow>
    </div>
  );
}
```

#### 4.2.2 Node Palette (NodePalette.tsx)

```typescript
interface NodePaletteProps {
  onNodeDragStart: (nodeType: string) => void;
}

export function NodePalette({ onNodeDragStart }: NodePaletteProps) {
  const nodeCategories = {
    "AI Services": [
      { type: "llm.inference", name: "LLM", icon: "Brain", color: "#4F46E5" },
      { type: "image.generation", name: "Image Gen", icon: "Image", color: "#EC4899" },
      { type: "video.generation", name: "Video Gen", icon: "Video", color: "#F59E0B" },
      { type: "audio.generation", name: "Audio Gen", icon: "Mic", color: "#10B981" },
      { type: "agent.communication", name: "Agent", icon: "Users", color: "#8B5CF6" }
    ],
    "Resources": [
      { type: "resource.memory", name: "Memory", icon: "Cpu", color: "#6366F1" },
      { type: "resource.storage", name: "Storage", icon: "Database", color: "#14B8A6" }
    ],
    "Utilities": [
      { type: "utility.input", name: "Input", icon: "ArrowDownCircle", color: "#64748B" },
      { type: "utility.output", name: "Output", icon: "ArrowUpCircle", color: "#64748B" },
      { type: "utility.condition", name: "If-Else", icon: "GitBranch", color: "#64748B" },
      { type: "utility.loop", name: "Loop", icon: "RefreshCw", color: "#64748B" }
    ]
  };

  return (
    <div className="w-64 bg-gray-50 border-r p-4 overflow-y-auto">
      <h2 className="text-lg font-bold mb-4">Nodes</h2>

      {Object.entries(nodeCategories).map(([category, nodes]) => (
        <div key={category} className="mb-6">
          <h3 className="text-sm font-semibold text-gray-600 mb-2">
            {category}
          </h3>
          <div className="space-y-2">
            {nodes.map(node => (
              <div
                key={node.type}
                draggable
                onDragStart={() => onNodeDragStart(node.type)}
                className="p-3 bg-white rounded-lg border cursor-move hover:shadow-md transition"
                style={{ borderLeftColor: node.color, borderLeftWidth: 3 }}
              >
                <div className="flex items-center gap-2">
                  <Icon name={node.icon} size={16} color={node.color} />
                  <span className="text-sm font-medium">{node.name}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
```

#### 4.2.3 Node Properties Panel (NodePropertiesPanel.tsx)

```typescript
interface NodePropertiesPanelProps {
  node: WorkflowNode | null;
  onUpdate: (nodeId: string, parameters: Record<string, any>) => void;
}

export function NodePropertiesPanel({ node, onUpdate }: NodePropertiesPanelProps) {
  if (!node) {
    return (
      <div className="w-80 bg-gray-50 border-l p-4">
        <p className="text-gray-500">노드를 선택하세요</p>
      </div>
    );
  }

  const nodeDefinition = getNodeDefinition(node.type);

  return (
    <div className="w-80 bg-gray-50 border-l p-4 overflow-y-auto">
      <h2 className="text-lg font-bold mb-4">{node.name}</h2>
      <p className="text-sm text-gray-600 mb-4">{nodeDefinition.description}</p>

      <div className="space-y-4">
        {Object.entries(nodeDefinition.parameters).map(([key, param]) => (
          <div key={key}>
            <label className="block text-sm font-medium mb-1">
              {param.label || key}
            </label>

            {param.type === "select" && (
              <select
                value={node.parameters[key]}
                onChange={(e) => onUpdate(node.id, {
                  ...node.parameters,
                  [key]: e.target.value
                })}
                className="w-full px-3 py-2 border rounded-md"
              >
                {param.options.map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            )}

            {param.type === "number" && (
              <input
                type="number"
                value={node.parameters[key]}
                min={param.min}
                max={param.max}
                step={param.step || 0.1}
                onChange={(e) => onUpdate(node.id, {
                  ...node.parameters,
                  [key]: parseFloat(e.target.value)
                })}
                className="w-full px-3 py-2 border rounded-md"
              />
            )}

            {param.type === "boolean" && (
              <input
                type="checkbox"
                checked={node.parameters[key]}
                onChange={(e) => onUpdate(node.id, {
                  ...node.parameters,
                  [key]: e.target.checked
                })}
                className="h-4 w-4"
              />
            )}

            {param.type === "string" && (
              <input
                type="text"
                value={node.parameters[key]}
                onChange={(e) => onUpdate(node.id, {
                  ...node.parameters,
                  [key]: e.target.value
                })}
                className="w-full px-3 py-2 border rounded-md"
              />
            )}

            {param.description && (
              <p className="text-xs text-gray-500 mt-1">
                {param.description}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

### 4.3 화면 레이아웃

```
┌─────────────────────────────────────────────────────────────┐
│  Header: Workflow Name | Save | Execute | Export | Settings │
├──────────┬────────────────────────────────────┬──────────────┤
│          │                                    │              │
│  Node    │       Workflow Canvas              │  Properties  │
│  Palette │       (ReactFlow)                  │  Panel       │
│          │                                    │              │
│  - AI    │  ┌──────┐      ┌──────┐           │  Node: LLM   │
│    LLM   │  │Input │─────▶│ LLM  │──┐        │              │
│    Image │  └──────┘      └──────┘  │        │  Parameters: │
│    Video │                          │        │  - model     │
│    Audio │  ┌──────┐      ┌──────┐  │        │  - temp      │
│    Agent │  │Image │─────▶│Video │◀─┘        │  - tokens    │
│          │  └──────┘      └──────┘           │              │
│  - Res   │                 │                 │              │
│    Memory│                 ▼                 │              │
│    Store │              ┌──────┐             │              │
│          │              │Output│             │              │
│  - Utils │              └──────┘             │              │
│    Input │                                    │              │
│    Output│  MiniMap  Controls  Background   │              │
│    If    │                                    │              │
│    Loop  │                                    │              │
│          │                                    │              │
└──────────┴────────────────────────────────────┴──────────────┘
│  Footer: Execution Status | Logs | Cost Estimation          │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. SDK 설계

### 5.1 JavaScript/TypeScript SDK

**설치**
```bash
npm install @distributed-llm/workflow-sdk
```

**사용 예시**
```typescript
import { WorkflowClient, Workflow } from '@distributed-llm/workflow-sdk';

// Client 초기화
const client = new WorkflowClient({
  apiKey: 'user_xxxxx',
  baseUrl: 'https://api.distributed-llm.com'
});

// Workflow JSON에서 로드
const workflowJson = require('./my-workflow.json');
const workflow = Workflow.fromJSON(workflowJson);

// 또는 프로그래밍 방식으로 생성
const workflow = new Workflow('My Workflow')
  .addNode('input', { type: 'utility.input', parameters: { input_type: 'text' } })
  .addNode('llm', {
    type: 'llm.inference',
    parameters: {
      model: 'llama-3-70b',
      temperature: 0.7
    }
  })
  .addNode('output', { type: 'utility.output', parameters: { output_format: 'json' } })
  .connect('input', 'data', 'llm', 'prompt')
  .connect('llm', 'response', 'output', 'data');

// Workflow 실행
const execution = await client.executeWorkflow(workflow, {
  input: { data: '안녕하세요' }
});

console.log('Execution ID:', execution.id);

// 실행 상태 모니터링
const status = await client.getExecutionStatus(execution.id);
console.log('Status:', status.state);  // running | completed | failed

// 실행 결과 가져오기
if (status.state === 'completed') {
  const result = await client.getExecutionResult(execution.id);
  console.log('Result:', result.output);
}

// 스트리밍 실행 (실시간)
await client.executeWorkflowStream(workflow, {
  input: { data: '안녕하세요' },
  onProgress: (nodeId, output) => {
    console.log(`Node ${nodeId}:`, output);
  },
  onComplete: (result) => {
    console.log('Final result:', result);
  },
  onError: (error) => {
    console.error('Error:', error);
  }
});
```

**SDK API Reference**
```typescript
class WorkflowClient {
  constructor(config: ClientConfig);

  // Workflow CRUD
  createWorkflow(workflow: Workflow): Promise<Workflow>;
  getWorkflow(id: string): Promise<Workflow>;
  updateWorkflow(id: string, updates: Partial<Workflow>): Promise<Workflow>;
  deleteWorkflow(id: string): Promise<void>;
  listWorkflows(filters?: WorkflowFilters): Promise<Workflow[]>;

  // Execution
  executeWorkflow(workflow: Workflow, input: Record<string, any>): Promise<Execution>;
  executeWorkflowStream(workflow: Workflow, options: StreamOptions): Promise<void>;
  getExecutionStatus(executionId: string): Promise<ExecutionStatus>;
  getExecutionResult(executionId: string): Promise<ExecutionResult>;
  cancelExecution(executionId: string): Promise<void>;

  // Export/Import
  exportWorkflowJSON(workflowId: string): Promise<string>;
  importWorkflowJSON(json: string): Promise<Workflow>;
}

class Workflow {
  id: string;
  name: string;
  nodes: WorkflowNode[];
  connections: Connection[];

  constructor(name: string);

  // Builder pattern
  addNode(id: string, config: NodeConfig): Workflow;
  removeNode(id: string): Workflow;
  connect(sourceId: string, sourceOutput: string, targetId: string, targetInput: string): Workflow;
  disconnect(connectionId: string): Workflow;

  // Serialization
  toJSON(): string;
  static fromJSON(json: string): Workflow;

  // Validation
  validate(): ValidationResult;
}
```

### 5.2 Java SDK

**Maven 의존성**
```xml
<dependency>
  <groupId>com.distributed-llm</groupId>
  <artifactId>workflow-sdk</artifactId>
  <version>1.0.0</version>
</dependency>
```

**사용 예시**
```java
import com.distributedllm.workflow.*;

// Client 초기화
WorkflowClient client = new WorkflowClient.Builder()
    .apiKey("user_xxxxx")
    .baseUrl("https://api.distributed-llm.com")
    .build();

// Workflow JSON에서 로드
String json = Files.readString(Path.of("my-workflow.json"));
Workflow workflow = Workflow.fromJSON(json);

// 또는 Builder 패턴으로 생성
Workflow workflow = new Workflow("My Workflow")
    .addNode("input", NodeConfig.builder()
        .type("utility.input")
        .parameter("input_type", "text")
        .build())
    .addNode("llm", NodeConfig.builder()
        .type("llm.inference")
        .parameter("model", "llama-3-70b")
        .parameter("temperature", 0.7)
        .build())
    .addNode("output", NodeConfig.builder()
        .type("utility.output")
        .parameter("output_format", "json")
        .build())
    .connect("input", "data", "llm", "prompt")
    .connect("llm", "response", "output", "data");

// Workflow 실행
Map<String, Object> input = Map.of("data", "안녕하세요");
Execution execution = client.executeWorkflow(workflow, input);

System.out.println("Execution ID: " + execution.getId());

// 동기 실행 (결과를 기다림)
ExecutionResult result = execution.waitForCompletion();
System.out.println("Result: " + result.getOutput());

// 비동기 실행
execution.onComplete(result -> {
    System.out.println("Completed: " + result.getOutput());
}).onError(error -> {
    System.err.println("Error: " + error.getMessage());
}).onProgress((nodeId, output) -> {
    System.out.println("Node " + nodeId + ": " + output);
});
```

### 5.3 Python SDK

**설치**
```bash
pip install distributed-llm-workflow
```

**사용 예시**
```python
from distributed_llm.workflow import WorkflowClient, Workflow

# Client 초기화
client = WorkflowClient(
    api_key='user_xxxxx',
    base_url='https://api.distributed-llm.com'
)

# Workflow JSON에서 로드
import json
with open('my-workflow.json') as f:
    workflow_dict = json.load(f)
workflow = Workflow.from_dict(workflow_dict)

# 또는 Builder 패턴으로 생성
workflow = (Workflow('My Workflow')
    .add_node('input', type='utility.input', parameters={'input_type': 'text'})
    .add_node('llm', type='llm.inference', parameters={
        'model': 'llama-3-70b',
        'temperature': 0.7
    })
    .add_node('output', type='utility.output', parameters={'output_format': 'json'})
    .connect('input', 'data', 'llm', 'prompt')
    .connect('llm', 'response', 'output', 'data')
)

# Workflow 실행
execution = client.execute_workflow(workflow, input={'data': '안녕하세요'})

print(f'Execution ID: {execution.id}')

# 동기 실행 (결과를 기다림)
result = execution.wait_for_completion()
print(f'Result: {result.output}')

# 스트리밍 실행
async for event in client.execute_workflow_stream(workflow, input={'data': '안녕하세요'}):
    if event.type == 'node_complete':
        print(f'Node {event.node_id}: {event.output}')
    elif event.type == 'complete':
        print(f'Final result: {event.output}')
    elif event.type == 'error':
        print(f'Error: {event.error}')
```

---

## 6. API Specification

### 6.1 Workflow Management API

#### 6.1.1 Create Workflow
```
POST /v1/workflows
Authorization: Bearer <user_api_key>
Content-Type: application/json

{
  "name": "My Workflow",
  "description": "Description here",
  "nodes": [...],
  "connections": [...],
  "settings": {...}
}

Response 201:
{
  "id": "wf_xxx",
  "name": "My Workflow",
  "created_at": "2025-11-17T10:00:00Z",
  ...
}
```

#### 6.1.2 Get Workflow
```
GET /v1/workflows/{workflow_id}
Authorization: Bearer <user_api_key>

Response 200:
{
  "id": "wf_xxx",
  "name": "My Workflow",
  "nodes": [...],
  "connections": [...],
  ...
}
```

#### 6.1.3 Update Workflow
```
PATCH /v1/workflows/{workflow_id}
Authorization: Bearer <user_api_key>
Content-Type: application/json

{
  "name": "Updated Name",
  "nodes": [...]
}

Response 200:
{
  "id": "wf_xxx",
  "name": "Updated Name",
  "updated_at": "2025-11-17T11:00:00Z",
  ...
}
```

#### 6.1.4 Delete Workflow
```
DELETE /v1/workflows/{workflow_id}
Authorization: Bearer <user_api_key>

Response 204: No Content
```

#### 6.1.5 List Workflows
```
GET /v1/workflows?category=chatbot&tag=llm&limit=20&offset=0
Authorization: Bearer <user_api_key>

Response 200:
{
  "workflows": [
    {
      "id": "wf_001",
      "name": "Simple Chat",
      "category": "chatbot",
      "tags": ["llm", "chat"],
      "execution_count": 42,
      "created_at": "2025-11-17T10:00:00Z"
    },
    ...
  ],
  "total": 156,
  "limit": 20,
  "offset": 0
}
```

### 6.2 Workflow Execution API

#### 6.2.1 Execute Workflow
```
POST /v1/workflows/{workflow_id}/execute
Authorization: Bearer <user_api_key>
Content-Type: application/json

{
  "input": {
    "data": "안녕하세요"
  },
  "options": {
    "timeout_seconds": 300,
    "enable_logging": true
  }
}

Response 202:
{
  "execution_id": "exec_xxx",
  "workflow_id": "wf_xxx",
  "state": "running",
  "created_at": "2025-11-17T12:00:00Z",
  "estimated_completion": "2025-11-17T12:05:00Z"
}
```

#### 6.2.2 Get Execution Status
```
GET /v1/executions/{execution_id}
Authorization: Bearer <user_api_key>

Response 200:
{
  "execution_id": "exec_xxx",
  "workflow_id": "wf_xxx",
  "state": "completed",  // running | completed | failed | cancelled
  "progress": {
    "completed_nodes": 3,
    "total_nodes": 3,
    "current_node": null
  },
  "started_at": "2025-11-17T12:00:00Z",
  "completed_at": "2025-11-17T12:03:45Z",
  "cost_usd": 0.0025
}
```

#### 6.2.3 Get Execution Result
```
GET /v1/executions/{execution_id}/result
Authorization: Bearer <user_api_key>

Response 200:
{
  "execution_id": "exec_xxx",
  "output": {
    "data": "안녕하세요! 무엇을 도와드릴까요?"
  },
  "node_outputs": {
    "node_input_1": { "data": "안녕하세요" },
    "node_llm_1": {
      "response": "안녕하세요! 무엇을 도와드릴까요?",
      "tokens_used": { "input": 10, "output": 25 }
    },
    "node_output_1": { "data": "안녕하세요! 무엇을 도와드릴까요?" }
  },
  "metrics": {
    "total_duration_ms": 3450,
    "total_cost_usd": 0.0025
  }
}
```

#### 6.2.4 Stream Execution (WebSocket or SSE)
```
GET /v1/executions/{execution_id}/stream
Authorization: Bearer <user_api_key>
Accept: text/event-stream

Response (SSE):
event: node_start
data: {"node_id": "node_llm_1", "timestamp": "2025-11-17T12:00:01Z"}

event: node_complete
data: {"node_id": "node_llm_1", "output": {...}, "timestamp": "2025-11-17T12:00:03Z"}

event: complete
data: {"execution_id": "exec_xxx", "output": {...}, "cost_usd": 0.0025}
```

#### 6.2.5 Cancel Execution
```
POST /v1/executions/{execution_id}/cancel
Authorization: Bearer <user_api_key>

Response 200:
{
  "execution_id": "exec_xxx",
  "state": "cancelled",
  "cancelled_at": "2025-11-17T12:01:00Z"
}
```

### 6.3 Workflow Templates API

#### 6.3.1 List Public Templates
```
GET /v1/workflows/templates?category=content_creation&limit=20
Authorization: Bearer <user_api_key>

Response 200:
{
  "templates": [
    {
      "id": "wf_002",
      "name": "Multimodal Content Pipeline",
      "description": "텍스트 → 이미지 → 비디오 생성",
      "category": "content_creation",
      "tags": ["multimodal", "content-generation"],
      "fork_count": 156,
      "rating": 4.8,
      "author": "user_123"
    },
    ...
  ]
}
```

#### 6.3.2 Fork Template
```
POST /v1/workflows/{template_id}/fork
Authorization: Bearer <user_api_key>
Content-Type: application/json

{
  "name": "My Custom Workflow"
}

Response 201:
{
  "id": "wf_new_xxx",
  "name": "My Custom Workflow",
  "forked_from": "wf_002",
  ...
}
```

---

## 7. Workflow Execution Engine

### 7.1 실행 로직

```python
class WorkflowExecutor:
    """Workflow 실행 엔진"""

    def __init__(self, workflow: Workflow, execution_id: str):
        self.workflow = workflow
        self.execution_id = execution_id
        self.state = {}  # 노드 간 데이터 공유

    async def execute(self, input_data: dict) -> dict:
        """Workflow 실행"""
        # 1. 실행 순서 결정 (topological sort)
        execution_order = self._calculate_execution_order()

        # 2. 초기 입력 설정
        self.state['input'] = input_data

        # 3. 각 노드 실행
        for node_id in execution_order:
            node = self.workflow.get_node(node_id)

            # 노드가 비활성화되어 있으면 스킵
            if not node.enabled:
                continue

            # 노드 입력 준비
            node_input = self._prepare_node_input(node)

            # 노드 실행
            try:
                node_output = await self._execute_node(node, node_input)
                self.state[node_id] = node_output

                # 진행 상황 업데이트
                await self._update_progress(node_id, node_output)

            except Exception as e:
                # 에러 처리 전략에 따라 처리
                if self.workflow.settings.error_handling == "stop":
                    raise
                elif self.workflow.settings.error_handling == "continue":
                    self.state[node_id] = {"error": str(e)}
                    continue
                elif self.workflow.settings.error_handling == "rollback":
                    await self._rollback()
                    raise

        # 4. 최종 출력 반환
        output_nodes = [n for n in self.workflow.nodes if n.type == "utility.output"]
        return {
            "output": self.state.get(output_nodes[0].id) if output_nodes else None,
            "node_outputs": self.state,
            "execution_id": self.execution_id
        }

    def _calculate_execution_order(self) -> List[str]:
        """Topological sort로 실행 순서 결정"""
        # DAG의 topological sort
        graph = self._build_graph()
        return topological_sort(graph)

    async def _execute_node(self, node: WorkflowNode, input_data: dict) -> dict:
        """개별 노드 실행"""
        # 노드 타입에 따라 적절한 서비스 호출
        if node.type == "llm.inference":
            return await self._execute_llm_node(node, input_data)
        elif node.type == "image.generation":
            return await self._execute_image_gen_node(node, input_data)
        elif node.type == "video.generation":
            return await self._execute_video_gen_node(node, input_data)
        elif node.type == "audio.generation":
            return await self._execute_audio_gen_node(node, input_data)
        elif node.type == "agent.communication":
            return await self._execute_agent_comm_node(node, input_data)
        elif node.type == "resource.memory":
            return await self._execute_memory_alloc_node(node, input_data)
        elif node.type == "resource.storage":
            return await self._execute_storage_alloc_node(node, input_data)
        elif node.type.startswith("utility."):
            return await self._execute_utility_node(node, input_data)
        else:
            raise ValueError(f"Unknown node type: {node.type}")

    async def _execute_llm_node(self, node: WorkflowNode, input_data: dict) -> dict:
        """LLM 노드 실행"""
        # Router의 LLM API 호출
        response = await self.router_client.chat_completion(
            model=node.parameters['model'],
            messages=[
                {"role": "system", "content": node.parameters.get('system_prompt', '')},
                {"role": "user", "content": input_data.get('prompt', '')}
            ],
            temperature=node.parameters.get('temperature', 0.7),
            max_tokens=node.parameters.get('max_tokens', 1000),
            enable_cache=node.parameters.get('enable_cache', True)
        )

        return {
            "response": response['choices'][0]['message']['content'],
            "tokens_used": response['usage']
        }
```

### 7.2 병렬 실행 지원

```python
async def execute_parallel(self, input_data: dict) -> dict:
    """병렬 실행 가능한 노드들을 동시에 실행"""
    if not self.workflow.settings.parallel_execution:
        return await self.execute(input_data)

    # 1. 실행 레벨 계산 (같은 레벨의 노드들은 병렬 실행 가능)
    execution_levels = self._calculate_execution_levels()

    # 2. 각 레벨별로 노드 실행
    for level, node_ids in execution_levels.items():
        # 같은 레벨의 노드들을 병렬 실행
        tasks = []
        for node_id in node_ids:
            node = self.workflow.get_node(node_id)
            if node.enabled:
                node_input = self._prepare_node_input(node)
                tasks.append(self._execute_node(node, node_input))

        # 모든 노드가 완료될 때까지 대기
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 저장
        for node_id, result in zip(node_ids, results):
            if isinstance(result, Exception):
                # 에러 처리
                if self.workflow.settings.error_handling == "stop":
                    raise result
            else:
                self.state[node_id] = result

    # 3. 최종 출력 반환
    output_nodes = [n for n in self.workflow.nodes if n.type == "utility.output"]
    return {
        "output": self.state.get(output_nodes[0].id) if output_nodes else None,
        "node_outputs": self.state
    }
```

---

## 8. 비용 추정

### 8.1 실행 전 비용 예측

```python
class CostEstimator:
    """Workflow 실행 비용 예측"""

    def estimate_workflow_cost(self, workflow: Workflow, input_data: dict) -> dict:
        """워크플로우 실행 비용 추정"""
        total_cost = Decimal("0")
        breakdown = {}

        for node in workflow.nodes:
            node_cost = self._estimate_node_cost(node, input_data)
            breakdown[node.id] = {
                "node_name": node.name,
                "node_type": node.type,
                "estimated_cost_usd": float(node_cost)
            }
            total_cost += node_cost

        return {
            "total_estimated_cost_usd": float(total_cost),
            "breakdown": breakdown,
            "currency": "USD"
        }

    def _estimate_node_cost(self, node: WorkflowNode, input_data: dict) -> Decimal:
        """개별 노드 비용 추정"""
        if node.type == "llm.inference":
            # 예상 토큰 수 계산 (대략적)
            prompt_length = len(input_data.get('prompt', ''))
            estimated_input_tokens = prompt_length // 4  # 대략 4 char = 1 token
            estimated_output_tokens = node.parameters.get('max_tokens', 1000)

            # LLM 비용 계산
            return (
                Decimal(estimated_input_tokens) * Decimal("0.000001") * Decimal("1.05") +
                Decimal(estimated_output_tokens) * Decimal("0.000008") * Decimal("1.05")
            )

        elif node.type == "image.generation":
            # 이미지 생성 비용 (모델별)
            model_costs = {
                "stable-diffusion-xl": Decimal("0.02"),
                "dall-e-3": Decimal("0.04")
            }
            return model_costs.get(node.parameters['model'], Decimal("0.02"))

        elif node.type == "video.generation":
            # 비디오 생성 비용 (초당)
            duration = node.parameters.get('duration', 4)
            cost_per_second = Decimal("0.10")
            return Decimal(duration) * cost_per_second

        elif node.type == "audio.generation":
            # 오디오 생성 비용 (문자당)
            text_length = len(input_data.get('text', ''))
            cost_per_char = Decimal("0.00001")
            return Decimal(text_length) * cost_per_char

        elif node.type == "resource.memory":
            # 메모리 비용
            size_gb = node.parameters.get('size_gb', 2.0)
            ttl_hours = node.parameters.get('ttl_hours', 24)
            hourly_rate = Decimal("0.036")
            return Decimal(size_gb) * Decimal(ttl_hours) * hourly_rate

        elif node.type == "resource.storage":
            # 스토리지 비용
            size_gb = node.parameters.get('size_gb', 10)
            tier = node.parameters.get('tier', 'hot')
            hourly_rate = Decimal("0.0012") if tier == "hot" else Decimal("0.00024")
            # 1시간 기준
            return Decimal(size_gb) * hourly_rate

        else:
            return Decimal("0")
```

---

## 9. 구현 우선순위

### Phase 1: 기본 워크플로우 (Week 1-4)

**Week 1-2: Backend API**
- [ ] Workflow 데이터 모델 설계
- [ ] Workflow CRUD API 구현
- [ ] Workflow validation 로직
- [ ] Database schema (workflows, executions, execution_logs)

**Week 3-4: Execution Engine**
- [ ] 기본 실행 엔진 (sequential)
- [ ] LLM 노드 실행
- [ ] Input/Output 노드 실행
- [ ] Execution API 구현

### Phase 2: Frontend UI (Week 5-8)

**Week 5-6: Canvas Editor**
- [ ] React Flow 기반 캔버스 구현
- [ ] Node Palette 구현
- [ ] 노드 드래그 앤 드롭
- [ ] 연결 생성/삭제

**Week 7-8: Properties & Export**
- [ ] Node Properties Panel
- [ ] JSON Export/Import
- [ ] Workflow 저장/로드
- [ ] 실행 및 모니터링 UI

### Phase 3: Advanced Nodes (Week 9-12)

**Week 9-10: AI Service Nodes**
- [ ] Image Generation 노드
- [ ] Video Generation 노드
- [ ] Audio Generation 노드
- [ ] Agent Communication 노드

**Week 11-12: Resource & Utility Nodes**
- [ ] Memory/Storage 노드
- [ ] Conditional 노드
- [ ] Loop 노드
- [ ] 병렬 실행 지원

### Phase 4: SDK (Week 13-16)

**Week 13-14: JavaScript SDK**
- [ ] SDK 기본 구조
- [ ] Workflow Builder API
- [ ] Execution API
- [ ] Documentation

**Week 15-16: Java & Python SDK**
- [ ] Java SDK
- [ ] Python SDK
- [ ] 예제 코드
- [ ] Documentation

### Phase 5: Production Features (Week 17-20)

**Week 17-18: Templates & Marketplace**
- [ ] Public workflow templates
- [ ] Fork 기능
- [ ] Rating & Review
- [ ] Template discovery

**Week 19-20: Optimization**
- [ ] 비용 추정 기능
- [ ] 실행 최적화
- [ ] Caching 전략
- [ ] Performance tuning

---

## 10. 성공 지표 (KPI)

### Technical KPIs
- **Workflow 생성 성공률**: > 95%
- **Workflow 실행 성공률**: > 90%
- **평균 실행 시간**: < 30초 (단순 워크플로우)
- **SDK 다운로드 수**: 월 1,000+ (각 언어별)

### Business KPIs
- **활성 사용자**: 500명 이상
- **생성된 Workflow 수**: 5,000개 이상
- **Public Template 수**: 100개 이상
- **사용자 만족도**: 4.5/5.0 이상

---

## 11. 참고 자료

### 유사 플랫폼
- **n8n**: https://n8n.io/ - Workflow automation
- **Zapier**: https://zapier.com/ - No-code automation
- **Retool Workflows**: https://retool.com/products/workflows
- **LangFlow**: https://github.com/logspace-ai/langflow - LLM workflow builder

### 기술 스택
- **React Flow**: https://reactflow.dev/ - Node-based UI library
- **Monaco Editor**: https://microsoft.github.io/monaco-editor/ - Code editor
- **shadcn/ui**: https://ui.shadcn.com/ - UI components
- **Zustand**: https://github.com/pmndrs/zustand - State management
