# Multimodal AI 실전 가이드 (2024-2025)

## 목차
1. [개요](#개요)
2. [Vision-Language Models](#vision-language-models)
3. [Audio Processing](#audio-processing)
4. [Video Understanding](#video-understanding)
5. [실전 구현 패턴](#실전-구현-패턴)
6. [프로덕션 배포](#프로덕션-배포)

---

## 개요

### Multimodal AI란?
**정의**: 텍스트, 이미지, 오디오, 비디오 등 여러 모달리티를 동시에 처리하는 AI

**왜 중요한가?**
- 인간의 인식 방식과 유사 (시각 + 청각 + 언어)
- 더 풍부한 컨텍스트 이해
- 새로운 응용 분야 (문서 이해, 비디오 분석, AR/VR)

### 2024-2025 주요 모델

| 모델 | 회사 | 모달리티 | 특징 |
|------|------|----------|------|
| GPT-4V | OpenAI | 텍스트 + 이미지 | 최고 성능, 복잡한 시각 추론 |
| Gemini 1.5 Pro | Google | 텍스트 + 이미지 + 오디오 + 비디오 | 1M+ 토큰 컨텍스트, 네이티브 멀티모달 |
| Claude 3 Opus | Anthropic | 텍스트 + 이미지 | 정확한 문서 이해, OCR |
| LLaVA 1.6 | LMM Lab | 텍스트 + 이미지 | 오픈소스, 세밀한 이미지 이해 |
| Qwen-VL | Alibaba | 텍스트 + 이미지 | 다국어 지원 |

---

## Vision-Language Models

### 1. 기본 아키텍처

```python
"""
Vision-Language Model 구조

[Image] → Vision Encoder → Visual Features
                              ↓
                        Projection Layer
                              ↓
[Text]  → Text Tokenizer → [Visual Tokens + Text Tokens] → LLM → Output

핵심: 이미지를 "토큰"처럼 취급!
"""

class VisionLanguageModel:
    def __init__(self):
        # 1. Vision Encoder (CLIP, SigLIP 등)
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

        # 2. Projection (이미지 특징 → LLM 임베딩 공간)
        self.vision_projection = nn.Linear(1024, 4096)  # CLIP → LLaMA

        # 3. Language Model
        self.llm = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

    def forward(self, images, text):
        # Step 1: 이미지 인코딩
        # (batch, 3, 224, 224) → (batch, 256, 1024)
        visual_features = self.vision_encoder(images).last_hidden_state

        # Step 2: Projection
        # (batch, 256, 1024) → (batch, 256, 4096)
        visual_embeddings = self.vision_projection(visual_features)

        # Step 3: 텍스트 임베딩
        text_embeddings = self.llm.get_input_embeddings()(text)

        # Step 4: Concatenate (의도: 이미지 토큰 + 텍스트 토큰)
        # [<img_token_1>, ..., <img_token_256>, "Describe", "the", "image"]
        combined_embeddings = torch.cat([visual_embeddings, text_embeddings], dim=1)

        # Step 5: LLM 추론
        outputs = self.llm(inputs_embeds=combined_embeddings)
        return outputs
```

### 2. GPT-4V 실전 사용

```python
import base64
from openai import OpenAI

class GPT4Vision:
    def __init__(self):
        self.client = OpenAI()

    def encode_image(self, image_path):
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def analyze_image(self, image_path, question):
        """
        GPT-4V로 이미지 분석

        사용 사례:
        - 문서 이해 (차트, 그래프, 표)
        - 의료 영상 분석
        - UI/UX 리뷰
        """
        base64_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"  # "low" | "high" | "auto"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )

        return response.choices[0].message.content

# 실전 예시 1: 차트 분석
analyzer = GPT4Vision()
result = analyzer.analyze_image(
    "sales_chart.png",
    "이 차트에서 어떤 트렌드를 볼 수 있나요? 수치를 정확히 읽어주세요."
)
print(result)
# Output: "2024년 Q1부터 Q3까지 매출이 지속적으로 증가했습니다.
#          Q1: $2.3M, Q2: $3.1M, Q3: $4.5M으로 94% 성장을 보였습니다..."

# 실전 예시 2: UI 리뷰
result = analyzer.analyze_image(
    "app_screenshot.png",
    "이 UI의 접근성 문제를 찾아주세요. WCAG 2.1 기준으로 평가해주세요."
)
```

### 3. Claude 3 Vision - 문서 이해 특화

```python
import anthropic

class Claude3Vision:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def analyze_document(self, image_path, task):
        """
        Claude 3의 강점: 정확한 OCR + 구조 이해

        사용 사례:
        - PDF 문서 파싱
        - 영수증 데이터 추출
        - 계약서 분석
        """
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        message = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": task
                        }
                    ],
                }
            ],
        )

        return message.content[0].text

# 실전 예시: 영수증 파싱
analyzer = Claude3Vision()
result = analyzer.analyze_document(
    "receipt.jpg",
    """
    다음 정보를 JSON으로 추출해주세요:
    {
        "merchant": "상호명",
        "date": "YYYY-MM-DD",
        "total": 123.45,
        "items": [
            {"name": "상품명", "price": 12.34, "quantity": 2}
        ],
        "payment_method": "결제 수단"
    }
    """
)

import json
data = json.loads(result)
# {"merchant": "Starbucks", "date": "2024-11-15", "total": 15.75, ...}
```

### 4. LLaVA - 오픈소스 대안

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch

class LLaVA:
    """
    LLaVA 1.6: 오픈소스 Vision-Language Model

    장점:
    - 무료, 자체 호스팅 가능
    - 세밀한 이미지 이해 (336px → 672px)
    - Mistral/Vicuna 기반으로 빠른 추론
    """
    def __init__(self, model_id="llava-hf/llava-v1.6-mistral-7b-hf"):
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, image_path, prompt):
        # 이미지 로드
        image = Image.open(image_path)

        # Conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # 프롬프트 생성
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # 처리
        inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt"
        ).to("cuda")

        # 생성
        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )

        result = self.processor.decode(output[0], skip_special_tokens=True)
        return result

# 실전 사용
llava = LLaVA()
response = llava.generate(
    "product.jpg",
    "이 제품의 특징을 설명하고, 유사 제품과 비교해주세요."
)
```

---

## Audio Processing

### 1. Whisper - Speech to Text

```python
import whisper
import torch

class AudioTranscriber:
    """
    Whisper: OpenAI의 강력한 음성 인식 모델

    특징:
    - 99개 언어 지원
    - 번역 기능 (다른 언어 → 영어)
    - Timestamp 추출
    """
    def __init__(self, model_size="large-v3"):
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path, language="ko"):
        """
        음성을 텍스트로 변환
        """
        result = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            verbose=False,
            word_timestamps=True  # 단어별 타임스탬프
        )

        return result

    def transcribe_with_diarization(self, audio_path):
        """
        화자 분리 (Speaker Diarization)
        의도: 회의록 생성, 팟캐스트 분석
        """
        # Step 1: Whisper로 전사
        result = self.model.transcribe(audio_path, word_timestamps=True)

        # Step 2: pyannote로 화자 분리
        from pyannote.audio import Pipeline
        diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        )

        diarization_result = diarization(audio_path)

        # Step 3: 결합
        segments = []
        for segment, _, speaker in diarization_result.itertracks(yield_label=True):
            # 해당 시간대의 텍스트 찾기
            words = [
                w for w in result['segments']
                if w['start'] >= segment.start and w['end'] <= segment.end
            ]
            text = " ".join([w['text'] for w in words])

            segments.append({
                'speaker': speaker,
                'start': segment.start,
                'end': segment.end,
                'text': text
            })

        return segments

# 실전 사용
transcriber = AudioTranscriber()

# 예시 1: 단순 전사
result = transcriber.transcribe("meeting.mp3", language="ko")
print(result['text'])

# 예시 2: 타임스탬프와 함께
for segment in result['segments']:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")

# 예시 3: 화자 분리
segments = transcriber.transcribe_with_diarization("meeting.mp3")
for seg in segments:
    print(f"{seg['speaker']}: {seg['text']}")
# Output:
# SPEAKER_00: 안녕하세요, 오늘 회의를 시작하겠습니다.
# SPEAKER_01: 네, 지난주 프로젝트 진행 상황을 공유하겠습니다.
```

### 2. Audio Generation - Text to Speech

```python
from TTS.api import TTS
import torch

class TextToSpeech:
    """
    Coqui TTS: 오픈소스 음성 합성

    특징:
    - 다국어 지원
    - Voice Cloning (3초 샘플로 음성 복제)
    - 감정 제어
    """
    def __init__(self):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    def generate_speech(self, text, output_path, language="ko"):
        """
        텍스트를 음성으로 변환
        """
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            language=language
        )

    def clone_voice(self, text, speaker_audio, output_path):
        """
        Voice Cloning
        의도: 특정 화자의 목소리로 TTS
        """
        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_audio,  # 3초 이상 샘플
            file_path=output_path,
            language="ko"
        )

# 실전 사용
tts = TextToSpeech()

# 예시 1: 기본 TTS
tts.generate_speech(
    "안녕하세요. AI 음성 합성 테스트입니다.",
    "output.wav",
    language="ko"
)

# 예시 2: Voice Cloning
tts.clone_voice(
    "이것은 복제된 목소리로 말하는 것입니다.",
    "speaker_sample.wav",  # 원본 화자 샘플
    "cloned_output.wav"
)
```

---

## Video Understanding

### 1. Gemini 1.5 Pro - 네이티브 비디오 처리

```python
import google.generativeai as genai
import time

class GeminiVideo:
    """
    Gemini 1.5 Pro: 네이티브 비디오 이해

    특징:
    - 1시간+ 비디오 직접 처리
    - 프레임 추출 불필요
    - 시간 기반 질문 가능
    """
    def __init__(self):
        genai.configure(api_key="YOUR_API_KEY")
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')

    def upload_video(self, video_path):
        """
        비디오 업로드 (최대 2시간)
        """
        video_file = genai.upload_file(path=video_path)

        # 처리 대기
        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("비디오 처리 실패")

        return video_file

    def analyze_video(self, video_file, prompt):
        """
        비디오 분석

        사용 사례:
        - 회의 요약
        - 튜토리얼 인덱싱
        - 스포츠 하이라이트 추출
        """
        response = self.model.generate_content([video_file, prompt])
        return response.text

# 실전 사용
analyzer = GeminiVideo()

# 예시 1: 회의 요약
video = analyzer.upload_video("meeting_recording.mp4")
summary = analyzer.analyze_video(
    video,
    """
    이 회의를 요약해주세요:
    1. 주요 논의 사항
    2. 결정된 액션 아이템
    3. 다음 회의 일정

    각 항목에 대해 타임스탬프를 포함해주세요.
    """
)

# 예시 2: 튜토리얼 인덱싱
video = analyzer.upload_video("tutorial.mp4")
chapters = analyzer.analyze_video(
    video,
    """
    이 튜토리얼의 챕터를 생성해주세요.
    각 챕터는 다음 형식으로:
    - [00:00] 챕터 제목: 간단한 설명
    """
)
print(chapters)
# Output:
# [00:00] 소개: 프로젝트 개요
# [02:30] 환경 설정: Docker와 의존성 설치
# [08:15] 코드 작성: 메인 로직 구현
# ...
```

### 2. VideoLLaMA - 오픈소스 대안

```python
from transformers import VideoLlamaForConditionalGeneration, VideoLlamaProcessor
import torch
import cv2

class VideoLLaMA:
    """
    VideoLLaMA: 오픈소스 비디오-언어 모델

    특징:
    - 비디오 + 오디오 이해
    - 자체 호스팅 가능
    """
    def __init__(self):
        self.processor = VideoLlamaProcessor.from_pretrained("DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned")
        self.model = VideoLlamaForConditionalGeneration.from_pretrained(
            "DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def extract_frames(self, video_path, num_frames=8):
        """
        비디오에서 프레임 추출
        의도: 균등하게 분산된 프레임 샘플링
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = torch.linspace(0, total_frames - 1, num_frames).long()
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        return frames

    def analyze(self, video_path, question):
        """
        비디오 분석
        """
        # 프레임 추출
        frames = self.extract_frames(video_path)

        # 처리
        inputs = self.processor(
            text=question,
            videos=frames,
            return_tensors="pt"
        ).to("cuda")

        # 생성
        output = self.model.generate(**inputs, max_new_tokens=256)
        answer = self.processor.decode(output[0], skip_special_tokens=True)

        return answer

# 실전 사용
video_llama = VideoLLaMA()
answer = video_llama.analyze(
    "cooking_video.mp4",
    "이 요리의 주요 단계를 순서대로 설명해주세요."
)
```

---

## 실전 구현 패턴

### 1. Document Intelligence System

```python
class DocumentIntelligence:
    """
    멀티모달 문서 이해 시스템

    기능:
    - PDF → 텍스트 + 이미지 추출
    - 표/차트 이해
    - 구조화된 데이터 추출
    """
    def __init__(self):
        self.vision_model = Claude3Vision()
        self.text_model = OpenAI()

    def process_pdf(self, pdf_path):
        """
        PDF 처리 파이프라인
        """
        from pdf2image import convert_from_path

        # Step 1: PDF → 이미지
        images = convert_from_path(pdf_path)

        # Step 2: 각 페이지 분석
        pages = []
        for i, img in enumerate(images):
            img_path = f"page_{i}.jpg"
            img.save(img_path)

            # Claude 3로 페이지 분석
            analysis = self.vision_model.analyze_document(
                img_path,
                """
                이 페이지를 분석해주세요:
                1. 텍스트 내용 (OCR)
                2. 표/차트가 있다면 데이터 추출
                3. 구조 (제목, 본문, 각주 등)

                JSON 형식으로 반환:
                {
                    "text": "...",
                    "tables": [...],
                    "charts": [...],
                    "structure": {...}
                }
                """
            )

            pages.append(json.loads(analysis))

        # Step 3: 전체 문서 요약
        full_text = "\n\n".join([p['text'] for p in pages])
        summary = self.text_model.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "당신은 문서 요약 전문가입니다."},
                {"role": "user", "content": f"다음 문서를 요약해주세요:\n\n{full_text}"}
            ]
        )

        return {
            'pages': pages,
            'summary': summary.choices[0].message.content
        }

# 실전 사용
doc_intel = DocumentIntelligence()
result = doc_intel.process_pdf("contract.pdf")

print("요약:", result['summary'])
for i, page in enumerate(result['pages']):
    print(f"\n페이지 {i+1}:")
    print("텍스트:", page['text'][:200])
    print("표 개수:", len(page['tables']))
```

### 2. Multimodal RAG

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import chromadb

class MultimodalRAG:
    """
    멀티모달 RAG: 텍스트 + 이미지 검색

    아키텍처:
    1. 텍스트: 일반 임베딩
    2. 이미지: CLIP 임베딩
    3. 통합 검색
    """
    def __init__(self):
        self.text_embeddings = OpenAIEmbeddings()
        self.image_embeddings = CLIPEmbeddings()

        # Vector stores
        self.text_store = Chroma(
            collection_name="text",
            embedding_function=self.text_embeddings
        )
        self.image_store = Chroma(
            collection_name="images",
            embedding_function=self.image_embeddings
        )

        self.vlm = GPT4Vision()

    def add_document(self, text, images):
        """
        문서 추가 (텍스트 + 이미지)
        """
        # 텍스트 저장
        self.text_store.add_texts([text])

        # 이미지 저장
        for img_path in images:
            # 이미지 캡션 생성 (검색 향상)
            caption = self.vlm.analyze_image(
                img_path,
                "이 이미지를 상세히 설명해주세요. 검색에 유용한 키워드를 포함해주세요."
            )

            self.image_store.add_texts(
                [caption],
                metadatas=[{"image_path": img_path}]
            )

    def search(self, query, modality="both"):
        """
        멀티모달 검색

        modality: "text" | "image" | "both"
        """
        results = []

        # 텍스트 검색
        if modality in ["text", "both"]:
            text_results = self.text_store.similarity_search(query, k=3)
            results.extend(text_results)

        # 이미지 검색
        if modality in ["image", "both"]:
            image_results = self.image_store.similarity_search(query, k=3)
            results.extend(image_results)

        # 재랭킹 (Cross-encoder 사용)
        results = self.rerank(query, results)

        return results

    def answer_with_context(self, query):
        """
        컨텍스트 기반 답변 생성
        """
        # Step 1: 관련 컨텍스트 검색
        context = self.search(query, modality="both")

        # Step 2: 텍스트 컨텍스트
        text_context = "\n\n".join([doc.page_content for doc in context if 'image_path' not in doc.metadata])

        # Step 3: 이미지 컨텍스트
        image_docs = [doc for doc in context if 'image_path' in doc.metadata]

        # Step 4: GPT-4V로 통합 답변
        if image_docs:
            # 이미지 있는 경우
            answer = self.vlm.analyze_image(
                image_docs[0].metadata['image_path'],
                f"""
                질문: {query}

                관련 텍스트:
                {text_context}

                이미지를 참고하여 답변해주세요.
                """
            )
        else:
            # 텍스트만 있는 경우
            answer = self.text_model.generate(
                f"질문: {query}\n\n컨텍스트:\n{text_context}\n\n답변:"
            )

        return answer

# 실전 사용
rag = MultimodalRAG()

# 문서 추가
rag.add_document(
    text="2024년 Q3 매출은 전년 대비 50% 증가했습니다...",
    images=["sales_chart.png", "product_image.jpg"]
)

# 검색
answer = rag.answer_with_context("2024년 매출 트렌드는 어떤가요?")
# GPT-4V가 차트 이미지를 보고 정확한 수치와 함께 답변
```

### 3. Video Content Moderation

```python
class VideoModerator:
    """
    비디오 콘텐츠 모더레이션

    기능:
    - 부적절한 콘텐츠 탐지
    - 타임스탬프 제공
    - 자동 블러 처리
    """
    def __init__(self):
        self.gemini = GeminiVideo()
        self.vision_model = GPT4Vision()

    def moderate_video(self, video_path):
        """
        비디오 모더레이션
        """
        # Step 1: Gemini로 전체 비디오 분석
        video_file = self.gemini.upload_video(video_path)

        analysis = self.gemini.analyze_video(
            video_file,
            """
            이 비디오에서 부적절한 콘텐츠를 찾아주세요:
            1. 폭력적인 장면
            2. 선정적인 콘텐츠
            3. 혐오 표현

            각 항목에 대해 타임스탬프와 심각도(low/medium/high)를 제공해주세요.

            JSON 형식:
            [
                {
                    "timestamp": "00:12",
                    "type": "violence",
                    "severity": "medium",
                    "description": "..."
                }
            ]
            """
        )

        violations = json.loads(analysis)

        # Step 2: 고위험 프레임 재검증
        high_risk = [v for v in violations if v['severity'] == 'high']

        for violation in high_risk:
            # 해당 타임스탬프의 프레임 추출
            frame = self.extract_frame(video_path, violation['timestamp'])

            # GPT-4V로 재검증
            verification = self.vision_model.analyze_image(
                frame,
                f"이 이미지에 {violation['type']} 콘텐츠가 있나요? 예/아니오로 답하고 이유를 설명해주세요."
            )

            violation['verification'] = verification

        return violations

    def extract_frame(self, video_path, timestamp):
        """
        특정 타임스탬프의 프레임 추출
        """
        import cv2

        # timestamp "00:12" → seconds
        parts = timestamp.split(":")
        seconds = int(parts[0]) * 60 + int(parts[1])

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = int(seconds * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        # 저장
        frame_path = f"frame_{timestamp.replace(':', '_')}.jpg"
        cv2.imwrite(frame_path, frame)
        return frame_path

# 실전 사용
moderator = VideoModerator()
violations = moderator.moderate_video("user_upload.mp4")

for v in violations:
    print(f"[{v['timestamp']}] {v['type']} ({v['severity']}): {v['description']}")
    if 'verification' in v:
        print(f"  검증: {v['verification']}")
```

---

## 프로덕션 배포

### 1. 비용 최적화 전략

```python
class MultimodalCostOptimizer:
    """
    멀티모달 API 비용 최적화

    전략:
    1. 이미지 해상도 조절
    2. 프레임 샘플링
    3. 캐싱
    4. 모델 선택
    """
    def __init__(self):
        self.cache = {}
        self.stats = {'total_cost': 0, 'cache_hits': 0}

    def optimize_image(self, image_path, detail_level="auto"):
        """
        이미지 최적화

        GPT-4V 비용:
        - detail="low": $0.00085/image (512x512)
        - detail="high": $0.00765/image (최대 2048x2048)
        - detail="auto": 자동 선택
        """
        from PIL import Image

        img = Image.open(image_path)
        width, height = img.size

        # 복잡도 평가
        if detail_level == "auto":
            # 작은 이미지 or 단순한 이미지 → low
            if width * height < 512 * 512:
                detail_level = "low"
            else:
                # 복잡도 계산 (edge detection)
                import cv2
                import numpy as np

                img_cv = cv2.imread(image_path)
                edges = cv2.Canny(img_cv, 100, 200)
                complexity = np.sum(edges) / (width * height)

                detail_level = "high" if complexity > 0.1 else "low"

        # 리사이즈
        if detail_level == "low":
            max_size = 512
        else:
            max_size = 2048

        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

            optimized_path = f"optimized_{image_path}"
            img.save(optimized_path, quality=85)
            return optimized_path, detail_level

        return image_path, detail_level

    def process_with_cache(self, image_path, prompt):
        """
        캐시 활용

        절감: 동일 이미지 재사용 시 100% 비용 절감
        """
        import hashlib

        # 캐시 키 생성
        with open(image_path, 'rb') as f:
            image_hash = hashlib.md5(f.read()).hexdigest()

        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        cache_key = f"{image_hash}_{prompt_hash}"

        # 캐시 확인
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]

        # 이미지 최적화
        optimized_path, detail = self.optimize_image(image_path)

        # API 호출
        vision = GPT4Vision()
        result = vision.analyze_image(optimized_path, prompt)

        # 비용 계산
        cost = 0.00085 if detail == "low" else 0.00765
        self.stats['total_cost'] += cost

        # 캐시 저장
        self.cache[cache_key] = result

        return result

# 실전 사용
optimizer = MultimodalCostOptimizer()

# Before: $0.00765/image
# After: $0.00085/image (캐시 히트 시 $0)

for img in ["product1.jpg", "product2.jpg", "product1.jpg"]:  # product1 반복
    result = optimizer.process_with_cache(img, "이 제품을 설명해주세요")

print(f"총 비용: ${optimizer.stats['total_cost']:.5f}")
print(f"캐시 히트: {optimizer.stats['cache_hits']}회")
# 총 비용: $0.00170 (3개 중 1개만 실제 처리)
# 캐시 히트: 1회
```

### 2. 성능 최적화

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class MultimodalPipeline:
    """
    멀티모달 파이프라인 최적화

    전략:
    1. 병렬 처리
    2. 배치 처리
    3. 스트리밍
    """
    def __init__(self):
        self.vision_model = GPT4Vision()
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def process_images_parallel(self, image_prompts):
        """
        이미지 병렬 처리

        Before: 10개 이미지 × 2초 = 20초
        After: 10개 이미지 / 10 workers = 2초
        """
        async def process_one(image_path, prompt):
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.vision_model.analyze_image,
                image_path,
                prompt
            )
            return result

        tasks = [
            process_one(img, prompt)
            for img, prompt in image_prompts
        ]

        results = await asyncio.gather(*tasks)
        return results

    async def process_video_streaming(self, video_path, frame_callback):
        """
        비디오 스트리밍 처리

        의도: 전체 비디오 처리 기다리지 않고 점진적 결과 반환
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 2)  # 2초마다 1 프레임

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                # 프레임 저장
                frame_path = f"temp_frame_{frame_num}.jpg"
                cv2.imwrite(frame_path, frame)

                # 분석 (비동기)
                result = await self.process_one(
                    frame_path,
                    "이 프레임에서 일어나는 일을 설명해주세요."
                )

                # 콜백 호출
                timestamp = frame_num / fps
                await frame_callback(timestamp, result)

            frame_num += 1

        cap.release()

# 실전 사용
pipeline = MultimodalPipeline()

# 예시 1: 병렬 처리
image_prompts = [
    ("image1.jpg", "이 제품을 설명해주세요"),
    ("image2.jpg", "이 제품을 설명해주세요"),
    # ... 10개
]

results = asyncio.run(pipeline.process_images_parallel(image_prompts))
# 2초 만에 10개 처리!

# 예시 2: 스트리밍
async def on_frame_analyzed(timestamp, analysis):
    print(f"[{timestamp:.2f}s] {analysis}")

asyncio.run(pipeline.process_video_streaming("video.mp4", on_frame_analyzed))
# 실시간으로 결과 출력
```

### 3. 모니터링 & 품질 관리

```python
from prometheus_client import Counter, Histogram
import logging

class MultimodalMonitoring:
    """
    멀티모달 시스템 모니터링

    메트릭:
    1. API 호출 횟수/비용
    2. 레이턴시
    3. 품질 (정확도)
    """
    def __init__(self):
        # Prometheus 메트릭
        self.request_count = Counter(
            'multimodal_requests_total',
            'Total multimodal API requests',
            ['model', 'modality']
        )

        self.request_latency = Histogram(
            'multimodal_request_duration_seconds',
            'Multimodal request latency',
            ['model', 'modality']
        )

        self.cost_counter = Counter(
            'multimodal_cost_dollars',
            'Total cost in dollars',
            ['model']
        )

    def track_request(self, model, modality, duration, cost):
        """
        요청 추적
        """
        self.request_count.labels(model=model, modality=modality).inc()
        self.request_latency.labels(model=model, modality=modality).observe(duration)
        self.cost_counter.labels(model=model).inc(cost)

    def quality_check(self, result, ground_truth=None):
        """
        품질 검증

        방법:
        1. Ground truth 비교 (있는 경우)
        2. LLM-as-a-judge
        3. 휴리스틱 검증
        """
        checks = []

        # Check 1: 길이 검증
        if len(result) < 10:
            checks.append({
                'check': 'length',
                'passed': False,
                'reason': '응답이 너무 짧음'
            })
        else:
            checks.append({'check': 'length', 'passed': True})

        # Check 2: Ground truth 비교
        if ground_truth:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, result, ground_truth).ratio()

            checks.append({
                'check': 'accuracy',
                'passed': similarity > 0.8,
                'similarity': similarity
            })

        # Check 3: LLM-as-a-judge
        judge_prompt = f"""
        다음 응답의 품질을 평가해주세요:

        {result}

        기준:
        1. 정확성: 사실에 기반한가?
        2. 완성도: 질문에 충분히 답했는가?
        3. 명확성: 이해하기 쉬운가?

        점수: 0-100
        """

        # 간단한 버전 (실제로는 LLM 호출)
        score = 85  # 예시

        checks.append({
            'check': 'quality',
            'passed': score >= 70,
            'score': score
        })

        return {
            'all_passed': all(c['passed'] for c in checks),
            'checks': checks
        }

# 실전 사용
monitoring = MultimodalMonitoring()

import time

start = time.time()
vision = GPT4Vision()
result = vision.analyze_image("test.jpg", "이 이미지를 설명해주세요")
duration = time.time() - start

# 메트릭 기록
monitoring.track_request(
    model="gpt-4-vision",
    modality="image",
    duration=duration,
    cost=0.00765
)

# 품질 검증
quality = monitoring.quality_check(result)
if not quality['all_passed']:
    logging.warning(f"품질 검증 실패: {quality['checks']}")
```

---

## 학습 로드맵

### 기초 (1-2주)
1. **Vision-Language Models 이해**
   - CLIP 아키텍처 학습
   - GPT-4V API 실습
   - 이미지 전처리 (리사이즈, 정규화)

2. **Audio 기초**
   - Whisper로 음성 전사
   - TTS 실습

### 중급 (2-3주)
3. **Multimodal RAG**
   - 텍스트 + 이미지 임베딩
   - 하이브리드 검색
   - 재랭킹

4. **Video Processing**
   - Gemini 1.5 Pro 실습
   - 프레임 샘플링 전략
   - 비디오 요약

### 고급 (3-4주)
5. **Production 시스템**
   - 비용 최적화
   - 병렬 처리
   - 모니터링

6. **Fine-tuning**
   - LLaVA fine-tuning
   - Custom vision encoder

---

## 핵심 요약

### 주요 모델
- **GPT-4V**: 최고 성능, 복잡한 시각 추론
- **Gemini 1.5 Pro**: 긴 비디오, 네이티브 멀티모달
- **Claude 3**: 정확한 문서/OCR
- **LLaVA**: 오픈소스 대안

### 실전 패턴
1. **Document Intelligence**: PDF → 구조화된 데이터
2. **Multimodal RAG**: 텍스트 + 이미지 검색
3. **Video Moderation**: 자동 콘텐츠 검토

### 비용 최적화
- 이미지 해상도 조절: 70% 절감
- 캐싱: 동일 입력 100% 절감
- 모델 선택: 복잡도 기반 라우팅

### 다음 단계
- [AI Agents](./06-ai-agents.md) - LangChain, AutoGPT
- [LLMOps](./04-llmops-production.md) - 프로덕션 배포
- [Research Trends](./02-research-trends-2024-2025.md) - 최신 연구

---

**작성일**: 2024-11-18
**업데이트**: Phase 6 - Current Trends
