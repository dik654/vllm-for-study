# Day 3-4: 모델 허브 구조 이해

## 📋 목차
1. [config.json - 모델 아키텍처 설정](#1-configjson)
2. [tokenizer_config.json - 토크나이저 설정](#2-tokenizer_configjson)
3. [generation_config.json - 생성 설정](#3-generation_configjson)
4. [Sharded 모델 이해](#4-sharded-models)

---

## 1. config.json - 모델 아키텍처 설정

### 📖 이론

`config.json`은 모델의 아키텍처와 하이퍼파라미터를 정의합니다.

### 💻 실습 1: config.json 분석

```python
# examples/analyze_config.py
from transformers import AutoConfig
import json

def analyze_model_config(model_name):
    """
    모델의 config.json 상세 분석
    """
    print(f"=== {model_name} Config 분석 ===\n")

    # Config 로딩
    config = AutoConfig.from_pretrained(model_name)

    # JSON으로 출력
    config_dict = config.to_dict()
    print(json.dumps(config_dict, indent=2))

    # 주요 파라미터 설명
    print("\n=== 주요 파라미터 설명 ===\n")

    if hasattr(config, 'hidden_size'):
        print(f"hidden_size: {config.hidden_size}")
        print("  → 각 토큰의 임베딩 차원")

    if hasattr(config, 'num_hidden_layers'):
        print(f"\nnum_hidden_layers: {config.num_hidden_layers}")
        print("  → Transformer 블록 개수")

    if hasattr(config, 'num_attention_heads'):
        print(f"\nnum_attention_heads: {config.num_attention_heads}")
        print("  → Multi-head attention의 head 개수")
        print(f"  → Head당 차원: {config.hidden_size // config.num_attention_heads}")

    if hasattr(config, 'intermediate_size'):
        print(f"\nintermediate_size: {config.intermediate_size}")
        print(f"  → FFN 중간 레이어 크기")
        print(f"  → 확장 비율: {config.intermediate_size / config.hidden_size:.1f}x")

    if hasattr(config, 'max_position_embeddings'):
        print(f"\nmax_position_embeddings: {config.max_position_embeddings}")
        print("  → 최대 시퀀스 길이")

    if hasattr(config, 'vocab_size'):
        print(f"\nvocab_size: {config.vocab_size}")
        print("  → 어휘 크기")

    # 전체 파라미터 수 계산
    if hasattr(config, 'hidden_size') and hasattr(config, 'num_hidden_layers'):
        # 간단한 추정 (정확하지 않지만 대략적)
        embedding_params = config.vocab_size * config.hidden_size
        layer_params = config.num_hidden_layers * (
            4 * config.hidden_size ** 2 +  # Attention Q,K,V,O
            2 * config.hidden_size * config.intermediate_size  # FFN
        )
        total_params = embedding_params + layer_params
        print(f"\n대략적인 파라미터 수: {total_params / 1e6:.1f}M")

if __name__ == "__main__":
    models = [
        "bert-base-uncased",
        "gpt2",
        "t5-small",
    ]

    for model in models:
        analyze_model_config(model)
        print("\n" + "="*80 + "\n")
```

### 💻 실습 2: Custom Config 생성

```python
# examples/create_custom_config.py
from transformers import BertConfig, BertModel
import torch

def create_custom_bert():
    """
    커스텀 BERT 모델 생성
    """
    # 커스텀 설정
    config = BertConfig(
        vocab_size=30000,          # 어휘 크기
        hidden_size=512,           # 임베딩 차원 (BERT-base는 768)
        num_hidden_layers=6,       # 레이어 수 (BERT-base는 12)
        num_attention_heads=8,     # Attention heads (512/8=64 per head)
        intermediate_size=2048,    # FFN 중간 크기 (512*4)
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    print("=== Custom BERT Config ===")
    print(config)

    # 모델 생성
    model = BertModel(config)

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n총 파라미터: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"훈련 가능 파라미터: {trainable_params:,}")

    # 테스트
    input_ids = torch.randint(0, config.vocab_size, (2, 10))  # (batch, seq_len)
    outputs = model(input_ids)

    print(f"\n입력 shape: {input_ids.shape}")
    print(f"출력 shape: {outputs.last_hidden_state.shape}")

    # Config 저장
    config.save_pretrained("./custom_bert_config")
    model.save_pretrained("./custom_bert_model")
    print("\nConfig와 모델 저장 완료!")

    return model, config

if __name__ == "__main__":
    model, config = create_custom_bert()
```

---

## 2. tokenizer_config.json - 토크나이저 설정

### 📖 이론

`tokenizer_config.json`은 토크나이저의 동작 방식을 정의합니다:
- Special tokens ([CLS], [SEP], [PAD], etc.)
- Padding/Truncation 전략
- 토크나이제이션 알고리즘 파라미터

### 💻 실습 3: Tokenizer Config 분석

```python
# examples/analyze_tokenizer_config.py
from transformers import AutoTokenizer
import json

def analyze_tokenizer(model_name):
    """
    토크나이저 설정 상세 분석
    """
    print(f"=== {model_name} Tokenizer 분석 ===\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Special tokens
    print("=== Special Tokens ===")
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")

    # Vocab 정보
    print(f"\n=== Vocabulary ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Model max length: {tokenizer.model_max_length}")

    # 토크나이제이션 예제
    print(f"\n=== Tokenization 예제 ===")
    text = "Hello, how are you?"

    # Basic tokenization
    tokens = tokenizer.tokenize(text)
    print(f"텍스트: {text}")
    print(f"토큰: {tokens}")

    # Encoding
    encoded = tokenizer.encode(text, add_special_tokens=True)
    print(f"인코딩 (special tokens 포함): {encoded}")

    # Decoding
    decoded = tokenizer.decode(encoded)
    print(f"디코딩: {decoded}")

    # Batch encoding with padding
    print(f"\n=== Batch Encoding ===")
    texts = ["Short text", "This is a longer text with more tokens"]

    # Padding to longest
    encoded_batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    print("Input IDs:")
    print(encoded_batch['input_ids'])
    print("\nAttention Mask:")
    print(encoded_batch['attention_mask'])

    # Config 저장
    tokenizer.save_pretrained("./custom_tokenizer")
    print("\n토크나이저 저장 완료!")

if __name__ == "__main__":
    models = ["bert-base-uncased", "gpt2", "t5-small"]

    for model in models:
        analyze_tokenizer(model)
        print("\n" + "="*80 + "\n")
```

### 💻 실습 4: Custom Tokenizer 훈련

```python
# examples/train_custom_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

def train_custom_tokenizer(texts, vocab_size=1000):
    """
    커스텀 BPE 토크나이저 훈련
    """
    # BPE 토크나이저 초기화
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Whitespace로 pre-tokenization
    tokenizer.pre_tokenizer = Whitespace()

    # Trainer 설정
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True,
    )

    # 텍스트 파일에 저장
    with open("temp_corpus.txt", "w") as f:
        for text in texts:
            f.write(text + "\n")

    # 훈련
    print("토크나이저 훈련 중...")
    tokenizer.train(["temp_corpus.txt"], trainer)

    # Post-processing (BERT 스타일)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # HuggingFace Tokenizer로 래핑
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    # 테스트
    print("\n=== 토크나이저 테스트 ===")
    test_text = "This is a test sentence."
    encoded = wrapped_tokenizer.encode(test_text)
    tokens = wrapped_tokenizer.tokenize(test_text)

    print(f"텍스트: {test_text}")
    print(f"토큰: {tokens}")
    print(f"인코딩: {encoded}")
    print(f"디코딩: {wrapped_tokenizer.decode(encoded)}")

    # 저장
    wrapped_tokenizer.save_pretrained("./my_custom_tokenizer")
    print("\n커스텀 토크나이저 저장 완료!")

    return wrapped_tokenizer

if __name__ == "__main__":
    # 샘플 코퍼스
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of data.",
    ] * 100  # 반복하여 더 큰 코퍼스 생성

    tokenizer = train_custom_tokenizer(corpus, vocab_size=500)
```

---

## 3. generation_config.json - 생성 설정

### 📖 이론

`generation_config.json`은 텍스트 생성 시 사용되는 디폴트 파라미터를 정의합니다:
- **max_length**: 최대 생성 길이
- **temperature**: 샘플링 온도 (높을수록 다양)
- **top_k/top_p**: 샘플링 전략
- **num_beams**: Beam search beams 수

### 💻 실습 5: Generation Strategies 비교

```python
# examples/generation_strategies.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compare_generation_strategies(model_name="gpt2"):
    """
    다양한 생성 전략 비교
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"Prompt: {prompt}\n")
    print("="*80 + "\n")

    # 1. Greedy Decoding
    print("=== 1. Greedy Decoding ===")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,  # Greedy
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\n" + "="*80 + "\n")

    # 2. Beam Search
    print("=== 2. Beam Search (num_beams=5) ===")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\n" + "="*80 + "\n")

    # 3. Sampling (Temperature)
    for temp in [0.5, 1.0, 1.5]:
        print(f"=== 3. Sampling (Temperature={temp}) ===")
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id,
        )
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print("\n" + "="*80 + "\n")

    # 4. Top-k Sampling
    print("=== 4. Top-k Sampling (k=50) ===")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\n" + "="*80 + "\n")

    # 5. Nucleus (Top-p) Sampling
    print("=== 5. Nucleus Sampling (p=0.9) ===")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\n" + "="*80 + "\n")

    # 6. 조합 (Top-k + Top-p + Temperature)
    print("=== 6. 조합 (top_k=50, top_p=0.95, temp=0.7) ===")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    compare_generation_strategies()
```

---

## 4. Sharded Models - 대형 모델 로딩

### 📖 이론

큰 모델은 여러 파일로 분할(shard)되어 저장됩니다:
- `model-00001-of-00005.safetensors`
- `model-00002-of-00005.safetensors`
- ...
- `model.safetensors.index.json` (메타데이터)

### 💻 실습 6: Sharded Model 로딩

```python
# examples/load_sharded_model.py
import json
from transformers import AutoModel, AutoModelForCausalLM
import torch

def analyze_sharded_model(model_name):
    """
    Sharded 모델 구조 분석
    """
    print(f"=== {model_name} Sharding 분석 ===\n")

    # Index 파일 읽기 (로컬에 있다면)
    try:
        with open(f"{model_name}/model.safetensors.index.json") as f:
            index = json.load(f)

        print("=== Index 파일 내용 ===")
        print(json.dumps(index, indent=2))

        # 각 shard의 크기 정보
        print("\n=== Shard별 레이어 분포 ===")
        shard_layers = {}
        for layer_name, shard_file in index["weight_map"].items():
            if shard_file not in shard_layers:
                shard_layers[shard_file] = []
            shard_layers[shard_file].append(layer_name)

        for shard, layers in shard_layers.items():
            print(f"\n{shard}: {len(layers)} layers")
            print(f"  첫 레이어: {layers[0]}")
            print(f"  마지막 레이어: {layers[-1]}")

    except FileNotFoundError:
        print("로컬에 모델이 없습니다. 다운로드 중...")

        # 모델 로딩 (자동으로 sharding 처리)
        model = AutoModel.from_pretrained(model_name)

        print(f"\n모델 로딩 완료!")
        print(f"총 파라미터: {sum(p.numel() for p in model.parameters()):,}")

def demonstrate_device_map():
    """
    device_map을 사용한 멀티 GPU 로딩
    """
    model_name = "gpt2"  # 작은 모델로 데모

    print("=== Device Map 데모 ===\n")

    # 자동 device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # 자동으로 GPU에 분산
    )

    # 각 레이어의 디바이스 확인
    print("레이어별 디바이스 할당:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")
        if name.count('.') < 2:  # 주요 레이어만 출력
            continue
        break

    # Custom device map
    if torch.cuda.device_count() > 1:
        device_map = {
            "transformer.wte": 0,
            "transformer.wpe": 0,
            "transformer.h.0": 0,
            "transformer.h.1": 0,
            "transformer.h.2": 1,
            "transformer.h.3": 1,
            # ... 나머지는 자동
        }

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
        )
        print("\nCustom device map 적용 완료!")

if __name__ == "__main__":
    # 큰 모델로 테스트 (예: LLaMA-7B)
    # analyze_sharded_model("meta-llama/Llama-2-7b-hf")

    # 작은 모델로 데모
    demonstrate_device_map()
```

---

## ✅ Day 3-4 완료 체크리스트

### 이론 이해
- [ ] config.json의 모든 파라미터 의미 이해
- [ ] Tokenizer config와 special tokens 이해
- [ ] Generation strategies 차이 명확히 구분
- [ ] Model sharding의 필요성과 동작 원리 이해

### 실습 완료
- [ ] 여러 모델의 config 분석
- [ ] Custom config로 모델 생성
- [ ] Tokenizer config 분석 및 커스텀 훈련
- [ ] 다양한 generation 전략 비교
- [ ] Sharded model 로딩 및 device map 사용

### 심화 과제
- [ ] 자신만의 모델 아키텍처 config 설계
- [ ] 특정 도메인을 위한 커스텀 토크나이저 훈련
- [ ] Generation 전략별 품질 평가 실험

---

## 📚 추가 학습 자료

### 문서
- [HuggingFace Model Configuration](https://huggingface.co/docs/transformers/main_classes/configuration)
- [Tokenizer Summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)

### 블로그
- [How to generate text with Transformers](https://huggingface.co/blog/how-to-generate)
