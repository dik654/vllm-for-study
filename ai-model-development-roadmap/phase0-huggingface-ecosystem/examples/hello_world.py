"""
Hello World - HuggingFace 첫 실습
이 스크립트는 HuggingFace의 기본 기능을 테스트합니다.
"""

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import sys

def check_environment():
    """환경 설정 확인"""
    print("=" * 60)
    print("환경 확인 중...")
    print("=" * 60)

    # Python 버전
    print(f"Python 버전: {sys.version.split()[0]}")

    # PyTorch
    print(f"PyTorch 버전: {torch.__version__}")

    # CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 사용 가능: {cuda_available}")
    if cuda_available:
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print()

def test_model_loading():
    """모델 로딩 테스트"""
    print("=" * 60)
    print("1. 모델 로딩 테스트")
    print("=" * 60)

    model_name = "distilbert-base-uncased"
    print(f"모델 로딩 중: {model_name}")

    try:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 파라미터 수
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 모델 로딩 성공!")
        print(f"  총 파라미터: {total_params:,} ({total_params/1e6:.1f}M)")

        return model, tokenizer
    except Exception as e:
        print(f"✗ 모델 로딩 실패: {e}")
        return None, None

def test_tokenization(tokenizer):
    """토크나이제이션 테스트"""
    if tokenizer is None:
        return

    print("\n" + "=" * 60)
    print("2. 토크나이제이션 테스트")
    print("=" * 60)

    text = "Hello, I'm learning AI!"
    print(f"입력 텍스트: {text}")

    # 토큰화
    tokens = tokenizer.tokenize(text)
    print(f"토큰: {tokens}")

    # 인코딩
    encoded = tokenizer.encode(text, return_tensors="pt")
    print(f"인코딩된 ID: {encoded[0].tolist()}")

    # 디코딩
    decoded = tokenizer.decode(encoded[0])
    print(f"디코딩: {decoded}")

    print("✓ 토크나이제이션 성공!")

def test_inference(model, tokenizer):
    """추론 테스트"""
    if model is None or tokenizer is None:
        return

    print("\n" + "=" * 60)
    print("3. 모델 추론 테스트")
    print("=" * 60)

    text = "The quick brown fox jumps over the lazy dog."
    print(f"입력: {text}")

    # 인코딩
    inputs = tokenizer(text, return_tensors="pt")

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # 결과
    last_hidden_state = outputs.last_hidden_state
    print(f"출력 shape: {last_hidden_state.shape}")
    print(f"  - Batch size: {last_hidden_state.shape[0]}")
    print(f"  - Sequence length: {last_hidden_state.shape[1]}")
    print(f"  - Hidden size: {last_hidden_state.shape[2]}")

    print("✓ 추론 성공!")

def test_pipeline():
    """Pipeline 테스트"""
    print("\n" + "=" * 60)
    print("4. Pipeline 테스트 (감성 분석)")
    print("=" * 60)

    try:
        # 감성 분석 파이프라인
        classifier = pipeline("sentiment-analysis")

        texts = [
            "I love this tutorial!",
            "This is terrible.",
            "It's okay, nothing special."
        ]

        for text in texts:
            result = classifier(text)[0]
            emoji = "😊" if result['label'] == 'POSITIVE' else "😞"
            print(f"{emoji} {text}")
            print(f"   → {result['label']} (신뢰도: {result['score']:.3f})")

        print("✓ Pipeline 성공!")
    except Exception as e:
        print(f"✗ Pipeline 실패: {e}")

def test_model_save_load():
    """모델 저장 및 로딩 테스트"""
    print("\n" + "=" * 60)
    print("5. 모델 저장/로딩 테스트")
    print("=" * 60)

    try:
        # 작은 모델 로딩
        model_name = "distilbert-base-uncased"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 로컬에 저장
        save_path = "./test_model"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"✓ 모델 저장 완료: {save_path}")

        # 다시 로딩
        loaded_model = AutoModel.from_pretrained(save_path)
        loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)
        print(f"✓ 모델 로딩 완료: {save_path}")

        # 정리
        import shutil
        shutil.rmtree(save_path)
        print("✓ 테스트 파일 정리 완료")

    except Exception as e:
        print(f"✗ 저장/로딩 실패: {e}")

def main():
    """메인 함수"""
    print("\n" + "🎉" * 30)
    print("HuggingFace Hello World")
    print("🎉" * 30 + "\n")

    # 1. 환경 확인
    check_environment()

    # 2. 모델 로딩
    model, tokenizer = test_model_loading()

    # 3. 토크나이제이션
    test_tokenization(tokenizer)

    # 4. 추론
    test_inference(model, tokenizer)

    # 5. Pipeline
    test_pipeline()

    # 6. 저장/로딩
    test_model_save_load()

    # 완료
    print("\n" + "=" * 60)
    print("✅ 모든 테스트 완료!")
    print("=" * 60)
    print("\n축하합니다! HuggingFace 환경이 올바르게 설정되었습니다. 🚀")
    print("이제 본격적인 학습을 시작할 수 있습니다!\n")
    print("다음 단계: phase0-huggingface-ecosystem/01-core-terminology.md")
    print()

if __name__ == "__main__":
    main()
