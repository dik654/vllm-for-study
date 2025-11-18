# 문서 전처리 & 방법론 완벽 가이드

## 목차
1. [텍스트 정제 (Text Cleaning)](#텍스트-정제-text-cleaning)
2. [토큰화 (Tokenization)](#토큰화-tokenization)
3. [정규화 (Normalization)](#정규화-normalization)
4. [특수 문서 처리](#특수-문서-처리)
5. [임베딩 전처리](#임베딩-전처리)
6. [프로덕션 파이프라인](#프로덕션-파이프라인)

---

## 텍스트 정제 (Text Cleaning)

### 1. 기본 정제

```python
import re
import string
from typing import List, Dict

class TextCleaner:
    """
    텍스트 기본 정제

    목적: 노이즈 제거, 표준화
    """
    def __init__(self, config: Dict = None):
        self.config = config or {}

    def remove_html_tags(self, text: str) -> str:
        """
        HTML 태그 제거

        Before: "<p>Hello <b>World</b></p>"
        After: "Hello World"
        """
        clean = re.sub(r'<[^>]+>', '', text)
        return clean.strip()

    def remove_urls(self, text: str) -> str:
        """
        URL 제거

        Before: "Visit https://example.com for more"
        After: "Visit for more"
        """
        # http/https URL
        text = re.sub(r'https?://\S+', '', text)

        # www. URL
        text = re.sub(r'www\.\S+', '', text)

        return text.strip()

    def remove_emails(self, text: str) -> str:
        """
        이메일 주소 제거

        Before: "Contact me at john@example.com"
        After: "Contact me at"
        """
        return re.sub(r'\S+@\S+', '', text).strip()

    def remove_phone_numbers(self, text: str) -> str:
        """
        전화번호 제거 (한국/미국 형식)

        Before: "Call 010-1234-5678 or (555) 123-4567"
        After: "Call or"
        """
        # 한국 전화번호 (010-xxxx-xxxx, 02-xxx-xxxx)
        text = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '', text)

        # 미국 전화번호 ((555) 123-4567, 555-123-4567)
        text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)

        return text.strip()

    def remove_special_characters(self, text: str, keep: str = '') -> str:
        """
        특수 문자 제거

        keep: 유지할 문자 (예: '.,!?')

        Before: "Hello@World#2024!"
        After: "HelloWorld2024" (keep='') or "Hello World 2024!" (keep=' !')
        """
        # 유지할 문자를 제외한 모든 특수문자
        pattern = f'[^a-zA-Z0-9가-힣{re.escape(keep)}]'
        return re.sub(pattern, '', text).strip()

    def remove_extra_whitespace(self, text: str) -> str:
        """
        불필요한 공백 제거

        Before: "Hello    World  \n\n  !"
        After: "Hello World !"
        """
        # 연속 공백 → 단일 공백
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def remove_emojis(self, text: str) -> str:
        """
        이모지 제거

        Before: "I love Python 🐍❤️"
        After: "I love Python"
        """
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)

    def clean(self, text: str, pipeline: List[str] = None) -> str:
        """
        전체 파이프라인

        pipeline: 적용할 정제 단계 리스트
        """
        if pipeline is None:
            pipeline = [
                'remove_html_tags',
                'remove_urls',
                'remove_emails',
                'remove_phone_numbers',
                'remove_emojis',
                'remove_extra_whitespace'
            ]

        for step in pipeline:
            if hasattr(self, step):
                text = getattr(self, step)(text)

        return text

# 실전 사용
cleaner = TextCleaner()

# 예시 텍스트
raw_text = """
<div class="content">
    안녕하세요! 제 이메일은 john@example.com입니다.
    전화주세요: 010-1234-5678 🎉
    웹사이트: https://mysite.com


    Special@Characters#Here!!!
</div>
"""

cleaned = cleaner.clean(raw_text)
print("Cleaned Text:")
print(cleaned)

# Output:
# 안녕하세요 제 이메일은 입니다 전화주세요 웹사이트 SpecialCharactersHere

"""
팁:

1. 도메인별 커스터마이징:
   - 법률 문서: 조항 번호 유지
   - 의료 문서: 약물명, 용량 유지
   - SNS: 해시태그, 멘션 유지 or 별도 처리

2. 정보 손실 주의:
   - URL이 중요한 경우 (웹 문서): 유지 or <URL> 토큰으로 대체
   - 이모지가 감성 분석에 중요: 유지

3. 성능 최적화:
   - 정규식 컴파일 (re.compile) 미리 해두기
   - 큰 텍스트는 chunk 단위 처리
"""
```

---

## 토큰화 (Tokenization)

### 1. 다양한 토큰화 방법

```python
class Tokenizer:
    """
    토큰화 방법론

    1. Word-level (공백 기준)
    2. Subword (BPE, WordPiece, Unigram)
    3. Character-level
    4. Sentence-level
    """
    def __init__(self, method='subword'):
        self.method = method

    # ========== Word-level ==========
    def word_tokenize(self, text: str) -> List[str]:
        """
        단어 단위 토큰화

        장점: 간단, 빠름
        단점: OOV (Out-of-Vocabulary) 문제

        Before: "I love NLP!"
        After: ['I', 'love', 'NLP', '!']
        """
        import nltk
        from nltk.tokenize import word_tokenize

        # NLTK 데이터 다운로드 (최초 1회)
        # nltk.download('punkt')

        tokens = word_tokenize(text)
        return tokens

    def korean_word_tokenize(self, text: str) -> List[str]:
        """
        한국어 형태소 분석

        도구: KoNLPy (Okt, Mecab, Komoran, Hannanum, Kkma)

        Before: "나는 학교에 갑니다"
        After: ['나', '는', '학교', '에', '가', 'ㅂ니다'] (형태소)
        """
        from konlpy.tag import Okt

        okt = Okt()

        # 형태소 분석
        morphs = okt.morphs(text)
        return morphs

    def korean_pos_tag(self, text: str) -> List[tuple]:
        """
        품사 태깅

        Before: "아버지가 방에 들어가신다"
        After: [('아버지', 'Noun'), ('가', 'Josa'), ('방', 'Noun'), ...]
        """
        from konlpy.tag import Mecab

        mecab = Mecab()

        pos_tags = mecab.pos(text)
        return pos_tags

    # ========== Subword Tokenization ==========
    def subword_tokenize(self, text: str, model='bpe') -> List[str]:
        """
        Subword 토큰화

        장점: OOV 해결, 언어 독립적
        단점: 단어 의미 손실 가능

        방법:
        - BPE (Byte Pair Encoding): GPT
        - WordPiece: BERT
        - Unigram: T5, mBART

        Before: "unhappiness"
        After: ['un', 'happiness'] or ['un', 'happi', 'ness']
        """
        from transformers import AutoTokenizer

        # 사전 학습된 토크나이저
        if model == 'bpe':
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        elif model == 'wordpiece':
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif model == 'unigram':
            tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
        else:
            raise ValueError(f"Unknown model: {model}")

        tokens = tokenizer.tokenize(text)
        return tokens

    def train_custom_tokenizer(self, texts: List[str], vocab_size=10000):
        """
        커스텀 BPE 토크나이저 학습

        사용 사례: 도메인 특화 (의료, 법률)
        """
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # BPE 모델
        tokenizer = HFTokenizer(BPE(unk_token="<unk>"))

        # Pre-tokenizer (공백 기준 split)
        tokenizer.pre_tokenizer = Whitespace()

        # Trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]
        )

        # 학습
        tokenizer.train_from_iterator(texts, trainer=trainer)

        return tokenizer

    # ========== Sentence Tokenization ==========
    def sentence_tokenize(self, text: str) -> List[str]:
        """
        문장 단위 토큰화

        Before: "Hello World. How are you? I'm fine."
        After: ['Hello World.', 'How are you?', "I'm fine."]

        어려움: 약어, 소수점 구분
        """
        import nltk
        from nltk.tokenize import sent_tokenize

        # nltk.download('punkt')

        sentences = sent_tokenize(text)
        return sentences

    def korean_sentence_tokenize(self, text: str) -> List[str]:
        """
        한국어 문장 분리

        어려움: 마침표가 문장 끝이 아닌 경우
        """
        # kss (Korean Sentence Splitter)
        import kss

        sentences = kss.split_sentences(text)
        return sentences

# ========== 실전 예시 ==========

tokenizer = Tokenizer()

# 영어 텍스트
english_text = "I love natural language processing! It's amazing."

# Word tokenization
word_tokens = tokenizer.word_tokenize(english_text)
print(f"Word tokens: {word_tokens}")
# ['I', 'love', 'natural', 'language', 'processing', '!', 'It', "'s", 'amazing', '.']

# Subword tokenization (BPE)
subword_tokens = tokenizer.subword_tokenize(english_text, model='bpe')
print(f"BPE tokens: {subword_tokens}")
# ['I', 'Ġlove', 'Ġnatural', 'Ġlanguage', 'Ġprocessing', '!', 'ĠIt', "'s", 'Ġamazing', '.']
# (Ġ = space)

# 한국어 텍스트
korean_text = "자연어 처리는 정말 재미있습니다. 저는 매일 공부합니다."

# 형태소 분석
morphs = tokenizer.korean_word_tokenize(korean_text)
print(f"Korean morphs: {morphs}")
# ['자연어', '처리', '는', '정말', '재미', '있', '습니다', '.', '저', '는', '매일', '공부', '합니다', '.']

# 품사 태깅
pos_tags = tokenizer.korean_pos_tag(korean_text)
print(f"POS tags: {pos_tags}")
# [('자연어', 'NNG'), ('처리', 'NNG'), ('는', 'JX'), ...]

"""
토큰화 선택 가이드:

┌──────────────────────────────────────────────────────────┐
│ 작업           │ 추천 방법                  │ 이유     │
├──────────────────────────────────────────────────────────┤
│ LLM (GPT)      │ BPE                        │ OOV 해결 │
│ BERT 계열      │ WordPiece                  │ 표준     │
│ 다국어         │ Unigram (SentencePiece)    │ 유연성   │
│ 정보 추출      │ Word + POS tagging         │ 정확도   │
│ 감성 분석      │ Subword                    │ 균형     │
│ 기계 번역      │ Subword                    │ OOV 대응 │
│ 문서 분류      │ Word or Subword            │ 속도     │
└──────────────────────────────────────────────────────────┘

팁:

1. 한국어 형태소 분석기 비교:
   - Mecab: 가장 빠름, 높은 정확도
   - Okt (Twitter): 신조어, SNS 강함
   - Komoran: Java 기반, 안정적
   - Kkma: 상세한 분석, 느림

2. Subword 학습 시:
   - Vocab size: 너무 크면 메모리, 작으면 긴 토큰
   - 권장: 8K-32K (소규모), 50K-100K (대규모)

3. 특수 토큰:
   - <pad>: 패딩
   - <unk>: Unknown
   - <bos>, <eos>: 문장 시작/끝
   - <cls>, <sep>: BERT 특수 토큰
"""
```

---

## 정규화 (Normalization)

### 1. 텍스트 정규화

```python
class TextNormalizer:
    """
    텍스트 정규화

    목적: 다양한 표현을 표준 형태로 통일
    """
    def __init__(self):
        pass

    # ========== Case Normalization ==========
    def lowercase(self, text: str) -> str:
        """
        소문자 변환

        Before: "Hello WORLD"
        After: "hello world"

        주의: 고유명사 정보 손실
        """
        return text.lower()

    def titlecase(self, text: str) -> str:
        """
        제목 형식 (각 단어 첫 글자 대문자)

        Before: "hello world from PYTHON"
        After: "Hello World From Python"
        """
        return text.title()

    # ========== Unicode Normalization ==========
    def unicode_normalize(self, text: str, form='NFC') -> str:
        """
        유니코드 정규화

        형식:
        - NFC: Canonical Composition (한글 조합형)
        - NFD: Canonical Decomposition (한글 분해형)
        - NFKC: Compatibility Composition
        - NFKD: Compatibility Decomposition

        예: "가" = U+AC00 (NFC) vs U+1100 U+1161 (NFD)
        """
        import unicodedata

        return unicodedata.normalize(form, text)

    # ========== Spelling Correction ==========
    def correct_spelling(self, text: str, lang='en') -> str:
        """
        맞춤법 교정

        영어: TextBlob, autocorrect
        한국어: hanspell, py-hanspell
        """
        if lang == 'en':
            from textblob import TextBlob

            blob = TextBlob(text)
            corrected = blob.correct()
            return str(corrected)

        elif lang == 'ko':
            from hanspell import spell_checker

            spelled_sent = spell_checker.check(text)
            return spelled_sent.checked

    # ========== Contraction Expansion ==========
    def expand_contractions(self, text: str) -> str:
        """
        축약형 확장

        Before: "I'm not gonna do it"
        After: "I am not going to do it"
        """
        import contractions

        expanded = contractions.fix(text)
        return expanded

    # ========== Number Normalization ==========
    def normalize_numbers(self, text: str, target='word') -> str:
        """
        숫자 정규화

        target:
        - 'word': 숫자 → 단어 ("123" → "one hundred twenty three")
        - 'digit': 단어 → 숫자 ("twenty" → "20")
        - 'token': 숫자를 특수 토큰으로 ("<NUM>")
        """
        if target == 'word':
            from num2words import num2words

            # 숫자 찾아서 변환
            def replace_with_words(match):
                num = match.group()
                return num2words(int(num))

            return re.sub(r'\d+', replace_with_words, text)

        elif target == 'token':
            # 모든 숫자 → <NUM>
            return re.sub(r'\d+', '<NUM>', text)

        else:  # 'digit'
            # word2number 라이브러리 사용
            from word2number import w2n

            words = text.split()
            result = []

            for word in words:
                try:
                    num = w2n.word_to_num(word)
                    result.append(str(num))
                except ValueError:
                    result.append(word)

            return ' '.join(result)

    # ========== Korean Normalization ==========
    def normalize_korean(self, text: str) -> str:
        """
        한국어 특화 정규화

        - 반복 문자 정규화: "ㅋㅋㅋㅋㅋ" → "ㅋㅋ"
        - 이모티콘 정규화: "^^", "ㅠㅠ"
        """
        # 반복 문자 (3번 이상 → 2번)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # 한글 자음/모음 반복
        text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1+', r'\1', text)

        return text

    def remove_stopwords(self, tokens: List[str], lang='en') -> List[str]:
        """
        불용어 제거

        불용어: 의미 없는 흔한 단어 (the, is, a, ...)

        주의: 작업에 따라 중요할 수 있음
        (예: "not"은 감성 분석에 중요)
        """
        if lang == 'en':
            from nltk.corpus import stopwords

            # nltk.download('stopwords')

            stop_words = set(stopwords.words('english'))

        elif lang == 'ko':
            # 한국어 불용어 (커스텀 리스트)
            stop_words = set([
                '이', '그', '저', '것', '수', '등', '들',
                '및', '그리고', '또는', '하지만', '그러나'
            ])

        filtered = [token for token in tokens if token.lower() not in stop_words]
        return filtered

# ========== 실전 예시 ==========

normalizer = TextNormalizer()

# 예시 1: 영어 정규화
text = "I'm gonna visit the U.S.A. There're 123 people."

print("Original:", text)
print("Lowercase:", normalizer.lowercase(text))
print("Expanded:", normalizer.expand_contractions(text))
print("Numbers to words:", normalizer.normalize_numbers(text, target='word'))

# 출력:
# Original: I'm gonna visit the U.S.A. There're 123 people.
# Lowercase: i'm gonna visit the u.s.a. there're 123 people.
# Expanded: I am going to visit the U.S.A. There are 123 people.
# Numbers to words: I'm gonna visit the U.S.A. There're one hundred and twenty-three people.

# 예시 2: 한국어 정규화
korean_text = "와 진짜 대박이에요ㅋㅋㅋㅋㅋㅋ ㅠㅠㅠㅠ"

print("\nOriginal:", korean_text)
print("Normalized:", normalizer.normalize_korean(korean_text))

# 출력:
# Original: 와 진짜 대박이에요ㅋㅋㅋㅋㅋㅋ ㅠㅠㅠㅠ
# Normalized: 와 진짜 대박이에요ㅋㅋ ㅠ

"""
정규화 체크리스트:

[  ] 소문자 변환 (필요시)
[  ] 유니코드 정규화 (NFC)
[  ] 맞춤법 교정 (데이터 품질 낮을 때)
[  ] 축약형 확장 (I'm → I am)
[  ] 숫자 정규화 (작업에 따라)
[  ] 반복 문자 정규화 (SNS 데이터)
[  ] 불용어 제거 (작업에 따라)

정규화 순서 권장:

1. 유니코드 정규화
2. 맞춤법 교정
3. 축약형 확장
4. 소문자 변환
5. 반복 문자 정규화
6. 숫자 정규화
7. 불용어 제거 (토큰화 후)
"""
```

---

## 특수 문서 처리

### 1. PDF 문서 처리

```python
class PDFProcessor:
    """
    PDF 문서 처리

    도구:
    - PyPDF2: 텍스트 추출
    - pdfplumber: 표/이미지 추출
    - Camelot: 표 추출 (고급)
    """
    def __init__(self):
        pass

    def extract_text_simple(self, pdf_path: str) -> str:
        """
        간단한 텍스트 추출

        한계: 복잡한 레이아웃, 표, 이미지 처리 어려움
        """
        import PyPDF2

        text = ""

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page in pdf_reader.pages:
                text += page.extract_text()

        return text

    def extract_with_layout(self, pdf_path: str) -> List[Dict]:
        """
        레이아웃 보존 추출

        반환: 각 페이지의 구조화된 데이터
        """
        import pdfplumber

        pages_data = []

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # 텍스트
                text = page.extract_text()

                # 표
                tables = page.extract_tables()

                # 이미지 (좌표)
                images = page.images

                pages_data.append({
                    'page_num': i + 1,
                    'text': text,
                    'tables': tables,
                    'images': images
                })

        return pages_data

    def extract_tables(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        표 추출 (고급)

        Camelot: 더 정확한 표 인식
        """
        import camelot

        # 표 추출
        tables = camelot.read_pdf(pdf_path, pages='all')

        # DataFrame으로 변환
        dfs = [table.df for table in tables]

        return dfs

    def ocr_pdf(self, pdf_path: str) -> str:
        """
        OCR (이미지 기반 PDF)

        도구: Tesseract OCR
        """
        from pdf2image import convert_from_path
        import pytesseract

        # PDF → 이미지
        images = convert_from_path(pdf_path)

        text = ""

        for img in images:
            # OCR
            page_text = pytesseract.image_to_string(img, lang='kor+eng')
            text += page_text + "\n\n"

        return text

# 실전 사용
pdf_processor = PDFProcessor()

# 방법 1: 간단한 추출
# text = pdf_processor.extract_text_simple("document.pdf")

# 방법 2: 레이아웃 보존
# pages = pdf_processor.extract_with_layout("document.pdf")

# for page in pages:
#     print(f"Page {page['page_num']}:")
#     print(f"Text: {page['text'][:200]}...")
#     print(f"Tables: {len(page['tables'])} found")

# 방법 3: 표 추출
# tables = pdf_processor.extract_tables("document.pdf")
# for i, df in enumerate(tables):
#     print(f"\nTable {i+1}:")
#     print(df.head())

"""
PDF 처리 팁:

1. 텍스트 기반 vs 이미지 기반 구분:
   - 텍스트: PyPDF2, pdfplumber
   - 이미지 (스캔): OCR (Tesseract, Google Vision API)

2. 표 추출 정확도 향상:
   - Camelot: stream vs lattice 모드 선택
   - pdfplumber: table_settings 조정

3. 대용량 PDF:
   - 페이지별 처리 (메모리 절약)
   - 멀티프로세싱 활용

4. 품질 향상:
   - OCR 전 이미지 전처리 (노이즈 제거, 이진화)
   - Tesseract PSM (Page Segmentation Mode) 조정
"""
```

---

### 2. 웹 문서 (HTML) 처리

```python
from bs4 import BeautifulSoup
import requests

class HTMLProcessor:
    """
    HTML 문서 처리

    도구: BeautifulSoup, trafilatura
    """
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_html(self, url: str) -> str:
        """웹 페이지 가져오기"""
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.text

    def extract_main_content(self, html: str) -> str:
        """
        본문 추출 (광고, 네비게이션 제외)

        도구: trafilatura (뉴스 기사 등에 효과적)
        """
        import trafilatura

        # 본문만 추출
        content = trafilatura.extract(html)

        return content

    def extract_with_metadata(self, html: str) -> Dict:
        """
        메타데이터 포함 추출
        """
        import trafilatura

        # 메타데이터 포함
        result = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_images=True,
            with_metadata=True,
            output_format='json'
        )

        if result:
            import json
            return json.loads(result)
        else:
            return {}

    def custom_extraction(self, html: str, config: Dict) -> Dict:
        """
        커스텀 추출 (특정 사이트 구조에 맞춤)

        config: CSS 선택자 설정
        """
        soup = BeautifulSoup(html, 'html.parser')

        data = {}

        # 제목
        if 'title_selector' in config:
            title_elem = soup.select_one(config['title_selector'])
            data['title'] = title_elem.get_text().strip() if title_elem else ""

        # 본문
        if 'content_selector' in config:
            content_elems = soup.select(config['content_selector'])
            data['content'] = ' '.join([elem.get_text().strip() for elem in content_elems])

        # 작성자
        if 'author_selector' in config:
            author_elem = soup.select_one(config['author_selector'])
            data['author'] = author_elem.get_text().strip() if author_elem else ""

        # 날짜
        if 'date_selector' in config:
            date_elem = soup.select_one(config['date_selector'])
            data['date'] = date_elem.get_text().strip() if date_elem else ""

        return data

    def clean_html_text(self, text: str) -> str:
        """
        HTML에서 추출한 텍스트 정제
        """
        # 연속 줄바꿈 제거
        text = re.sub(r'\n+', '\n', text)

        # 연속 공백 제거
        text = re.sub(r' +', ' ', text)

        # 앞뒤 공백
        text = text.strip()

        return text

# 실전 사용
html_processor = HTMLProcessor()

# 예시: 뉴스 기사
url = "https://news.example.com/article/12345"

# HTML 가져오기
# html = html_processor.fetch_html(url)

# 본문 추출 (trafilatura - 자동)
# content = html_processor.extract_main_content(html)

# 메타데이터 포함
# data = html_processor.extract_with_metadata(html)
# print(f"Title: {data.get('title', 'N/A')}")
# print(f"Author: {data.get('author', 'N/A')}")
# print(f"Date: {data.get('date', 'N/A')}")
# print(f"Content: {data.get('text', '')[:500]}...")

# 커스텀 추출 (특정 사이트)
config = {
    'title_selector': 'h1.article-title',
    'content_selector': 'div.article-body p',
    'author_selector': 'span.author-name',
    'date_selector': 'time.publish-date'
}

# data = html_processor.custom_extraction(html, config)

"""
웹 스크래핑 팁:

1. robots.txt 준수:
   - https://example.com/robots.txt 확인
   - 크롤링 허용 여부, 주기 제한

2. Rate Limiting:
   - 요청 간 딜레이 (time.sleep)
   - 동시 요청 제한

3. 동적 페이지 (JavaScript):
   - Selenium or Playwright
   - API 직접 호출 (개발자 도구로 찾기)

4. 법적 이슈:
   - 저작권 주의
   - 개인정보 처리 주의
"""
```

---

## 임베딩 전처리

### 1. 텍스트 → 임베딩

```python
class EmbeddingPreprocessor:
    """
    임베딩을 위한 전처리

    목적: 임베딩 품질 향상
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.max_length = 512  # 모델별 다름

    def preprocess_for_embedding(self, text: str) -> str:
        """
        임베딩 최적화 전처리

        주의사항:
        - 너무 긴 텍스트: 잘라내기 or 청킹
        - 너무 짧은 텍스트: 컨텍스트 추가
        """
        # 기본 정제
        cleaner = TextCleaner()
        text = cleaner.clean(text)

        # 길이 제한
        tokens = text.split()
        if len(tokens) > self.max_length:
            # Truncate
            text = ' '.join(tokens[:self.max_length])

        return text

    def chunk_long_document(self, text: str, chunk_size=256, overlap=50) -> List[str]:
        """
        긴 문서를 청크로 분할

        chunk_size: 청크당 토큰 수
        overlap: 청크 간 겹침 (컨텍스트 유지)

        Before: "..." (10,000 토큰)
        After: ["chunk1", "chunk2", ..., "chunk39"]
        """
        tokens = text.split()

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + chunk_size
            chunk = ' '.join(tokens[start:end])
            chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    def semantic_chunking(self, text: str, sentences: List[str] = None) -> List[str]:
        """
        의미 기반 청킹

        방법: 문장 임베딩 유사도로 그룹화
        """
        if sentences is None:
            # 문장 분리
            tokenizer = Tokenizer()
            sentences = tokenizer.sentence_tokenize(text)

        # 문장별 임베딩
        embeddings = self.model.encode(sentences)

        # 유사도 기반 그룹화
        from sklearn.metrics.pairwise import cosine_similarity

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            # 현재 청크의 평균 임베딩
            chunk_emb = np.mean([embeddings[j] for j in range(i - len(current_chunk), i)], axis=0)

            # 다음 문장과의 유사도
            sim = cosine_similarity([chunk_emb], [embeddings[i]])[0][0]

            if sim > 0.7:  # 유사하면 같은 청크
                current_chunk.append(sentences[i])
            else:  # 다르면 새 청크
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]

        # 마지막 청크
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def embed_with_metadata(self, text: str, metadata: Dict) -> np.ndarray:
        """
        메타데이터를 활용한 임베딩

        예: "[TITLE] 제목 [CONTENT] 본문"
        """
        # 메타데이터를 텍스트에 통합
        augmented_text = ""

        if 'title' in metadata:
            augmented_text += f"[TITLE] {metadata['title']} "

        if 'category' in metadata:
            augmented_text += f"[CATEGORY] {metadata['category']} "

        augmented_text += f"[CONTENT] {text}"

        # 임베딩
        embedding = self.model.encode(augmented_text)

        return embedding

# 실전 사용
embedder = EmbeddingPreprocessor()

# 긴 문서
long_doc = "..." * 5000  # 매우 긴 텍스트

# 방법 1: 고정 크기 청킹
chunks_fixed = embedder.chunk_long_document(long_doc, chunk_size=256, overlap=50)
print(f"Fixed chunks: {len(chunks_fixed)}")

# 방법 2: 의미 기반 청킹
chunks_semantic = embedder.semantic_chunking(long_doc)
print(f"Semantic chunks: {len(chunks_semantic)}")

# 메타데이터 활용
text = "파이썬은 프로그래밍 언어입니다."
metadata = {
    'title': '파이썬 소개',
    'category': '프로그래밍'
}

embedding = embedder.embed_with_metadata(text, metadata)
print(f"Embedding shape: {embedding.shape}")

"""
임베딩 전처리 팁:

1. 청킹 전략:
   - 고정 크기: 간단, 빠름
   - 문장 단위: 의미 유지
   - 의미 기반: 최고 품질, 느림

2. 메타데이터 활용:
   - 제목, 카테고리 추가 → 검색 정확도 향상
   - 날짜, 저자 등도 필요시 추가

3. 모델 선택:
   - 영어: all-MiniLM-L6-v2 (빠름)
   - 다국어: paraphrase-multilingual-MiniLM-L12-v2
   - 고품질: all-mpnet-base-v2

4. Batch Processing:
   - 대량 문서는 배치로 임베딩 (속도 향상)
   - model.encode(texts, batch_size=32)
"""
```

---

## 프로덕션 파이프라인

### 1. 통합 전처리 파이프라인

```python
class ProductionPipeline:
    """
    프로덕션급 전처리 파이프라인

    특징:
    - 설정 기반 (YAML/JSON)
    - 로깅
    - 에러 핸들링
    - 성능 모니터링
    """
    def __init__(self, config_path: str):
        import yaml

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 컴포넌트 초기화
        self.cleaner = TextCleaner()
        self.normalizer = TextNormalizer()
        self.tokenizer = Tokenizer(method=self.config.get('tokenization', {}).get('method', 'subword'))

        # 로깅
        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process(self, text: str) -> Dict:
        """
        전체 파이프라인 실행

        반환: 처리 결과 + 메타데이터
        """
        import time

        start_time = time.time()

        result = {
            'original': text,
            'original_length': len(text)
        }

        try:
            # Step 1: Cleaning
            if self.config.get('cleaning', {}).get('enabled', True):
                text = self.cleaner.clean(
                    text,
                    pipeline=self.config['cleaning'].get('pipeline', None)
                )
                result['cleaned'] = text

                self.logger.info(f"Cleaning done: {len(text)} chars")

            # Step 2: Normalization
            if self.config.get('normalization', {}).get('enabled', True):
                norm_config = self.config['normalization']

                if norm_config.get('lowercase', False):
                    text = self.normalizer.lowercase(text)

                if norm_config.get('unicode_normalize', False):
                    text = self.normalizer.unicode_normalize(text)

                if norm_config.get('expand_contractions', False):
                    text = self.normalizer.expand_contractions(text)

                result['normalized'] = text

                self.logger.info(f"Normalization done")

            # Step 3: Tokenization
            if self.config.get('tokenization', {}).get('enabled', True):
                tokens = self.tokenizer.subword_tokenize(text)
                result['tokens'] = tokens
                result['num_tokens'] = len(tokens)

                self.logger.info(f"Tokenization done: {len(tokens)} tokens")

            # 처리 시간
            elapsed = time.time() - start_time
            result['processing_time_ms'] = elapsed * 1000

            result['status'] = 'success'

        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def batch_process(self, texts: List[str], n_jobs=4) -> List[Dict]:
        """
        배치 처리 (병렬)

        n_jobs: 병렬 작업 수
        """
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(tqdm(
                executor.map(self.process, texts),
                total=len(texts),
                desc="Processing"
            ))

        return results

# 설정 파일 예시 (config.yaml)
"""
cleaning:
  enabled: true
  pipeline:
    - remove_html_tags
    - remove_urls
    - remove_emails
    - remove_extra_whitespace

normalization:
  enabled: true
  lowercase: true
  unicode_normalize: true
  expand_contractions: true

tokenization:
  enabled: true
  method: subword  # word, subword, char
"""

# 실전 사용
# pipeline = ProductionPipeline('config.yaml')

# 단일 문서
# text = "<p>I'm gonna visit https://example.com</p>"
# result = pipeline.process(text)

# print(f"Status: {result['status']}")
# print(f"Tokens: {result['num_tokens']}")
# print(f"Processing time: {result['processing_time_ms']:.2f}ms")

# 배치 처리
# texts = [...]  # 1000개 문서
# results = pipeline.batch_process(texts, n_jobs=8)

"""
프로덕션 체크리스트:

[  ] 설정 파일로 관리 (YAML/JSON)
[  ] 로깅 (INFO, ERROR 레벨)
[  ] 에러 핸들링 (try-except)
[  ] 성능 모니터링 (처리 시간)
[  ] 배치 처리 (병렬화)
[  ] 테스트 (단위 테스트, 통합 테스트)
[  ] 문서화 (API 문서, 사용 예시)
[  ] 버전 관리 (설정, 모델 버전)

성능 최적화:

1. 병렬 처리:
   - ProcessPoolExecutor (CPU-bound)
   - ThreadPoolExecutor (I/O-bound)

2. 캐싱:
   - 동일 텍스트 반복 처리 시
   - functools.lru_cache

3. Lazy Loading:
   - 큰 모델은 필요할 때만 로드
   - 메모리 절약

4. 프로파일링:
   - cProfile로 병목 찾기
   - line_profiler로 라인별 분석
"""
```

---

## 핵심 요약

### 전처리 체크리스트

```python
"""
문서 전처리 단계별 체크리스트

1. 기본 정제
   [  ] HTML 태그 제거
   [  ] URL 제거/대체
   [  ] 이메일 제거
   [  ] 전화번호 제거
   [  ] 특수 문자 처리
   [  ] 불필요한 공백 제거

2. 정규화
   [  ] 대소문자 통일
   [  ] 유니코드 정규화 (NFC)
   [  ] 맞춤법 교정 (필요시)
   [  ] 축약형 확장
   [  ] 숫자 처리
   [  ] 반복 문자 정규화

3. 토큰화
   [  ] 방법 선택 (Word/Subword/Char)
   [  ] 언어별 처리 (한국어: 형태소 분석)
   [  ] 특수 토큰 추가

4. 추가 처리
   [  ] 불용어 제거 (작업에 따라)
   [  ] 어간 추출/표제어 추출
   [  ] 품사 태깅

5. 임베딩 최적화
   [  ] 길이 제한 확인
   [  ] 청킹 (필요시)
   [  ] 메타데이터 통합
"""
```

### 작업별 추천 파이프라인

| 작업 | 정제 | 정규화 | 토큰화 | 불용어 제거 |
|------|------|--------|--------|-------------|
| 감성 분석 | ✓ | 소문자 | Subword | ✗ (not 중요) |
| 문서 분류 | ✓ | 소문자 | Word/Subword | ✓ |
| NER | ✓ | ✗ (대문자 중요) | Subword | ✗ |
| 기계 번역 | ✓ | ✗ | Subword | ✗ |
| 검색 | ✓ | 소문자 | Subword | ✓ |
| 요약 | ✓ | ✗ | Subword | ✗ |
| 질의응답 | ✓ | ✗ | Subword | ✗ |

---

**작성일**: 2024-11-18
**업데이트**: Phase 6 - Current Trends
