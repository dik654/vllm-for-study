# 전통적 NLP/ML 이론 및 MCP 구현 가이드

이 문서는 전통적인 NLP/ML 기법들을 처음부터 끝까지 상세히 설명하고, 이를 MCP 서버로 구현하는 방법을 다룹니다.

## 📐 수학적 배경지식

### 1. 선형대수 (Linear Algebra) 기초

#### 1.1 벡터 (Vector)

**직관적 이해**: 벡터는 크기와 방향을 가진 화살표입니다. 문서를 벡터로 표현하면 수학적 연산이 가능해집니다.

```python
"""
예시: "고양이가 물고기를 먹었다" 문서를 벡터로 표현

단어 사전: ["고양이", "물고기", "먹다", "강아지", "뛰다"]
문서 벡터: [1, 1, 1, 0, 0]
           ↑  ↑  ↑  ↑  ↑
         고양이 물고기 먹다 강아지 뛰다

각 숫자는 해당 단어의 출현 횟수를 의미합니다.
"""

import numpy as np

# 벡터 생성
doc1 = np.array([1, 1, 1, 0, 0])  # "고양이가 물고기를 먹었다"
doc2 = np.array([1, 0, 0, 1, 1])  # "고양이가 뛰었다"

print(f"문서1: {doc1}")
print(f"문서2: {doc2}")
```

**수학적 정의**:
```
벡터 v ∈ ℝⁿ (n차원 실수 공간)
v = [v₁, v₂, ..., vₙ]

예: v = [2, 3, 1] ∈ ℝ³
```

#### 1.2 벡터 내적 (Dot Product)

**직관적 이해**: 두 벡터가 얼마나 비슷한 방향을 가리키는지 측정합니다.

```python
# 내적 계산
dot_product = np.dot(doc1, doc2)
print(f"내적: {dot_product}")  # 1 (공통 단어 "고양이" 1개)
```

**수학적 정의**:
```
u · v = u₁v₁ + u₂v₂ + ... + uₙvₙ
      = ||u|| ||v|| cos(θ)
```

#### 1.3 코사인 유사도 (Cosine Similarity)

**직관적 이해**: 두 문서가 얼마나 비슷한지 0~1 사이로 측정합니다.

```python
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sim = cosine_similarity(doc1, doc2)
print(f"코사인 유사도: {sim:.4f}")
```

#### 1.4 행렬 (Matrix)

**직관적 이해**: 여러 문서를 한 번에 표현하는 표.

```python
# 문서-단어 행렬
doc_term_matrix = np.array([
    [1, 1, 1, 0, 0],  # 문서1
    [1, 0, 0, 1, 1],  # 문서2
    [0, 1, 1, 0, 0]   # 문서3
])
```

#### 1.5 특이값 분해 (SVD)

**직관적 이해**: 복잡한 행렬을 간단한 3개 행렬로 분해합니다. LSA의 핵심입니다.

```python
# SVD 수행
U, S, VT = np.linalg.svd(doc_term_matrix, full_matrices=False)

# 차원 축소 (상위 2개 토픽만)
k = 2
approx = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
```

**수학적 정의**:
```
A = UΣV^T

- U: 문서-토픽 행렬
- Σ: 특이값 (중요도)
- V^T: 토픽-단어 행렬
```

### 2. 확률과 통계 기초

#### 2.1 확률 분포

```python
from collections import Counter

text = "AI is powerful. AI helps people. AI is the future."
words = text.lower().replace(".", "").split()
word_counts = Counter(words)
total_words = len(words)

# 확률 계산
for word, count in word_counts.most_common():
    prob = count / total_words
    print(f"P({word}) = {prob:.3f}")
```

#### 2.2 조건부 확률

**베이즈 정리**: P(A|B) = P(B|A)P(A) / P(B)

#### 2.3 엔트로피 (Entropy)

**직관적 이해**: 불확실성의 정도. 높을수록 예측하기 어렵습니다.

```python
def entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# 균등 분포 vs 편향 분포
H_uniform = entropy([0.25, 0.25, 0.25, 0.25])  # 높은 엔트로피
H_skewed = entropy([0.7, 0.1, 0.1, 0.1])       # 낮은 엔트로피
```

**수학적 정의**:
```
H(X) = -Σ p(x) log₂ p(x)
```

#### 2.4 상호 정보량 (Mutual Information)

**직관적 이해**: 두 변수가 공유하는 정보량.

```
MI(X; Y) = H(X) + H(Y) - H(X, Y)
```

### 3. 정보이론 기초

#### 3.1 점별 상호 정보량 (PMI: Pointwise Mutual Information)

**직관적 이해**: 두 단어가 우연히 함께 나타날 확률보다 얼마나 더 자주 함께 나타나는지 측정합니다.

```python
def pmi(word1, word2, corpus):
    """
    PMI(w1, w2) = log[P(w1, w2) / (P(w1) × P(w2))]

    > 0: 함께 자주 나타남
    = 0: 독립적
    < 0: 회피 관계
    """
    # P(w1, w2): 동시 출현 확률
    p_w1_w2 = count_cooccurrence(word1, word2, corpus) / len(corpus)

    # P(w1), P(w2): 개별 출현 확률
    p_w1 = count_word(word1, corpus) / len(corpus)
    p_w2 = count_word(word2, corpus) / len(corpus)

    return np.log2(p_w1_w2 / (p_w1 * p_w2))
```

**수학적 정의**:
```
PMI(x, y) = log[P(x,y) / (P(x)P(y))]

PPMI (Positive PMI):
PPMI(x, y) = max(0, PMI(x, y))

활용:
- 단어 연관성 측정
- 동시 출현 행렬 가중치
- Word2Vec SGNS와 관계
```

---

## 📚 전통적 NLP/ML 개념 상세 정리

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)

#### 개념 설명

**직관적 이해**:
- TF (Term Frequency): 문서 내 단어 빈도
- IDF (Inverse Document Frequency): 여러 문서에 나타나는 단어는 중요도가 낮음
- TF-IDF = TF × IDF: 빈번하지만 특별한 단어에 높은 점수

**예시**:
```
문서1: "고양이가 물고기를 먹었다"
문서2: "강아지가 고양이를 쫓았다"
문서3: "물고기가 물속에 있다"

"고양이"는 2개 문서에 등장 → IDF 중간
"물고기"도 2개 문서에 등장 → IDF 중간
"먹었다"는 1개 문서에만 등장 → IDF 높음 (중요!)
```

#### 수학적 정의

```
TF(t, d) = (단어 t가 문서 d에 등장한 횟수) / (문서 d의 총 단어 수)

IDF(t) = log(전체 문서 수 / 단어 t가 등장한 문서 수)

TF-IDF(t, d) = TF(t, d) × IDF(t)
```

#### Python 구현 (from scratch)

```python
import numpy as np
from collections import Counter
import math

class TFIDFVectorizer:
    """TF-IDF 벡터화기 (처음부터 구현)"""

    def __init__(self):
        self.vocabulary = {}  # 단어 → 인덱스
        self.idf = {}  # 단어 → IDF 값
        self.doc_count = 0

    def fit(self, documents):
        """문서 집합에서 IDF 계산"""
        self.doc_count = len(documents)

        # 1. 어휘 사전 구축
        all_words = set()
        for doc in documents:
            words = doc.lower().split()
            all_words.update(words)

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}

        # 2. 각 단어가 등장한 문서 수 계산
        doc_freq = Counter()
        for doc in documents:
            words = set(doc.lower().split())
            for word in words:
                doc_freq[word] += 1

        # 3. IDF 계산
        for word in self.vocabulary:
            self.idf[word] = math.log(self.doc_count / doc_freq[word])

    def transform(self, documents):
        """문서들을 TF-IDF 벡터로 변환"""
        vectors = []

        for doc in documents:
            # TF 계산
            words = doc.lower().split()
            word_count = Counter(words)
            total_words = len(words)

            # TF-IDF 벡터 생성
            vector = np.zeros(len(self.vocabulary))
            for word, count in word_count.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    tf = count / total_words
                    vector[idx] = tf * self.idf[word]

            vectors.append(vector)

        return np.array(vectors)

    def fit_transform(self, documents):
        """fit + transform"""
        self.fit(documents)
        return self.transform(documents)

# 사용 예시
documents = [
    "고양이가 물고기를 먹었다",
    "강아지가 고양이를 쫓았다",
    "물고기가 물속에 있다"
]

vectorizer = TFIDFVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("TF-IDF 행렬:")
print(tfidf_matrix)
print(f"\n어휘: {list(vectorizer.vocabulary.keys())}")
print(f"IDF 값: {vectorizer.idf}")

# 문서 유사도 계산
from scipy.spatial.distance import cosine
sim_01 = 1 - cosine(tfidf_matrix[0], tfidf_matrix[1])
print(f"\n문서0과 문서1 유사도: {sim_01:.4f}")
```

#### scikit-learn 사용

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 간단한 사용
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)

print("단어:", vectorizer.get_feature_names_out())
print("TF-IDF:\n", tfidf.toarray())
```

#### 활용 사례

1. **문서 검색**: 쿼리와 문서 간 유사도 계산
2. **키워드 추출**: 높은 TF-IDF 값을 가진 단어 = 중요 키워드
3. **문서 분류**: TF-IDF 벡터를 피처로 사용

### 2. LSA (Latent Semantic Analysis) - 잠재 의미 분석

#### 개념 설명

**직관적 이해**:
- 문서와 단어 사이의 "숨겨진 주제(토픽)"을 찾아냅니다
- SVD를 사용하여 차원을 축소하고 노이즈를 제거합니다
- 유의어 문제 해결: "자동차"와 "차"를 같은 토픽으로 인식

**예시**:
```
원본 문서-단어 행렬:
        자동차  차  운전  고양이  개
문서1    3    0    1     0     0    → 교통 토픽
문서2    0    3    1     0     0    → 교통 토픽
문서3    0    0    0     3     2    → 동물 토픽

LSA 후:
        토픽1(교통)  토픽2(동물)
문서1      0.9          0.1
문서2      0.9          0.1
문서3      0.1          0.9
```

#### 수학적 정의

```
1. 문서-단어 행렬 A 생성 (m × n)
2. SVD 수행: A = UΣV^T
3. 상위 k개 특이값만 선택 (차원 축소)
4. A_k = U_k Σ_k V_k^T (근사)

여기서:
- U_k: 문서-토픽 행렬 (m × k)
- Σ_k: 토픽 중요도 (k × k 대각행렬)
- V_k^T: 토픽-단어 행렬 (k × n)
```

#### Python 구현

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

class LSAModel:
    """LSA 모델 구현"""

    def __init__(self, n_topics=2):
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer()
        self.svd = TruncatedSVD(n_components=n_topics)

    def fit_transform(self, documents):
        """문서를 LSA 공간으로 변환"""
        # 1. TF-IDF 벡터화
        tfidf_matrix = self.vectorizer.fit_transform(documents)

        # 2. SVD 수행 (차원 축소)
        doc_topic_matrix = self.svd.fit_transform(tfidf_matrix)

        return doc_topic_matrix

    def get_topics(self, n_words=5):
        """각 토픽의 주요 단어 추출"""
        words = self.vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(self.svd.components_):
            top_word_indices = topic.argsort()[-n_words:][::-1]
            top_words = [words[i] for i in top_word_indices]
            topics.append(top_words)

        return topics

# 사용 예시
documents = [
    "자동차 운전 교통 도로",
    "차량 주행 고속도로",
    "고양이 개 애완동물",
    "강아지 고양이 반려동물",
    "자동차 엔진 정비",
]

lsa = LSAModel(n_topics=2)
doc_topic = lsa.fit_transform(documents)

print("문서-토픽 행렬:")
print(doc_topic)

print("\n각 토픽의 주요 단어:")
for i, words in enumerate(lsa.get_topics()):
    print(f"토픽 {i}: {words}")

# 문서 유사도 (LSA 공간에서)
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(doc_topic)
print("\n문서 유사도 행렬 (LSA 공간):")
print(similarities)
```

#### 장단점

**장점**:
- 유의어 문제 해결
- 차원 축소로 계산 효율 증가
- 노이즈 제거

**단점**:
- 해석이 어려움 (토픽이 무엇을 의미하는지 불명확)
- 음수 값 가능 (확률이 아님)
- 새 문서 추가 시 재계산 필요

### 3. Word Embeddings (단어 임베딩)

#### 개념 설명

**직관적 이해**:
- 단어를 고차원 벡터 공간에 매핑
- 의미가 비슷한 단어는 가까운 위치에
- 벡터 연산 가능: king - man + woman ≈ queen

**시각화 (2차원으로 축소)**:
```
  고양이 •

  개 •
         •강아지

         • 자동차
    • 차
```

#### 3.1 Word2Vec

**두 가지 방식**:

**CBOW (Continuous Bag of Words)**:
- 주변 단어(context)로 중심 단어(target) 예측
- 빠른 학습

```
입력: "고양이가 [?] 먹었다"
출력: "물고기를"
```

**Skip-gram**:
- 중심 단어로 주변 단어 예측
- 더 좋은 임베딩 (특히 희귀 단어)

```
입력: "물고기를"
출력: "고양이가", "먹었다"
```

#### 수학적 정의

```
목적 함수 (Skip-gram):
maximize Σ_t Σ_{-c≤j≤c, j≠0} log P(w_{t+j} | w_t)

P(w_O | w_I) = exp(v'_{w_O}^T v_{w_I}) / Σ_w exp(v'_w^T v_{w_I})

여기서:
- w_I: 입력 단어
- w_O: 출력 (문맥) 단어
- v_w, v'_w: 단어 벡터
```

#### Python 구현 (Gensim 사용)

```python
from gensim.models import Word2Vec

# 예시 문장들
sentences = [
    ["고양이", "물고기", "먹다"],
    ["강아지", "고양이", "놀다"],
    ["물고기", "물", "헤엄치다"],
    ["고양이", "쥐", "잡다"],
    ["개", "강아지", "짖다"]
]

# Word2Vec 학습
model = Word2Vec(
    sentences,
    vector_size=100,  # 임베딩 차원
    window=2,         # 문맥 윈도우 크기
    min_count=1,      # 최소 빈도
    sg=1              # Skip-gram (0=CBOW)
)

# 단어 벡터 얻기
vec_cat = model.wv['고양이']
print(f"'고양이' 벡터 (일부): {vec_cat[:5]}")

# 유사 단어 찾기
similar = model.wv.most_similar('고양이', topn=3)
print(f"\n'고양이'와 유사한 단어: {similar}")

# 벡터 연산
result = model.wv.most_similar(
    positive=['강아지', '짖다'],
    negative=['고양이'],
    topn=1
)
print(f"\n강아지:짖다 = 고양이:? → {result}")

# 단어 간 유사도
similarity = model.wv.similarity('고양이', '강아지')
print(f"\n'고양이'와 '강아지' 유사도: {similarity:.4f}")
```

#### 3.2 GloVe (Global Vectors)

**개념**:
- Word2Vec와 달리 전체 말뭉치 통계 활용
- 동시 출현 행렬(Co-occurrence Matrix) 사용

```python
# GloVe 사전학습 모델 사용
from gensim.models import KeyedVectors

# GloVe 파일 로드 (사전 다운로드 필요)
glove_file = 'glove.6B.100d.txt'
model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

# 사용법은 Word2Vec과 동일
```

#### 3.3 FastText

**개념**:
- 단어를 character n-gram으로 분해
- OOV (Out-of-Vocabulary) 문제 해결
- 형태가 비슷한 단어도 유사하게 임베딩

```python
from gensim.models import FastText

# FastText 학습
model = FastText(
    sentences,
    vector_size=100,
    window=2,
    min_count=1
)

# OOV 단어도 벡터 생성 가능
unknown_word = "고양이새끼"  # 학습 데이터에 없음
vec = model.wv[unknown_word]  # 그래도 벡터 생성!
print(f"'{unknown_word}' 벡터: {vec[:5]}")
```

#### 3.4 최신 임베딩: BERT, GPT 등

**Contextualized Embeddings**:
- 문맥에 따라 다른 임베딩
- "사과(과일)" vs "사과(apple 기업)" → 다른 벡터

```python
from transformers import BertTokenizer, BertModel
import torch

# BERT 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

def get_bert_embedding(text):
    """BERT 임베딩 추출"""
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] 토큰의 임베딩 사용
    return outputs.last_hidden_state[0][0].numpy()

# 문맥에 따라 다른 임베딩
emb1 = get_bert_embedding("나는 사과를 먹었다")  # 과일
emb2 = get_bert_embedding("사과는 위대한 회사다")  # 기업

print(f"임베딩 차원: {len(emb1)}")
```

### 4. Topic Modeling (토픽 모델링)

#### 4.1 LDA (Latent Dirichlet Allocation)

**직관적 이해**:
- 문서는 여러 토픽의 혼합
- 각 토픽은 단어 분포
- 확률 모델 (음수 없음)

**예시**:
```
문서1: "자동차 엔진 수리" = 0.9×토픽(자동차) + 0.1×토픽(기타)
문서2: "고양이 개 동물" = 0.9×토픽(동물) + 0.1×토픽(기타)

토픽(자동차) = {자동차:0.3, 엔진:0.25, 수리:0.2, ...}
토픽(동물) = {고양이:0.3, 개:0.25, 동물:0.2, ...}
```

#### 수학적 정의

```
생성 과정:
1. 각 문서 d에 대해 토픽 분포 θ_d ~ Dir(α) 샘플링
2. 각 토픽 k에 대해 단어 분포 φ_k ~ Dir(β) 샘플링
3. 문서 d의 각 단어 n에 대해:
   a. 토픽 z_{d,n} ~ Multinomial(θ_d) 선택
   b. 단어 w_{d,n} ~ Multinomial(φ_{z_{d,n}}) 선택

여기서:
- α: 문서-토픽 디리클레 분포 파라미터
- β: 토픽-단어 디리클레 분포 파라미터
```

#### Python 구현

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "자동차 엔진 수리 정비",
    "차량 브레이크 교체",
    "고양이 개 애완동물",
    "강아지 사료 간식",
    "자동차 타이어 교환"
]

# 1. 단어 빈도 벡터화
vectorizer = CountVectorizer()
doc_term_matrix = vectorizer.fit_transform(documents)

# 2. LDA 모델 학습
lda = LatentDirichletAllocation(
    n_components=2,  # 토픽 개수
    random_state=42
)
doc_topic_dist = lda.fit_transform(doc_term_matrix)

# 3. 결과 분석
print("문서-토픽 분포:")
print(doc_topic_dist)

# 각 토픽의 주요 단어
words = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-5:][::-1]
    top_words = [words[i] for i in top_words_idx]
    print(f"\n토픽 {topic_idx}: {top_words}")
    print(f"단어 확률: {topic[top_words_idx]}")
```

#### 4.2 NMF (Non-negative Matrix Factorization)

**개념**:
- 행렬을 두 개의 음수가 아닌 행렬로 분해
- LDA보다 해석하기 쉬움
- 더 명확한 토픽 분리

```
A ≈ WH

여기서:
- A: 문서-단어 행렬 (m × n)
- W: 문서-토픽 행렬 (m × k), W ≥ 0
- H: 토픽-단어 행렬 (k × n), H ≥ 0
```

```python
from sklearn.decomposition import NMF

# TF-IDF 사용 (NMF는 보통 TF-IDF와 함께)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
tfidf = vectorizer.fit_transform(documents)

# NMF 학습
nmf = NMF(n_components=2, random_state=42)
doc_topic = nmf.fit_transform(tfidf)
topic_word = nmf.components_

print("NMF 문서-토픽 분포:")
print(doc_topic)

# 토픽 주요 단어
words = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(topic_word):
    top_words_idx = topic.argsort()[-5:][::-1]
    top_words = [words[i] for i in top_words_idx]
    print(f"\n토픽 {topic_idx}: {top_words}")
```

### 5. Co-occurrence Matrix (동시 출현 행렬)

**개념**: 단어들이 함께 나타나는 빈도를 행렬로 표현합니다.

```python
from collections import defaultdict
import numpy as np

def build_cooccurrence_matrix(sentences, window_size=2):
    """동시 출현 행렬 생성"""
    # 어휘 사전
    vocab = set()
    for sent in sentences:
        vocab.update(sent)
    vocab = sorted(vocab)
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    # 동시 출현 행렬 초기화
    matrix = np.zeros((len(vocab), len(vocab)))

    # 동시 출현 카운트
    for sent in sentences:
        for i, word in enumerate(sent):
            # 윈도우 내 단어들
            start = max(0, i - window_size)
            end = min(len(sent), i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    matrix[word2idx[word]][word2idx[sent[j]]] += 1

    return matrix, vocab

# 사용 예시
sentences = [
    ["고양이", "물고기", "먹다"],
    ["강아지", "고양이", "놀다"],
    ["물고기", "물", "헤엄치다"]
]

cooc_matrix, vocab = build_cooccurrence_matrix(sentences, window_size=1)

print("동시 출현 행렬:")
print(cooc_matrix)
print(f"\n어휘: {vocab}")

# PMI로 변환
def pmi_matrix(cooc_matrix):
    """동시 출현 행렬을 PMI 행렬로 변환"""
    total = cooc_matrix.sum()
    row_sums = cooc_matrix.sum(axis=1, keepdims=True)
    col_sums = cooc_matrix.sum(axis=0, keepdims=True)

    expected = (row_sums @ col_sums) / total

    pmi = np.log2(cooc_matrix / expected)
    pmi[pmi < 0] = 0  # PPMI: 음수는 0으로
    pmi[np.isinf(pmi)] = 0  # log(0) 처리

    return pmi

ppmi = pmi_matrix(cooc_matrix)
print("\nPPMI 행렬:")
print(ppmi)
```

### 6. N-gram 모델

**개념**: 연속된 N개 단어의 시퀀스를 분석합니다.

```python
from collections import Counter, defaultdict

def extract_ngrams(text, n=2):
    """N-gram 추출"""
    words = text.split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return ngrams

# 예시
text = "고양이가 물고기를 먹었다 강아지가 고양이를 쫓았다"

unigrams = extract_ngrams(text, n=1)
bigrams = extract_ngrams(text, n=2)
trigrams = extract_ngrams(text, n=3)

print("Bigrams:", bigrams)
print("Trigrams:", trigrams)

# N-gram 언어 모델
class NgramLanguageModel:
    """N-gram 확률 언어 모델"""

    def __init__(self, n=2):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()

    def train(self, texts):
        """학습"""
        for text in texts:
            words = ['<s>'] * (self.n - 1) + text.split() + ['</s>']

            for i in range(len(words) - self.n + 1):
                ngram = tuple(words[i:i+self.n])
                context = ngram[:-1]

                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def prob(self, word, context):
        """P(word | context) 계산"""
        ngram = context + (word,)
        if self.context_counts[context] == 0:
            return 0
        return self.ngram_counts[ngram] / self.context_counts[context]

    def generate(self, context, n_words=5):
        """다음 단어 생성"""
        result = list(context)

        for _ in range(n_words):
            # 가능한 다음 단어들과 확률
            candidates = {}
            for ngram in self.ngram_counts:
                if ngram[:-1] == tuple(result[-(self.n-1):]):
                    next_word = ngram[-1]
                    candidates[next_word] = self.prob(next_word, ngram[:-1])

            if not candidates:
                break

            # 가장 높은 확률의 단어 선택
            next_word = max(candidates, key=candidates.get)
            result.append(next_word)

        return ' '.join(result)

# 사용 예시
texts = [
    "고양이가 물고기를 먹었다",
    "강아지가 고양이를 쫓았다",
    "물고기가 물속에서 헤엄친다"
]

model = NgramLanguageModel(n=2)
model.train(texts)

# 확률 계산
prob = model.prob("먹었다", ("물고기를",))
print(f"\nP(먹었다 | 물고기를) = {prob:.3f}")

# 문장 생성
generated = model.generate(("고양이가",), n_words=3)
print(f"생성된 문장: {generated}")
```

---

## 🔧 MCP 서버 구현 - 전통적 NLP/ML 기법

이 섹션에서는 위에서 학습한 전통적 NLP/ML 기법들을 실제 MCP 서버로 구현하는 방법을 다룹니다.

### MCP 서버 1: Text Analysis Server (TF-IDF 기반 텍스트 분석)

#### 기능
- TF-IDF 벡터화
- 문서 유사도 계산
- 키워드 추출
- 문서 검색

#### 구현

```python
# text_analysis_mcp_server.py
import asyncio
import json
from typing import List, Dict, Any
from mcp.server.models import InitializationOptions
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text-analysis-server")

# MCP 서버 생성
app = Server("text-analysis-server")

# 전역 상태: 문서 저장소
document_store: Dict[str, Any] = {
    "documents": [],
    "vectorizer": None,
    "tfidf_matrix": None
}

@app.list_tools()
async def list_tools() -> List[Tool]:
    """사용 가능한 도구 목록"""
    return [
        Tool(
            name="add_documents",
            description="문서를 컬렉션에 추가하고 TF-IDF 인덱스를 생성합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "추가할 문서 목록"
                    }
                },
                "required": ["documents"]
            }
        ),
        Tool(
            name="search_documents",
            description="쿼리와 유사한 문서를 검색합니다 (TF-IDF + 코사인 유사도)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "반환할 문서 개수",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="extract_keywords",
            description="문서에서 중요 키워드를 TF-IDF 기반으로 추출합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "키워드를 추출할 텍스트"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "추출할 키워드 개수",
                        "default": 10
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="document_similarity",
            description="두 문서 간의 유사도를 계산합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc1": {"type": "string"},
                    "doc2": {"type": "string"}
                },
                "required": ["doc1", "doc2"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """도구 실행"""
    try:
        if name == "add_documents":
            return await add_documents(arguments["documents"])

        elif name == "search_documents":
            top_k = arguments.get("top_k", 5)
            return await search_documents(arguments["query"], top_k)

        elif name == "extract_keywords":
            top_k = arguments.get("top_k", 10)
            return await extract_keywords(arguments["text"], top_k)

        elif name == "document_similarity":
            return await document_similarity(arguments["doc1"], arguments["doc2"])

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error in {name}: {str(e)}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def add_documents(documents: List[str]) -> List[TextContent]:
    """문서 추가 및 TF-IDF 인덱스 생성"""
    document_store["documents"].extend(documents)

    # TF-IDF 벡터화
    document_store["vectorizer"] = TfidfVectorizer()
    document_store["tfidf_matrix"] = document_store["vectorizer"].fit_transform(
        document_store["documents"]
    )

    result = {
        "status": "success",
        "total_documents": len(document_store["documents"]),
        "vocabulary_size": len(document_store["vectorizer"].vocabulary_)
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def search_documents(query: str, top_k: int = 5) -> List[TextContent]:
    """문서 검색"""
    if document_store["vectorizer"] is None:
        return [TextContent(type="text", text="Error: 문서를 먼저 추가해주세요")]

    # 쿼리 벡터화
    query_vector = document_store["vectorizer"].transform([query])

    # 코사인 유사도 계산
    similarities = cosine_similarity(query_vector, document_store["tfidf_matrix"])[0]

    # 상위 k개 문서 추출
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "rank": len(results) + 1,
            "document": document_store["documents"][idx],
            "similarity": float(similarities[idx])
        })

    return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]

async def extract_keywords(text: str, top_k: int = 10) -> List[TextContent]:
    """키워드 추출"""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text])

    # TF-IDF 점수가 높은 단어 추출
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf.toarray()[0]

    top_indices = tfidf_scores.argsort()[-top_k:][::-1]

    keywords = [
        {
            "keyword": feature_names[idx],
            "tfidf_score": float(tfidf_scores[idx])
        }
        for idx in top_indices
    ]

    return [TextContent(type="text", text=json.dumps(keywords, ensure_ascii=False, indent=2))]

async def document_similarity(doc1: str, doc2: str) -> List[TextContent]:
    """문서 유사도 계산"""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([doc1, doc2])

    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    result = {
        "doc1_preview": doc1[:100] + "..." if len(doc1) > 100 else doc1,
        "doc2_preview": doc2[:100] + "..." if len(doc2) > 100 else doc2,
        "cosine_similarity": float(similarity),
        "interpretation": "매우 유사" if similarity > 0.7 else "유사" if similarity > 0.4 else "다소 유사" if similarity > 0.2 else "거의 다름"
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def main():
    """서버 실행"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="text-analysis-server",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
```

#### 사용 예시 (Claude Agent와 통합)

```python
# agent_example_tfidf.py
"""
AI 에이전트가 Text Analysis MCP 서버를 활용하는 예시
"""

# MCP 서버 설정 (~/.config/claude/mcp_config.json)
"""
{
  "mcpServers": {
    "text-analysis": {
      "command": "python",
      "args": ["/path/to/text_analysis_mcp_server.py"]
    }
  }
}
"""

# 사용 예시
"""
User: 내 블로그 포스트 데이터베이스에서 "머신러닝 최적화"와 관련된 글을 찾아줘

Agent: Text Analysis MCP 서버를 사용하겠습니다.

[MCP Tool Call: add_documents]
{
  "documents": [
    "머신러닝 모델 최적화 기법: 하이퍼파라미터 튜닝과 정규화",
    "딥러닝 성능 향상을 위한 배치 정규화 방법",
    "강화학습 알고리즘의 수렴 속도 개선",
    "자연어처리에서의 전이학습 활용법",
    "경사하강법 최적화: Adam, RMSprop 비교"
  ]
}

[MCP Tool Call: search_documents]
{
  "query": "머신러닝 최적화",
  "top_k": 3
}

Agent: 다음 3개의 글이 가장 관련성이 높습니다:

1. "머신러닝 모델 최적화 기법: 하이퍼파라미터 튜닝과 정규화" (유사도: 0.89)
2. "경사하강법 최적화: Adam, RMSprop 비교" (유사도: 0.72)
3. "딥러닝 성능 향상을 위한 배치 정규화 방법" (유사도: 0.54)

---

User: 이 논문의 핵심 키워드를 추출해줘: [논문 텍스트]

Agent:
[MCP Tool Call: extract_keywords]
{
  "text": "[논문 전체 텍스트]",
  "top_k": 10
}

결과:
1. neural_network (TF-IDF: 0.45)
2. optimization (TF-IDF: 0.38)
3. gradient_descent (TF-IDF: 0.32)
...
"""
```

### MCP 서버 2: Semantic Analysis Server (LSA 기반 의미 분석)

#### 기능
- LSA 기반 문서 클러스터링
- 토픽 발견 및 해석
- 의미적 문서 검색
- 차원 축소 및 시각화

#### 구현

```python
# semantic_analysis_mcp_server.py
import asyncio
import json
from typing import List, Dict, Any
from mcp.server.models import InitializationOptions
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic-analysis-server")

app = Server("semantic-analysis-server")

# 전역 상태
semantic_store: Dict[str, Any] = {
    "documents": [],
    "vectorizer": None,
    "lsa_model": None,
    "doc_topic_matrix": None,
    "n_topics": 5
}

@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="build_lsa_model",
            description="문서 컬렉션에서 LSA 모델을 구축합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "분석할 문서 목록"
                    },
                    "n_topics": {
                        "type": "integer",
                        "description": "추출할 토픽 개수",
                        "default": 5
                    }
                },
                "required": ["documents"]
            }
        ),
        Tool(
            name="get_topics",
            description="LSA로 발견된 토픽과 각 토픽의 주요 단어를 반환합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "n_words": {
                        "type": "integer",
                        "description": "토픽당 표시할 단어 수",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="semantic_search",
            description="LSA 공간에서 의미적으로 유사한 문서를 검색합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="cluster_documents",
            description="LSA 공간에서 문서를 클러스터링합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "n_clusters": {
                        "type": "integer",
                        "description": "클러스터 개수",
                        "default": 3
                    }
                }
            }
        ),
        Tool(
            name="document_topic_distribution",
            description="특정 문서의 토픽 분포를 반환합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_index": {
                        "type": "integer",
                        "description": "문서 인덱스"
                    }
                },
                "required": ["document_index"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    try:
        if name == "build_lsa_model":
            n_topics = arguments.get("n_topics", 5)
            return await build_lsa_model(arguments["documents"], n_topics)

        elif name == "get_topics":
            n_words = arguments.get("n_words", 10)
            return await get_topics(n_words)

        elif name == "semantic_search":
            top_k = arguments.get("top_k", 5)
            return await semantic_search(arguments["query"], top_k)

        elif name == "cluster_documents":
            n_clusters = arguments.get("n_clusters", 3)
            return await cluster_documents(n_clusters)

        elif name == "document_topic_distribution":
            return await document_topic_distribution(arguments["document_index"])

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error in {name}: {str(e)}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def build_lsa_model(documents: List[str], n_topics: int) -> List[TextContent]:
    """LSA 모델 구축"""
    semantic_store["documents"] = documents
    semantic_store["n_topics"] = n_topics

    # 1. TF-IDF 벡터화
    semantic_store["vectorizer"] = TfidfVectorizer(max_features=1000)
    tfidf_matrix = semantic_store["vectorizer"].fit_transform(documents)

    # 2. SVD (LSA)
    semantic_store["lsa_model"] = TruncatedSVD(n_components=n_topics, random_state=42)
    semantic_store["doc_topic_matrix"] = semantic_store["lsa_model"].fit_transform(tfidf_matrix)

    # 설명된 분산 비율
    explained_variance = semantic_store["lsa_model"].explained_variance_ratio_

    result = {
        "status": "success",
        "n_documents": len(documents),
        "n_topics": n_topics,
        "vocabulary_size": len(semantic_store["vectorizer"].vocabulary_),
        "explained_variance_ratio": [float(v) for v in explained_variance],
        "total_variance_explained": float(explained_variance.sum())
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def get_topics(n_words: int = 10) -> List[TextContent]:
    """토픽 추출"""
    if semantic_store["lsa_model"] is None:
        return [TextContent(type="text", text="Error: LSA 모델을 먼저 구축해주세요")]

    words = semantic_store["vectorizer"].get_feature_names_out()
    topics = []

    for topic_idx, topic in enumerate(semantic_store["lsa_model"].components_):
        top_word_indices = topic.argsort()[-n_words:][::-1]
        top_words = [
            {
                "word": words[i],
                "weight": float(topic[i])
            }
            for i in top_word_indices
        ]

        topics.append({
            "topic_id": topic_idx,
            "top_words": top_words
        })

    return [TextContent(type="text", text=json.dumps(topics, ensure_ascii=False, indent=2))]

async def semantic_search(query: str, top_k: int = 5) -> List[TextContent]:
    """의미적 검색"""
    if semantic_store["lsa_model"] is None:
        return [TextContent(type="text", text="Error: LSA 모델을 먼저 구축해주세요")]

    # 쿼리를 LSA 공간으로 변환
    query_tfidf = semantic_store["vectorizer"].transform([query])
    query_lsa = semantic_store["lsa_model"].transform(query_tfidf)

    # LSA 공간에서 코사인 유사도
    similarities = cosine_similarity(query_lsa, semantic_store["doc_topic_matrix"])[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = [
        {
            "rank": i + 1,
            "document": semantic_store["documents"][idx],
            "semantic_similarity": float(similarities[idx])
        }
        for i, idx in enumerate(top_indices)
    ]

    return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]

async def cluster_documents(n_clusters: int = 3) -> List[TextContent]:
    """문서 클러스터링"""
    if semantic_store["doc_topic_matrix"] is None:
        return [TextContent(type="text", text="Error: LSA 모델을 먼저 구축해주세요")]

    # K-means 클러스터링 (LSA 공간에서)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(semantic_store["doc_topic_matrix"])

    clusters = {}
    for doc_idx, cluster_id in enumerate(cluster_labels):
        cluster_id = int(cluster_id)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append({
            "document_index": doc_idx,
            "document": semantic_store["documents"][doc_idx]
        })

    result = {
        "n_clusters": n_clusters,
        "clusters": [
            {
                "cluster_id": cluster_id,
                "size": len(docs),
                "documents": docs
            }
            for cluster_id, docs in clusters.items()
        ]
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def document_topic_distribution(document_index: int) -> List[TextContent]:
    """문서의 토픽 분포"""
    if semantic_store["doc_topic_matrix"] is None:
        return [TextContent(type="text", text="Error: LSA 모델을 먼저 구축해주세요")]

    if document_index >= len(semantic_store["documents"]):
        return [TextContent(type="text", text="Error: 잘못된 문서 인덱스")]

    topic_dist = semantic_store["doc_topic_matrix"][document_index]

    result = {
        "document": semantic_store["documents"][document_index],
        "topic_distribution": [
            {
                "topic_id": i,
                "score": float(score)
            }
            for i, score in enumerate(topic_dist)
        ]
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="semantic-analysis-server",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
```

#### 사용 예시

```python
"""
User: 고객 리뷰 데이터에서 주요 토픽을 찾아줘

Agent:
[MCP Tool Call: build_lsa_model]
{
  "documents": [...리뷰 500개...],
  "n_topics": 5
}

[MCP Tool Call: get_topics]
{
  "n_words": 8
}

Agent: LSA 분석 결과, 5개의 주요 토픽을 발견했습니다:

토픽 0 (제품 품질): 품질, 내구성, 견고함, 우수함, 만족
토픽 1 (배송): 배송, 빠름, 신속, 포장, 안전
토픽 2 (가격): 가격, 저렴, 합리적, 가성비, 경제적
토픽 3 (고객서비스): 서비스, 응대, 친절, 신속, 해결
토픽 4 (사용성): 사용, 편리, 간단, 직관적, 쉬움

---

User: "배송이 너무 느려요"와 유사한 리뷰를 찾아줘

Agent:
[MCP Tool Call: semantic_search]
{
  "query": "배송이 너무 느려요",
  "top_k": 5
}

유사한 리뷰 5개를 찾았습니다 (토픽 1 관련):
1. "배송 속도가 기대에 못 미쳐요" (유사도: 0.92)
2. "주문 후 일주일이 걸렸어요" (유사도: 0.87)
...
"""
```

### MCP 서버 3: Embedding Server (Word2Vec/FastText/BERT)

#### 구현

```python
# embedding_mcp_server.py
import asyncio
import json
from typing import List, Dict, Any, Optional
from mcp.server.models import InitializationOptions
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import numpy as np
from gensim.models import Word2Vec, FastText
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-server")

app = Server("embedding-server")

# 전역 상태
embedding_store: Dict[str, Any] = {
    "word2vec_model": None,
    "fasttext_model": None,
    "bert_tokenizer": None,
    "bert_model": None
}

@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="train_word2vec",
            description="Word2Vec 모델을 학습합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "sentences": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "description": "학습 문장 목록 (토큰화된 형태)"
                    },
                    "vector_size": {"type": "integer", "default": 100},
                    "window": {"type": "integer", "default": 5},
                    "sg": {
                        "type": "integer",
                        "description": "0=CBOW, 1=Skip-gram",
                        "default": 1
                    }
                },
                "required": ["sentences"]
            }
        ),
        Tool(
            name="train_fasttext",
            description="FastText 모델을 학습합니다 (OOV 처리 가능)",
            inputSchema={
                "type": "object",
                "properties": {
                    "sentences": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "vector_size": {"type": "integer", "default": 100}
                },
                "required": ["sentences"]
            }
        ),
        Tool(
            name="load_bert",
            description="BERT 모델을 로드합니다 (한국어 지원)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "BERT 모델 이름",
                        "default": "bert-base-multilingual-cased"
                    }
                }
            }
        ),
        Tool(
            name="get_word_embedding",
            description="단어의 임베딩 벡터를 반환합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "model_type": {
                        "type": "string",
                        "enum": ["word2vec", "fasttext"],
                        "default": "word2vec"
                    }
                },
                "required": ["word"]
            }
        ),
        Tool(
            name="get_bert_embedding",
            description="BERT를 사용하여 문장/단어의 contextualized embedding을 반환합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="find_similar_words",
            description="주어진 단어와 유사한 단어를 찾습니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                    "model_type": {
                        "type": "string",
                        "enum": ["word2vec", "fasttext"],
                        "default": "word2vec"
                    }
                },
                "required": ["word"]
            }
        ),
        Tool(
            name="word_analogy",
            description="단어 유추: king - man + woman = ?",
            inputSchema={
                "type": "object",
                "properties": {
                    "positive": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "더할 단어들 (예: [king, woman])"
                    },
                    "negative": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "뺄 단어들 (예: [man])"
                    },
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["positive", "negative"]
            }
        ),
        Tool(
            name="word_similarity",
            description="두 단어 간의 유사도를 계산합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "word1": {"type": "string"},
                    "word2": {"type": "string"},
                    "model_type": {
                        "type": "string",
                        "enum": ["word2vec", "fasttext"],
                        "default": "word2vec"
                    }
                },
                "required": ["word1", "word2"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    try:
        if name == "train_word2vec":
            return await train_word2vec(
                arguments["sentences"],
                arguments.get("vector_size", 100),
                arguments.get("window", 5),
                arguments.get("sg", 1)
            )

        elif name == "train_fasttext":
            return await train_fasttext(
                arguments["sentences"],
                arguments.get("vector_size", 100)
            )

        elif name == "load_bert":
            model_name = arguments.get("model_name", "bert-base-multilingual-cased")
            return await load_bert(model_name)

        elif name == "get_word_embedding":
            model_type = arguments.get("model_type", "word2vec")
            return await get_word_embedding(arguments["word"], model_type)

        elif name == "get_bert_embedding":
            return await get_bert_embedding(arguments["text"])

        elif name == "find_similar_words":
            return await find_similar_words(
                arguments["word"],
                arguments.get("top_k", 10),
                arguments.get("model_type", "word2vec")
            )

        elif name == "word_analogy":
            return await word_analogy(
                arguments["positive"],
                arguments["negative"],
                arguments.get("top_k", 5)
            )

        elif name == "word_similarity":
            return await word_similarity(
                arguments["word1"],
                arguments["word2"],
                arguments.get("model_type", "word2vec")
            )

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error in {name}: {str(e)}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def train_word2vec(sentences: List[List[str]], vector_size: int, window: int, sg: int) -> List[TextContent]:
    """Word2Vec 학습"""
    embedding_store["word2vec_model"] = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=sg,
        workers=4
    )

    result = {
        "status": "success",
        "model_type": "Skip-gram" if sg == 1 else "CBOW",
        "vocabulary_size": len(embedding_store["word2vec_model"].wv),
        "vector_size": vector_size
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def train_fasttext(sentences: List[List[str]], vector_size: int) -> List[TextContent]:
    """FastText 학습"""
    embedding_store["fasttext_model"] = FastText(
        sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4
    )

    result = {
        "status": "success",
        "model_type": "FastText",
        "vocabulary_size": len(embedding_store["fasttext_model"].wv),
        "vector_size": vector_size,
        "supports_oov": True
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def load_bert(model_name: str) -> List[TextContent]:
    """BERT 모델 로드"""
    embedding_store["bert_tokenizer"] = BertTokenizer.from_pretrained(model_name)
    embedding_store["bert_model"] = BertModel.from_pretrained(model_name)
    embedding_store["bert_model"].eval()

    result = {
        "status": "success",
        "model_name": model_name,
        "embedding_dim": embedding_store["bert_model"].config.hidden_size
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def get_word_embedding(word: str, model_type: str) -> List[TextContent]:
    """단어 임베딩 추출"""
    if model_type == "word2vec":
        if embedding_store["word2vec_model"] is None:
            return [TextContent(type="text", text="Error: Word2Vec 모델을 먼저 학습해주세요")]

        if word not in embedding_store["word2vec_model"].wv:
            return [TextContent(type="text", text=f"Error: '{word}'는 어휘에 없습니다")]

        vector = embedding_store["word2vec_model"].wv[word]

    else:  # fasttext
        if embedding_store["fasttext_model"] is None:
            return [TextContent(type="text", text="Error: FastText 모델을 먼저 학습해주세요")]

        vector = embedding_store["fasttext_model"].wv[word]

    result = {
        "word": word,
        "model_type": model_type,
        "vector_preview": vector[:10].tolist(),
        "vector_dim": len(vector),
        "vector_norm": float(np.linalg.norm(vector))
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def get_bert_embedding(text: str) -> List[TextContent]:
    """BERT 임베딩 추출"""
    if embedding_store["bert_model"] is None:
        return [TextContent(type="text", text="Error: BERT 모델을 먼저 로드해주세요")]

    inputs = embedding_store["bert_tokenizer"](text, return_tensors='pt')

    with torch.no_grad():
        outputs = embedding_store["bert_model"](**inputs)

    # [CLS] 토큰의 임베딩 사용
    cls_embedding = outputs.last_hidden_state[0][0].numpy()

    result = {
        "text": text,
        "model_type": "BERT",
        "embedding_preview": cls_embedding[:10].tolist(),
        "embedding_dim": len(cls_embedding)
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def find_similar_words(word: str, top_k: int, model_type: str) -> List[TextContent]:
    """유사 단어 찾기"""
    model = embedding_store.get(f"{model_type}_model")
    if model is None:
        return [TextContent(type="text", text=f"Error: {model_type} 모델을 먼저 학습해주세요")]

    similar = model.wv.most_similar(word, topn=top_k)

    result = {
        "query_word": word,
        "similar_words": [
            {"word": w, "similarity": float(sim)}
            for w, sim in similar
        ]
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def word_analogy(positive: List[str], negative: List[str], top_k: int) -> List[TextContent]:
    """단어 유추"""
    if embedding_store["word2vec_model"] is None:
        return [TextContent(type="text", text="Error: Word2Vec 모델을 먼저 학습해주세요")]

    results = embedding_store["word2vec_model"].wv.most_similar(
        positive=positive,
        negative=negative,
        topn=top_k
    )

    result = {
        "analogy": f"{' + '.join(positive)} - {' - '.join(negative)} = ?",
        "results": [
            {"word": w, "score": float(sim)}
            for w, sim in results
        ]
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def word_similarity(word1: str, word2: str, model_type: str) -> List[TextContent]:
    """단어 유사도"""
    model = embedding_store.get(f"{model_type}_model")
    if model is None:
        return [TextContent(type="text", text=f"Error: {model_type} 모델을 먼저 학습해주세요")]

    similarity = model.wv.similarity(word1, word2)

    result = {
        "word1": word1,
        "word2": word2,
        "cosine_similarity": float(similarity),
        "interpretation": "매우 유사" if similarity > 0.7 else "유사" if similarity > 0.4 else "다소 유사" if similarity > 0.2 else "거의 다름"
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="embedding-server",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### MCP 서버 4: Topic Modeling Server (LDA/NMF)

#### 구현

```python
# topic_modeling_mcp_server.py
import asyncio
import json
from typing import List, Dict, Any
from mcp.server.models import InitializationOptions
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("topic-modeling-server")

app = Server("topic-modeling-server")

topic_store: Dict[str, Any] = {
    "documents": [],
    "lda_model": None,
    "nmf_model": None,
    "vectorizer": None,
    "doc_term_matrix": None
}

@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="train_lda",
            description="LDA (Latent Dirichlet Allocation) 토픽 모델을 학습합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "n_topics": {"type": "integer", "default": 5},
                    "max_iter": {"type": "integer", "default": 10}
                },
                "required": ["documents"]
            }
        ),
        Tool(
            name="train_nmf",
            description="NMF (Non-negative Matrix Factorization) 토픽 모델을 학습합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "n_topics": {"type": "integer", "default": 5}
                },
                "required": ["documents"]
            }
        ),
        Tool(
            name="get_topics",
            description="학습된 토픽과 각 토픽의 주요 단어를 반환합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "enum": ["lda", "nmf"]
                    },
                    "n_words": {"type": "integer", "default": 10}
                },
                "required": ["model_type"]
            }
        ),
        Tool(
            name="get_document_topics",
            description="특정 문서의 토픽 분포를 반환합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_index": {"type": "integer"},
                    "model_type": {
                        "type": "string",
                        "enum": ["lda", "nmf"]
                    }
                },
                "required": ["document_index", "model_type"]
            }
        ),
        Tool(
            name="infer_topic",
            description="새로운 문서의 토픽 분포를 추론합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "model_type": {
                        "type": "string",
                        "enum": ["lda", "nmf"]
                    }
                },
                "required": ["text", "model_type"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    try:
        if name == "train_lda":
            return await train_lda(
                arguments["documents"],
                arguments.get("n_topics", 5),
                arguments.get("max_iter", 10)
            )

        elif name == "train_nmf":
            return await train_nmf(
                arguments["documents"],
                arguments.get("n_topics", 5)
            )

        elif name == "get_topics":
            return await get_topics(
                arguments["model_type"],
                arguments.get("n_words", 10)
            )

        elif name == "get_document_topics":
            return await get_document_topics(
                arguments["document_index"],
                arguments["model_type"]
            )

        elif name == "infer_topic":
            return await infer_topic(
                arguments["text"],
                arguments["model_type"]
            )

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error in {name}: {str(e)}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def train_lda(documents: List[str], n_topics: int, max_iter: int) -> List[TextContent]:
    """LDA 학습"""
    topic_store["documents"] = documents

    # CountVectorizer 사용 (LDA는 단어 빈도 사용)
    topic_store["vectorizer"] = CountVectorizer(max_features=1000)
    topic_store["doc_term_matrix"] = topic_store["vectorizer"].fit_transform(documents)

    # LDA 학습
    topic_store["lda_model"] = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        random_state=42
    )

    doc_topic_dist = topic_store["lda_model"].fit_transform(topic_store["doc_term_matrix"])

    result = {
        "status": "success",
        "model_type": "LDA",
        "n_documents": len(documents),
        "n_topics": n_topics,
        "perplexity": float(topic_store["lda_model"].perplexity(topic_store["doc_term_matrix"]))
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def train_nmf(documents: List[str], n_topics: int) -> List[TextContent]:
    """NMF 학습"""
    topic_store["documents"] = documents

    # TfidfVectorizer 사용 (NMF는 TF-IDF 사용)
    topic_store["vectorizer"] = TfidfVectorizer(max_features=1000)
    topic_store["doc_term_matrix"] = topic_store["vectorizer"].fit_transform(documents)

    # NMF 학습
    topic_store["nmf_model"] = NMF(
        n_components=n_topics,
        random_state=42
    )

    doc_topic_dist = topic_store["nmf_model"].fit_transform(topic_store["doc_term_matrix"])

    result = {
        "status": "success",
        "model_type": "NMF",
        "n_documents": len(documents),
        "n_topics": n_topics,
        "reconstruction_error": float(topic_store["nmf_model"].reconstruction_err_)
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def get_topics(model_type: str, n_words: int) -> List[TextContent]:
    """토픽 추출"""
    model = topic_store.get(f"{model_type}_model")
    if model is None:
        return [TextContent(type="text", text=f"Error: {model_type.upper()} 모델을 먼저 학습해주세요")]

    words = topic_store["vectorizer"].get_feature_names_out()
    topics = []

    for topic_idx, topic in enumerate(model.components_):
        top_word_indices = topic.argsort()[-n_words:][::-1]
        top_words = [
            {
                "word": words[i],
                "weight": float(topic[i])
            }
            for i in top_word_indices
        ]

        topics.append({
            "topic_id": topic_idx,
            "top_words": top_words
        })

    return [TextContent(type="text", text=json.dumps(topics, ensure_ascii=False, indent=2))]

async def get_document_topics(document_index: int, model_type: str) -> List[TextContent]:
    """문서의 토픽 분포"""
    model = topic_store.get(f"{model_type}_model")
    if model is None:
        return [TextContent(type="text", text=f"Error: {model_type.upper()} 모델을 먼저 학습해주세요")]

    if document_index >= len(topic_store["documents"]):
        return [TextContent(type="text", text="Error: 잘못된 문서 인덱스")]

    doc_vector = topic_store["doc_term_matrix"][document_index]
    topic_dist = model.transform(doc_vector)[0]

    # 정규화 (확률로 변환)
    topic_dist = topic_dist / topic_dist.sum()

    result = {
        "document": topic_store["documents"][document_index],
        "topic_distribution": [
            {
                "topic_id": i,
                "probability": float(prob)
            }
            for i, prob in enumerate(topic_dist)
        ]
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def infer_topic(text: str, model_type: str) -> List[TextContent]:
    """새 문서의 토픽 추론"""
    model = topic_store.get(f"{model_type}_model")
    if model is None:
        return [TextContent(type="text", text=f"Error: {model_type.upper()} 모델을 먼저 학습해주세요")]

    doc_vector = topic_store["vectorizer"].transform([text])
    topic_dist = model.transform(doc_vector)[0]

    # 정규화
    topic_dist = topic_dist / topic_dist.sum()

    # 주요 토픽 찾기
    dominant_topic = int(np.argmax(topic_dist))

    result = {
        "text": text,
        "dominant_topic": dominant_topic,
        "dominant_topic_probability": float(topic_dist[dominant_topic]),
        "all_topic_probabilities": [
            {
                "topic_id": i,
                "probability": float(prob)
            }
            for i, prob in enumerate(topic_dist)
        ]
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="topic-modeling-server",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎯 실전 사용 사례: AI 에이전트 + MCP 서버 오케스트레이션

### 사례 1: 학술 논문 자동 분석 시스템

**시나리오**: 연구자가 특정 주제의 논문 수백 개를 분석하고 싶어합니다.

```python
"""
User: arXiv에서 다운로드한 머신러닝 논문 500개를 분석해서 주요 연구 트렌드를 알려줘

Agent 워크플로우:

1. [Text Analysis MCP] - 논문 추가 및 인덱싱
   add_documents(논문 500개)

2. [Semantic Analysis MCP] - LSA로 토픽 발견
   build_lsa_model(논문 500개, n_topics=10)
   get_topics(n_words=15)

   결과:
   - 토픽 0: transformer, attention, self-attention, bert, gpt...
   - 토픽 1: reinforcement, learning, reward, policy, agent...
   - 토픽 2: computer, vision, cnn, detection, segmentation...
   - 토픽 3: optimization, gradient, adam, learning_rate...
   ...

3. [Topic Modeling MCP] - LDA로 세부 토픽 분석
   train_lda(논문 500개, n_topics=20)
   get_topics(model_type="lda", n_words=10)

4. [Embedding MCP] - 핵심 용어 임베딩
   train_word2vec(논문 토큰화, vector_size=200)
   find_similar_words("transformer", top_k=20)

   유사 단어: attention, bert, gpt, encoder, decoder...

5. 종합 분석 리포트 생성

Agent 응답:
분석 완료! 500개 논문의 주요 연구 트렌드:

📊 **주요 토픽 (LSA 분석)**
1. Transformer 기반 모델 (23% 비중)
2. 강화학습 (18% 비중)
3. 컴퓨터 비전 (15% 비중)
...

🔥 **최신 트렌드 (LDA 분석)**
- Self-supervised learning
- Few-shot learning
- Multi-modal learning
- Efficient transformers

📈 **연관 기술 (Word2Vec 분석)**
- Transformer ↔ Attention (유사도: 0.92)
- BERT ↔ Pre-training (유사도: 0.88)
...
"""
```

### 사례 2: 고객 리뷰 감성 및 주제 분석

```python
"""
User: 우리 제품 리뷰 10,000개를 분석해서 고객 불만사항을 찾아줘

Agent 워크플로우:

1. [Text Analysis MCP] - 키워드 추출
   for review in reviews:
       keywords = extract_keywords(review, top_k=5)

   빈도 높은 키워드: 배송, 품질, 가격, 포장, 서비스

2. [Topic Modeling MCP] - 불만 토픽 발견
   train_nmf(부정 리뷰들, n_topics=5)
   get_topics(model_type="nmf")

   토픽 0: 배송, 늦음, 지연, 오래, 느림 → **배송 지연 문제**
   토픽 1: 파손, 깨짐, 포장, 불량 → **포장 문제**
   토픽 2: 작동, 고장, 불량, 반품 → **제품 결함**
   토픽 3: 가격, 비싸, 부담, 할인 → **가격 불만**
   토픽 4: 응대, 서비스, 불친절, 환불 → **고객서비스 문제**

3. [Semantic Analysis MCP] - 유사 불만 그룹화
   build_lsa_model(부정 리뷰들, n_topics=5)
   cluster_documents(n_clusters=5)

   클러스터 0 (2,341개): 배송 관련
   클러스터 1 (1,892개): 품질 관련
   ...

4. [Embedding MCP] - 불만 키워드 임베딩으로 연관 분석
   train_fasttext(리뷰 토큰들)
   find_similar_words("배송", top_k=20)

   연관 단어: 늦음, 지연, 오래, 기다림, 도착...

Agent 응답:
🚨 **주요 불만사항 TOP 5**

1. **배송 지연** (23.4% / 2,341건)
   - 평균 배송일: 예상 3일 → 실제 7일
   - 핵심 키워드: 늦음, 지연, 기다림
   - 제안: 배송 프로세스 개선 필요

2. **포장 불량** (18.9% / 1,892건)
   - 파손율: 12%
   - 핵심 키워드: 깨짐, 파손, 포장
   - 제안: 완충재 보강

...

📊 **토픽별 감성 분포**
토픽 0 (배송): 부정 78%, 중립 15%, 긍정 7%
토픽 1 (품질): 부정 45%, 중립 30%, 긍정 25%
...
"""
```

### 사례 3: 법률 문서 유사도 검색 시스템

```python
"""
User: 이 계약서와 유사한 선례를 데이터베이스에서 찾아줘

Agent 워크플로우:

1. [Text Analysis MCP] - 계약서 데이터베이스 구축
   add_documents(계약서 10,000개)

2. [Embedding MCP] - BERT로 contextualized embedding
   load_bert("klue/bert-base")
   query_emb = get_bert_embedding(새 계약서)

   for doc in database:
       doc_emb = get_bert_embedding(doc)
       similarity = cosine_similarity(query_emb, doc_emb)

3. [Semantic Analysis MCP] - LSA로 의미적 유사 문서 검색
   semantic_search(새 계약서, top_k=10)

4. [Text Analysis MCP] - TF-IDF로 키워드 기반 검색
   search_documents(새 계약서 핵심 조항, top_k=10)

5. 앙상블: 3가지 방법의 결과 종합

Agent 응답:
📄 **유사 계약서 검색 결과**

**방법 1: BERT Embedding (문맥 기반)**
1. 계약서 #4532 (유사도: 0.94) - 동일 업종, 유사 조건
2. 계약서 #1203 (유사도: 0.89) - 유사한 배상 조항
...

**방법 2: LSA Semantic Search (토픽 기반)**
1. 계약서 #4532 (유사도: 0.91)
2. 계약서 #7821 (유사도: 0.87)
...

**방법 3: TF-IDF Keyword (키워드 기반)**
1. 계약서 #4532 (유사도: 0.88)
2. 계약서 #1203 (유사도: 0.85)
...

🎯 **최종 추천 (앙상블)**
1. 계약서 #4532 - 모든 방법에서 1위 (종합 유사도: 0.91)
   - 공통 조항: 7개 / 8개
   - 업종: 동일
   - 계약 유형: 동일
"""
```

### 사례 4: 뉴스 기사 자동 카테고리 분류

```python
"""
User: 실시간으로 들어오는 뉴스 기사를 자동으로 분류해줘

Agent 워크플로우:

1. [초기 학습] Topic Modeling MCP
   train_lda(과거 뉴스 10,000개, n_topics=10)
   get_topics(model_type="lda")

   토픽 0: 정치 (국회, 대통령, 법안, 정당...)
   토픽 1: 경제 (금리, 주식, 환율, GDP...)
   토픽 2: IT (AI, 스마트폰, 소프트웨어...)
   ...

2. [실시간 처리] 새 기사 입력
   for new_article in stream:
       # 토픽 추론
       topic_dist = infer_topic(new_article, model_type="lda")

       # 키워드 추출
       keywords = extract_keywords(new_article, top_k=5)

       # 유사 기사 찾기 (중복 체크)
       similar = semantic_search(new_article, top_k=3)

       # 분류 및 저장
       category = dominant_topic
       save_to_database(new_article, category, keywords, similar)

3. [주기적 재학습] 매일 자정
   새로운 기사들로 모델 업데이트
   train_lda(최근 30일 기사, n_topics=10)

Agent 응답:
✅ **뉴스 분류 시스템 가동 중**

📰 **최근 처리 기사 (최근 1시간)**
1. "AI 규제 법안 국회 통과"
   → 카테고리: 정치 (0.65) + IT (0.30)
   → 키워드: AI, 규제, 국회, 법안, 통과

2. "코스피 2,500 돌파"
   → 카테고리: 경제 (0.92)
   → 키워드: 코스피, 주식, 상승, 시장

3. "삼성, 신형 갤럭시 공개"
   → 카테고리: IT (0.88)
   → 키워드: 삼성, 갤럭시, 스마트폰, 출시
   → 중복 의심: 3시간 전 유사 기사 발견

📊 **오늘 분류 통계**
- 총 처리: 1,247건
- 정치: 234건 (18.8%)
- 경제: 312건 (25.0%)
- IT: 189건 (15.2%)
...
"""
```

---

## 📋 정리 및 비교

### 각 MCP 서버의 활용 분야

| MCP 서버 | 주요 기능 | 적용 분야 | 장점 |
|---------|---------|---------|------|
| **Text Analysis** | TF-IDF, 키워드 추출, 문서 검색 | 검색 엔진, 추천 시스템 | 빠르고 정확, 구현 간단 |
| **Semantic Analysis** | LSA, 의미 분석, 클러스터링 | 문서 분류, 토픽 발견 | 유의어 처리, 차원 축소 |
| **Embedding** | Word2Vec, FastText, BERT | 단어 유사도, 유추, 임베딩 | 의미 벡터화, 전이학습 |
| **Topic Modeling** | LDA, NMF, 토픽 추출 | 트렌드 분석, 주제 발견 | 확률 모델, 해석 가능 |

### 실전 조합 패턴

1. **검색 시스템**
   - Text Analysis (1차 필터링) → Semantic Analysis (의미 검색) → Embedding (재순위화)

2. **문서 분류**
   - Topic Modeling (토픽 발견) → Text Analysis (키워드 추출) → 분류

3. **추천 시스템**
   - Embedding (사용자/아이템 임베딩) → Semantic Analysis (유사도 계산)

4. **트렌드 분석**
   - Topic Modeling (시간별 토픽) → Text Analysis (키워드 변화) → 시각화

---

이제 전통적 NLP/ML 이론부터 MCP 구현, 실전 사용 사례까지 모두 정리되었습니다!
