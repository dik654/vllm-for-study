# Vector Databases & RAG (Retrieval-Augmented Generation)

## 🎯 목표

**LLM의 지식을 외부 데이터로 확장하기**

문제: GPT는 훈련 데이터까지만 알고, 최신 정보나 회사 내부 문서는 모름
해결: **Vector DB + RAG**로 실시간 정보 검색 및 활용!

---

## 🤔 왜 Vector Database가 필요한가?

### 전통적 검색의 한계

```python
# 키워드 기반 검색 (전통적)
query = "강아지 사료 추천"
results = sql_db.search("SELECT * FROM products WHERE name LIKE '%강아지%' AND name LIKE '%사료%'")

# 문제:
# ❌ "반려견 식사" → 검색 안됨 (키워드 다름)
# ❌ "puppy food" → 검색 안됨 (언어 다름)
# ❌ "애완동물 영양식" → 검색 안됨 (동의어 처리 안됨)
```

### 의미 기반 검색 (Semantic Search)

```python
# Vector 기반 검색 (현대적)
query = "강아지 사료 추천"
query_vector = embedding_model.encode(query)  # 벡터로 변환

results = vector_db.search(query_vector, top_k=5)

# 장점:
# ✅ "반려견 식사" → 검색됨 (의미가 유사)
# ✅ "puppy food" → 검색됨 (cross-lingual)
# ✅ "애완동물 영양식" → 검색됨 (동의어 이해)
```

**핵심**: 단어가 아닌 **의미**를 검색!

---

## 📐 Vector Embeddings 기초

### Embedding이란?

**텍스트를 고차원 벡터로 변환**

```python
from sentence_transformers import SentenceTransformer

# Embedding 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 텍스트 → 벡터
text1 = "강아지가 좋아요"
text2 = "개를 사랑해요"
text3 = "고양이가 귀여워요"

# 각 문장을 384차원 벡터로 변환
# 의도: 의미가 비슷한 문장은 벡터 공간에서 가까이 위치
vec1 = model.encode(text1)  # shape: (384,)
vec2 = model.encode(text2)  # shape: (384,)
vec3 = model.encode(text3)  # shape: (384,)

print(f"Vector dimension: {vec1.shape}")
# Output: Vector dimension: (384,)
```

### Similarity Metrics

```python
import numpy as np

def cosine_similarity(v1, v2):
    """
    Cosine Similarity: 두 벡터의 각도로 유사도 측정

    범위: [-1, 1]
    - 1: 완전히 같은 방향 (매우 유사)
    - 0: 직교 (무관)
    - -1: 반대 방향 (반대 의미)

    의도: 벡터의 크기가 아닌 방향으로 유사도 측정
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1, v2):
    """
    Euclidean Distance: 두 벡터 사이의 직선 거리

    범위: [0, ∞)
    - 0: 완전히 동일
    - 큰 값: 매우 다름

    의도: 벡터 공간에서의 실제 거리 측정
    """
    return np.linalg.norm(v1 - v2)

def dot_product(v1, v2):
    """
    Dot Product: 내적

    범위: (-∞, ∞)
    - 큰 양수: 유사하고 크기도 큼
    - 0 근처: 직교
    - 음수: 반대 방향

    의도: 정규화되지 않은 유사도 (크기 고려)
    """
    return np.dot(v1, v2)


# 유사도 계산
sim_12 = cosine_similarity(vec1, vec2)  # "강아지" vs "개"
sim_13 = cosine_similarity(vec1, vec3)  # "강아지" vs "고양이"

print(f"'강아지'와 '개' 유사도: {sim_12:.3f}")      # ~0.85 (높음)
print(f"'강아지'와 '고양이' 유사도: {sim_13:.3f}")  # ~0.65 (중간)
```

### Vector Space Visualization

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 여러 문장의 embedding
sentences = [
    "강아지가 좋아요",
    "개를 사랑해요",
    "강아지 산책",
    "고양이가 귀여워요",
    "고양이 밥주기",
    "자동차 운전",
    "비행기 여행"
]

# Embedding (384차원)
vectors = model.encode(sentences)

# PCA로 2차원으로 축소 (시각화용)
# 의도: 고차원 벡터를 2D로 투영하여 관계 파악
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# 시각화
plt.figure(figsize=(10, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

for i, sentence in enumerate(sentences):
    plt.annotate(sentence, (vectors_2d[i, 0], vectors_2d[i, 1]))

plt.title('Sentence Embeddings Visualization (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# 관찰:
# - "강아지" 관련 문장들이 가까이 클러스터링
# - "고양이" 관련 문장들도 가까이
# - "자동차", "비행기"는 멀리 떨어짐
```

---

## 🔍 Approximate Nearest Neighbor (ANN)

### 왜 ANN이 필요한가?

```python
# Naive Search (완전 탐색)
def naive_search(query_vector, database_vectors, top_k=5):
    """
    모든 벡터와 거리 계산 → 정렬

    시간 복잡도: O(n × d)
    - n: 데이터베이스 벡터 수
    - d: 벡터 차원

    문제: 백만 개 벡터면 백만 번 계산!
    """
    similarities = []
    for db_vector in database_vectors:
        sim = cosine_similarity(query_vector, db_vector)
        similarities.append(sim)

    # 정렬 O(n log n)
    top_indices = np.argsort(similarities)[-top_k:]
    return top_indices


# ANN Search (근사 검색)
"""
핵심 아이디어: 정확도를 약간 희생하고 속도를 대폭 향상

알고리즘:
1. HNSW (Hierarchical Navigable Small World)
   - 그래프 기반
   - 계층적 구조
   - 시간: O(log n)

2. IVF (Inverted File Index)
   - 클러스터링 기반
   - Voronoi cell로 분할
   - 시간: O(√n)

3. LSH (Locality Sensitive Hashing)
   - 해싱 기반
   - 유사한 벡터를 같은 버킷에
   - 시간: O(1) ~ O(log n)
"""
```

### HNSW 원리

```python
"""
HNSW (Hierarchical Navigable Small World)

개념:
  Layer 2: [A] ←→ [Z]              (sparse, long-range)
            ↓      ↓
  Layer 1: [A]←→[M]←→[Z]           (medium density)
            ↓    ↓    ↓
  Layer 0: [A]←→[D]←→[M]←→[W]←→[Z] (dense, short-range)

검색 과정:
  1. 최상위 layer에서 시작
  2. Greedy search로 가장 가까운 노드 찾기
  3. 다음 layer로 내려가서 반복
  4. Layer 0에서 정확한 이웃 찾기

장점:
  - 시간: O(log n)
  - 높은 정확도 (~99%)
  - 동적 추가/삭제 가능

단점:
  - 메모리 사용량 높음 (그래프 구조)
"""
```

---

## 🔬 내부 동작 원리 (Deep Dive)

### HNSW 상세 동작 원리

#### 1. 그래프 구조 생성

```python
import numpy as np
from collections import defaultdict
import heapq

class HNSWGraph:
    """
    HNSW 그래프 구조 직접 구현

    핵심 아이디어: 계층적 Skip List와 Small World Network 결합
    - Skip List: 빠른 탐색을 위한 계층 구조
    - Small World: 지역 연결 + 장거리 연결
    """

    def __init__(self, m=16, m_max=16, ef_construction=200, ml=1.0/np.log(2.0)):
        """
        Args:
            m: 각 노드의 최대 연결 수
            m_max: Layer 0의 최대 연결 수 (보통 m * 2)
            ef_construction: 구축 시 후보 수 (높을수록 정확, 느림)
            ml: Level 결정 확률 분포 파라미터

        의도:
        - m이 클수록: 정확도 ↑, 메모리 ↑, 검색 속도 ↓
        - ef_construction이 클수록: 인덱스 품질 ↑, 구축 시간 ↑
        """
        self.m = m
        self.m_max = m_max
        self.ef_construction = ef_construction
        self.ml = ml

        # 그래프 구조: layer별로 인접 리스트
        # layers[level][node_id] = [neighbor_ids]
        self.layers = defaultdict(lambda: defaultdict(list))

        # 각 노드의 벡터 저장
        self.vectors = {}

        # 각 노드의 최대 레벨
        self.node_levels = {}

        # 진입점 (최상위 레이어에서 검색 시작)
        self.entry_point = None
        self.max_level = -1

    def _get_random_level(self):
        """
        새 노드의 레벨을 확률적으로 결정

        수식: level = floor(-ln(uniform(0,1)) * ml)

        의도:
        - 대부분의 노드는 낮은 레벨 (Layer 0에만)
        - 소수의 노드는 높은 레벨 (여러 층에 걸쳐)
        - 이것이 계층 구조를 만듦!

        확률 분포:
        - Layer 0: 100% (모든 노드)
        - Layer 1: ~50%
        - Layer 2: ~25%
        - Layer 3: ~12.5%
        ...
        """
        return int(-np.log(np.random.uniform(0, 1)) * self.ml)

    def _distance(self, v1, v2):
        """벡터 간 거리 (L2 distance)"""
        return np.linalg.norm(np.array(v1) - np.array(v2))

    def _search_layer(self, query, entry_points, num_closest, level):
        """
        특정 레이어에서 Greedy Search

        알고리즘:
        1. visited = 빈 set
        2. candidates = entry_points (우선순위 큐, min-heap)
        3. w = 현재까지 찾은 최근접 이웃들 (max-heap)

        반복:
        4. c = candidates에서 가장 가까운 점 pop
        5. 만약 c가 w의 가장 먼 점보다 멀면 → 종료
        6. c의 이웃들 확인:
           - 아직 방문 안했으면 candidates와 w에 추가
        7. w가 num_closest개 넘으면 가장 먼 점 제거

        의도: Greedy하게 점점 query에 가까워지는 경로 탐색
        """
        visited = set()
        # candidates: (distance, node_id) - min heap
        candidates = [(self._distance(query, self.vectors[ep]), ep) for ep in entry_points]
        heapq.heapify(candidates)

        # w: (distance, node_id) - max heap (음수로 저장)
        w = [(-dist, node) for dist, node in candidates]
        heapq.heapify(w)

        for ep in entry_points:
            visited.add(ep)

        while candidates:
            # 가장 가까운 후보 선택
            current_dist, current = heapq.heappop(candidates)

            # w의 가장 먼 점보다 current가 멀면 종료
            # 의도: 더 이상 개선 불가능
            if current_dist > -w[0][0]:
                break

            # 이웃들 확인
            for neighbor in self.layers[level][current]:
                if neighbor not in visited:
                    visited.add(neighbor)

                    neighbor_dist = self._distance(query, self.vectors[neighbor])

                    # w의 가장 먼 점보다 가까우면 추가
                    if neighbor_dist < -w[0][0] or len(w) < num_closest:
                        heapq.heappush(candidates, (neighbor_dist, neighbor))
                        heapq.heappush(w, (-neighbor_dist, neighbor))

                        # w 크기 제한
                        if len(w) > num_closest:
                            heapq.heappop(w)

        # 결과 반환 (거리 순으로 정렬)
        return sorted([(-dist, node) for dist, node in w])

    def insert(self, node_id, vector):
        """
        새 노드 삽입

        알고리즘:
        1. 노드의 레벨 결정 (확률적)
        2. 최상위 레이어부터 검색 시작
        3. 각 레이어에서:
           - 가장 가까운 m개 노드 찾기
           - 양방향 연결 생성
        4. 진입점 업데이트 (필요시)

        의도: 계층 구조를 유지하면서 효율적으로 삽입
        """
        self.vectors[node_id] = vector
        level = self._get_random_level()
        self.node_levels[node_id] = level

        if self.entry_point is None:
            # 첫 노드
            self.entry_point = node_id
            self.max_level = level
            return

        # 최상위 레이어부터 검색
        current_nearest = [self.entry_point]

        # Level (max_level → level+1): 검색만 (연결 안함)
        for lc in range(self.max_level, level, -1):
            current_nearest = [node for _, node in self._search_layer(
                vector, current_nearest, 1, lc
            )]

        # Level (level → 0): 검색 + 연결
        for lc in range(level, -1, -1):
            candidates = self._search_layer(
                vector, current_nearest, self.ef_construction, lc
            )

            # M개 선택 (heuristic)
            m = self.m if lc > 0 else self.m_max
            neighbors = self._select_neighbors(candidates, m)

            # 양방향 연결 생성
            # 의도: 그래프의 대칭성 유지
            for neighbor_dist, neighbor in neighbors:
                self.layers[lc][node_id].append(neighbor)
                self.layers[lc][neighbor].append(node_id)

                # 이웃의 연결 수가 m 초과하면 pruning
                if len(self.layers[lc][neighbor]) > m:
                    self._prune_connections(neighbor, m, lc)

            current_nearest = [node for _, node in neighbors]

        # 진입점 업데이트
        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id

    def _select_neighbors(self, candidates, m):
        """
        이웃 선택 휴리스틱

        단순 버전: 가장 가까운 m개
        고급 버전: Diversity 고려 (각도 분산)

        의도: 그래프가 한쪽으로 치우치지 않도록
        """
        return candidates[:m]

    def _prune_connections(self, node_id, m, level):
        """
        연결 수 제한

        의도: 각 노드의 연결 수를 m으로 유지
        - 메모리 사용량 제한
        - 검색 속도 유지
        """
        neighbors = self.layers[level][node_id]

        # 거리 계산 후 정렬
        neighbor_dists = [
            (self._distance(self.vectors[node_id], self.vectors[n]), n)
            for n in neighbors
        ]
        neighbor_dists.sort()

        # 가장 가까운 m개만 유지
        self.layers[level][node_id] = [n for _, n in neighbor_dists[:m]]

    def search(self, query, k=10, ef=50):
        """
        K-NN 검색

        Args:
            query: 쿼리 벡터
            k: 반환할 이웃 수
            ef: 검색 시 후보 수 (높을수록 정확, 느림)

        알고리즘:
        1. 최상위 레이어에서 시작 (entry_point)
        2. 각 레이어에서 greedy search
        3. Layer 0에서 ef개 후보 찾기
        4. 가장 가까운 k개 반환

        시간 복잡도: O(log n)
        - 각 레이어에서 평균 O(1)번 점프
        - 레이어 수는 O(log n)
        """
        if self.entry_point is None:
            return []

        # 최상위 레이어부터 검색
        current_nearest = [self.entry_point]

        # Layer (max_level → 1): 빠른 이동
        for level in range(self.max_level, 0, -1):
            current_nearest = [node for _, node in self._search_layer(
                query, current_nearest, 1, level
            )]

        # Layer 0: 정확한 검색
        candidates = self._search_layer(query, current_nearest, ef, 0)

        return candidates[:k]


# 사용 예제
hnsw = HNSWGraph(m=16, ef_construction=200)

# 벡터 추가
np.random.seed(42)
for i in range(1000):
    vector = np.random.randn(128)
    hnsw.insert(i, vector)

# 검색
query = np.random.randn(128)
results = hnsw.search(query, k=5, ef=50)

print("Top-5 nearest neighbors:")
for dist, node_id in results:
    print(f"  Node {node_id}: distance = {dist:.4f}")
```

**핵심 통찰**:

1. **계층 구조의 의미**:
   - 상위 레이어: "고속도로" (장거리 빠른 이동)
   - 하위 레이어: "골목길" (정확한 위치 찾기)
   - 검색은 고속도로 → 골목길 순으로 진행

2. **왜 O(log n)인가?**:
   - 레벨 수: E[max_level] ≈ log n
   - 각 레벨에서 탐색: 평균 O(1) ~ O(log log n)
   - 전체: O(log n)

3. **메모리 사용량**:
   - 각 노드당 평균 m개 연결
   - 전체: O(n × m × d) (d = 벡터 차원)
   - m=16, d=384 → 노드당 ~6KB

---

### Vector Quantization (벡터 양자화)

#### 왜 Quantization이 필요한가?

```python
"""
문제: 메모리 사용량

예: 100만 개 벡터, 차원 768 (BERT), Float32
= 1,000,000 × 768 × 4 bytes
= 3.072 GB (벡터만!)
+ HNSW 그래프 구조
= ~6-10 GB

해결: Quantization으로 메모리 1/4 ~ 1/32 감소!
"""
```

#### 1. Scalar Quantization (스칼라 양자화)

```python
class ScalarQuantizer:
    """
    Scalar Quantization: Float32 → Int8

    핵심 아이디어: 각 차원을 독립적으로 양자화

    압축률: 4배 (32bit → 8bit)
    정확도 손실: ~1-2%
    """

    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def fit(self, vectors):
        """
        벡터 범위 학습

        의도: 각 차원의 min, max 파악
        """
        vectors = np.array(vectors)
        self.min_vals = np.min(vectors, axis=0)  # 각 차원의 최소값
        self.max_vals = np.max(vectors, axis=0)  # 각 차원의 최대값

    def encode(self, vector):
        """
        Float32 → Uint8 변환

        수식: q = round((v - min) / (max - min) * 255)

        의도:
        - [min, max] 범위를 [0, 255]로 매핑
        - 8비트로 표현 가능
        """
        vector = np.array(vector)

        # Normalize to [0, 1]
        normalized = (vector - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)

        # Scale to [0, 255]
        quantized = np.round(normalized * 255).astype(np.uint8)

        return quantized

    def decode(self, quantized):
        """
        Uint8 → Float32 복원

        수식: v = q / 255 * (max - min) + min

        주의: 정확한 복원 불가능 (손실 압축)
        """
        quantized = np.array(quantized, dtype=np.float32)

        # [0, 255] → [0, 1]
        normalized = quantized / 255.0

        # [0, 1] → [min, max]
        reconstructed = normalized * (self.max_vals - self.min_vals) + self.min_vals

        return reconstructed

    def memory_usage(self, num_vectors, dim):
        """메모리 사용량 계산"""
        # Original: num_vectors × dim × 4 bytes (float32)
        original = num_vectors * dim * 4

        # Quantized: num_vectors × dim × 1 byte (uint8)
        quantized = num_vectors * dim * 1

        # Codebook: dim × 2 × 4 bytes (min, max per dimension)
        codebook = dim * 2 * 4

        return {
            'original_mb': original / (1024**2),
            'quantized_mb': (quantized + codebook) / (1024**2),
            'compression_ratio': original / (quantized + codebook)
        }


# 사용 예제
quantizer = ScalarQuantizer()

# 훈련 데이터로 범위 학습
train_vectors = np.random.randn(10000, 384)
quantizer.fit(train_vectors)

# 양자화
vector = np.random.randn(384)
quantized = quantizer.encode(vector)
reconstructed = quantizer.decode(quantized)

print(f"Original dtype: {vector.dtype}, size: {vector.nbytes} bytes")
print(f"Quantized dtype: {quantized.dtype}, size: {quantized.nbytes} bytes")
print(f"Reconstruction error: {np.mean((vector - reconstructed)**2):.6f}")

# 메모리 절감
usage = quantizer.memory_usage(1_000_000, 384)
print(f"\n1M vectors (dim=384):")
print(f"  Original: {usage['original_mb']:.2f} MB")
print(f"  Quantized: {usage['quantized_mb']:.2f} MB")
print(f"  Compression: {usage['compression_ratio']:.2f}x")
```

#### 2. Product Quantization (곱 양자화)

```python
class ProductQuantizer:
    """
    Product Quantization (PQ)

    핵심 아이디어: 벡터를 subvector로 분할 → 각각 클러스터링

    압축률: 8배, 16배, 32배 등 (설정 가능)
    정확도: Scalar Quantization보다 약간 낮음
    메모리: 매우 효율적
    """

    def __init__(self, n_subvectors=8, n_clusters=256):
        """
        Args:
            n_subvectors: 벡터를 몇 개로 분할? (보통 8, 16, 32)
            n_clusters: 각 subvector의 클러스터 수 (보통 256 = 2^8)

        의도:
        - n_subvectors가 클수록: 정확도 ↓, 속도 ↑
        - n_clusters가 클수록: 정확도 ↑, 메모리 ↑
        """
        self.n_subvectors = n_subvectors
        self.n_clusters = n_clusters
        self.codebooks = []  # 각 subvector의 클러스터 중심점

    def fit(self, vectors, iterations=20):
        """
        Codebook 학습 (K-Means)

        Process:
        1. 벡터를 n_subvectors개로 분할
        2. 각 subvector에 K-Means 적용
        3. 클러스터 중심점을 codebook으로 저장

        의도: 각 subvector를 cluster ID로 표현
        """
        vectors = np.array(vectors)
        n, d = vectors.shape

        assert d % self.n_subvectors == 0, "차원이 n_subvectors로 나누어떨어져야 함"

        subvector_dim = d // self.n_subvectors

        # 각 subvector에 대해 K-Means
        for i in range(self.n_subvectors):
            start = i * subvector_dim
            end = (i + 1) * subvector_dim

            # Subvector 추출
            subvectors = vectors[:, start:end]

            # K-Means clustering
            # 의도: subvector 공간을 n_clusters개 영역으로 분할
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, max_iter=iterations)
            kmeans.fit(subvectors)

            # Cluster 중심점 저장 (codebook)
            # 의도: 이 중심점들로 모든 subvector 근사
            self.codebooks.append(kmeans.cluster_centers_)

    def encode(self, vector):
        """
        벡터 → PQ codes

        Returns:
            codes: [c1, c2, ..., c_m] (각 subvector의 cluster ID)
            dtype: uint8 (0-255)

        의도: 벡터를 m개의 cluster ID로 표현
        """
        vector = np.array(vector)
        codes = []

        subvector_dim = len(vector) // self.n_subvectors

        for i in range(self.n_subvectors):
            start = i * subvector_dim
            end = (i + 1) * subvector_dim

            subvector = vector[start:end]

            # 가장 가까운 cluster 찾기
            # 의도: subvector를 codebook의 한 항목으로 근사
            distances = np.linalg.norm(
                self.codebooks[i] - subvector,
                axis=1
            )
            cluster_id = np.argmin(distances)

            codes.append(cluster_id)

        return np.array(codes, dtype=np.uint8)

    def decode(self, codes):
        """
        PQ codes → 벡터 복원

        의도: 각 code를 codebook에서 lookup
        """
        reconstructed = []

        for i, code in enumerate(codes):
            # Codebook에서 cluster 중심점 가져오기
            centroid = self.codebooks[i][code]
            reconstructed.append(centroid)

        return np.concatenate(reconstructed)

    def memory_usage(self, num_vectors, dim):
        """메모리 사용량 계산"""
        # Original
        original = num_vectors * dim * 4  # float32

        # PQ codes
        pq_codes = num_vectors * self.n_subvectors * 1  # uint8

        # Codebooks
        subvector_dim = dim // self.n_subvectors
        codebooks = self.n_subvectors * self.n_clusters * subvector_dim * 4  # float32

        return {
            'original_mb': original / (1024**2),
            'pq_mb': (pq_codes + codebooks) / (1024**2),
            'compression_ratio': original / (pq_codes + codebooks)
        }


# 사용 예제
pq = ProductQuantizer(n_subvectors=8, n_clusters=256)

# 훈련
train_vectors = np.random.randn(10000, 384)  # 384 = 8 × 48
pq.fit(train_vectors)

# 양자화
vector = np.random.randn(384)
codes = pq.encode(vector)
reconstructed = pq.decode(codes)

print(f"Original: {vector.nbytes} bytes")
print(f"PQ codes: {codes.nbytes} bytes (codes: {codes})")
print(f"Reconstruction error: {np.mean((vector - reconstructed)**2):.6f}")

# 메모리 절감
usage = pq.memory_usage(1_000_000, 384)
print(f"\n1M vectors (dim=384):")
print(f"  Original: {usage['original_mb']:.2f} MB")
print(f"  PQ: {usage['pq_mb']:.2f} MB")
print(f"  Compression: {usage['compression_ratio']:.2f}x")
```

**PQ의 핵심 통찰**:

```python
"""
왜 Product Quantization이 효과적인가?

수학적 설명:
- 벡터 공간 분할: d차원 → m개의 (d/m)차원
- 각 subspace에서 k-means
- Codebook 크기: m × k × (d/m) = k × d

메모리:
- Original: n × d × 4 bytes
- PQ codes: n × m × 1 byte (각 subvector의 cluster ID)
- Codebook: m × k × (d/m) × 4 bytes

예: n=1M, d=384, m=8, k=256
- Original: 1M × 384 × 4 = 1.5 GB
- PQ: 1M × 8 × 1 + 8 × 256 × 48 × 4 = 8 MB + 0.4 MB ≈ 8.4 MB
- 압축률: 178배!

정확도 손실:
- Recall@10: ~95-98% (적절한 k 선택 시)
"""
```

---

## 🗄️ Vector Database 아키텍처

### Vector DB의 핵심 기능

```python
"""
Vector Database 필수 기능:

1. CRUD Operations
   - Create: 벡터 추가
   - Read: 벡터 검색
   - Update: 벡터 수정
   - Delete: 벡터 삭제

2. Similarity Search
   - KNN (K-Nearest Neighbors)
   - Range search (반경 내 검색)
   - Batch search

3. Filtering
   - Metadata 기반 필터링
   - Hybrid search (벡터 + 키워드)

4. Scalability
   - Sharding (수평 확장)
   - Replication (고가용성)
   - Incremental indexing

5. Persistence
   - Disk 저장
   - Backup/Restore
"""
```

### 주요 Vector DB 비교

```
┌──────────────┬──────────┬──────────┬─────────┬───────────┐
│   Database   │  Speed   │  Scale   │ Feature │ Ease of Use│
├──────────────┼──────────┼──────────┼─────────┼───────────┤
│ Qdrant       │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐⭐  │
│ - Rust 기반  │          │          │ 필터링  │           │
│ - 고성능     │          │          │ 최고    │           │
├──────────────┼──────────┼──────────┼─────────┼───────────┤
│ Pinecone     │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐  │ ⭐⭐⭐⭐⭐  │
│ - Managed    │          │          │         │           │
│ - Serverless │          │          │         │           │
├──────────────┼──────────┼──────────┼─────────┼───────────┤
│ Weaviate     │ ⭐⭐⭐⭐  │ ⭐⭐⭐⭐  │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐   │
│ - GraphQL    │          │          │ 다양한  │           │
│ - 모듈형     │          │          │ 모듈    │           │
├──────────────┼──────────┼──────────┼─────────┼───────────┤
│ Milvus       │ ⭐⭐⭐⭐  │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐  │ ⭐⭐⭐     │
│ - 대규모     │          │          │         │           │
│ - 복잡한설정 │          │          │         │           │
├──────────────┼──────────┼──────────┼─────────┼───────────┤
│ Chroma       │ ⭐⭐⭐   │ ⭐⭐⭐   │ ⭐⭐⭐   │ ⭐⭐⭐⭐⭐  │
│ - 간단       │          │          │         │           │
│ - 임베디드   │          │          │         │           │
└──────────────┴──────────┴──────────┴─────────┴───────────┘

추천:
- 프로덕션: Qdrant, Pinecone
- 프로토타입: Chroma
- 대규모: Milvus
- 다양한 기능: Weaviate
```

---

### Qdrant 내부 아키텍처

#### 1. Storage Engine

```python
"""
Qdrant 스토리지 구조

┌─────────────────────────────────────────┐
│         Qdrant Server                   │
├─────────────────────────────────────────┤
│  API Layer (gRPC / REST)                │
├─────────────────────────────────────────┤
│  Collection Manager                     │
│  - Collection metadata                  │
│  - Shard management                     │
├─────────────────────────────────────────┤
│  Segment Manager                        │
│  - Mutable segments (in-memory)         │
│  - Immutable segments (on-disk)         │
├───────────────┬─────────────────────────┤
│  WAL (Write-  │  Payload Storage        │
│  Ahead Log)   │  (RocksDB)              │
│               │  - Metadata indexing    │
├───────────────┼─────────────────────────┤
│  Vector Index │  Vector Storage         │
│  (HNSW)       │  (Memory-mapped files)  │
└───────────────┴─────────────────────────┘
      ↓                 ↓
  WAL files      Data files (mmap)
"""

# Qdrant의 핵심 동작 원리 설명

class QdrantInternals:
    """
    Qdrant 내부 동작 원리 설명

    핵심 컴포넌트:
    1. WAL (Write-Ahead Log): 데이터 손실 방지
    2. Segments: 데이터 저장 단위
    3. HNSW Index: 벡터 검색
    4. RocksDB: Payload (메타데이터) 저장
    """

    def __init__(self):
        self.wal = WriteAheadLog()
        self.segments = SegmentManager()
        self.payload_storage = RocksDBStorage()

    class WriteAheadLog:
        """
        WAL (Write-Ahead Log)

        목적: 데이터 내구성 보장

        동작:
        1. Write 요청 → 먼저 WAL에 기록 (disk flush)
        2. WAL 기록 성공 → 클라이언트에 ACK
        3. 백그라운드에서 실제 인덱스에 적용

        이점:
        - 서버 크래시 시 WAL로부터 복구
        - 빠른 응답 (메모리 인덱스 업데이트 대기 불필요)

        의도: Postgres의 WAL과 동일한 개념
        """

        def write_operation(self, operation):
            """
            연산을 WAL에 기록

            연산 타입:
            - Insert: 새 벡터 추가
            - Update: 벡터 또는 payload 수정
            - Delete: 벡터 삭제
            """
            # 1. WAL에 시리얼라이즈하여 기록
            wal_entry = self.serialize(operation)

            # 2. Disk에 flush (fsync)
            # 의도: 전원이 꺼져도 데이터 보존
            self.append_to_wal(wal_entry)
            self.fsync()

            # 3. ACK 반환
            return "OK"

        def replay_wal(self):
            """
            서버 재시작 시 WAL replay

            의도: 크래시 복구
            """
            for entry in self.read_wal():
                operation = self.deserialize(entry)
                self.apply_to_index(operation)

    class SegmentManager:
        """
        Segment 관리

        Segment: 벡터 데이터의 독립적인 저장 단위

        타입:
        1. Mutable Segment (가변):
           - 메모리에 상주
           - 빠른 write
           - 작은 크기 (~10K-100K 벡터)

        2. Immutable Segment (불변):
           - 디스크에 저장 (memory-mapped)
           - Read-only
           - 큰 크기 (~100K-1M 벡터)
           - HNSW 인덱스 최적화됨

        동작:
        - Write → Mutable segment에 추가
        - Mutable segment가 가득 차면 → Immutable로 변환 (flush)
        - Background compaction: 작은 immutable segments 병합
        """

        def __init__(self):
            self.mutable_segments = []
            self.immutable_segments = []
            self.segment_threshold = 100_000  # 벡터 수

        def insert_vector(self, vector_id, vector, payload):
            """
            벡터 삽입

            의도:
            1. 먼저 mutable segment에 추가 (빠름)
            2. 주기적으로 immutable로 변환 (최적화)
            """
            # 현재 mutable segment 가져오기 또는 생성
            if not self.mutable_segments:
                self.mutable_segments.append(MutableSegment())

            current = self.mutable_segments[-1]

            # 삽입
            current.insert(vector_id, vector, payload)

            # Threshold 초과 시 flush
            if current.size() >= self.segment_threshold:
                self.flush_to_immutable(current)

        def flush_to_immutable(self, mutable_segment):
            """
            Mutable → Immutable 변환

            Process:
            1. HNSW 인덱스 구축 (완전한 그래프)
            2. 벡터 데이터를 memory-mapped file로 저장
            3. Payload를 RocksDB에 저장
            4. Segment를 read-only로 마킹

            의도: 검색 성능 최적화
            """
            # 1. HNSW 인덱스 빌드
            hnsw = self.build_hnsw_index(mutable_segment.vectors)

            # 2. 디스크에 저장
            segment_file = self.allocate_segment_file()
            self.write_vectors_mmap(segment_file, mutable_segment.vectors)
            self.write_hnsw_graph(segment_file, hnsw)

            # 3. Immutable segment 생성
            immutable = ImmutableSegment(segment_file, hnsw)
            self.immutable_segments.append(immutable)

            # 4. Mutable segment 제거
            self.mutable_segments.remove(mutable_segment)

        def search(self, query_vector, top_k):
            """
            전체 segment에서 검색

            알고리즘:
            1. 각 segment에서 독립적으로 검색
            2. 결과 병합 (merge)
            3. Top-K 선택

            의도: Segment 단위로 병렬 검색 가능
            """
            all_results = []

            # 각 segment에서 검색
            for segment in self.mutable_segments + self.immutable_segments:
                results = segment.search(query_vector, top_k)
                all_results.extend(results)

            # 병합 및 정렬
            all_results.sort(key=lambda x: x.distance)

            return all_results[:top_k]

        def optimize_segments(self):
            """
            백그라운드 Compaction

            목적:
            - 작은 segment들을 큰 segment로 병합
            - 삭제된 벡터 정리 (garbage collection)
            - HNSW 그래프 최적화

            트리거:
            - Segment 수가 많아질 때
            - 삭제 비율이 높을 때
            - 주기적 (예: 매 1시간)
            """
            # Small segments 병합
            small_segments = [s for s in self.immutable_segments if s.size() < 50_000]

            if len(small_segments) >= 3:
                merged = self.merge_segments(small_segments)
                for s in small_segments:
                    self.immutable_segments.remove(s)
                self.immutable_segments.append(merged)

    class RocksDBStorage:
        """
        Payload 저장소 (RocksDB)

        목적: 벡터의 메타데이터 (payload) 저장 및 인덱싱

        RocksDB 선택 이유:
        - Key-Value store (빠른 lookup)
        - LSM-Tree (Log-Structured Merge Tree)
        - 압축 지원
        - Range scan 지원

        구조:
        Key: vector_id
        Value: {
            "text": "...",
            "category": "...",
            "author": "...",
            ...
        }

        인덱싱:
        - Secondary index로 필터링 지원
        - 예: category="AI" 인 벡터만 검색
        """

        def __init__(self):
            import rocksdb  # PyRocksDB
            self.db = rocksdb.DB("payload.db", rocksdb.Options(create_if_missing=True))

            # Secondary indexes
            self.indexes = {}  # field → {value → [vector_ids]}

        def store_payload(self, vector_id, payload):
            """
            Payload 저장

            의도:
            1. Primary storage (RocksDB)
            2. Secondary indexes 업데이트
            """
            import json

            # 1. Primary storage
            self.db.put(
                vector_id.encode(),
                json.dumps(payload).encode()
            )

            # 2. Secondary indexes
            for field, value in payload.items():
                if field not in self.indexes:
                    self.indexes[field] = {}

                if value not in self.indexes[field]:
                    self.indexes[field][value] = set()

                self.indexes[field][value].add(vector_id)

        def get_payload(self, vector_id):
            """Payload 조회"""
            import json
            data = self.db.get(vector_id.encode())
            return json.loads(data.decode()) if data else None

        def filter_by_field(self, field, value):
            """
            필터링

            의도: Vector search 전에 후보 줄이기
            """
            if field in self.indexes and value in self.indexes[field]:
                return list(self.indexes[field][value])
            return []


# 실제 Qdrant의 검색 과정

def qdrant_search_process(query_vector, filter_conditions, top_k):
    """
    Qdrant 검색의 내부 동작

    Process:
    1. Filter 평가 (Payload index)
    2. 각 Segment에서 Vector search
    3. 결과 병합
    4. Top-K 선택
    """

    # Step 1: Filter로 후보 벡터 ID 찾기
    # 의도: 불필요한 벡터 검색 제외
    candidate_ids = set()

    if filter_conditions:
        # Payload index 사용
        for condition in filter_conditions:
            field = condition['field']
            value = condition['value']
            ids = payload_storage.filter_by_field(field, value)
            candidate_ids.update(ids)
    else:
        candidate_ids = None  # 모든 벡터 검색

    # Step 2: 각 segment에서 검색
    results = []

    for segment in all_segments:
        # Segment 내에서 HNSW search
        segment_results = segment.hnsw_search(
            query_vector,
            top_k=top_k * 2,  # Over-fetch for merging
            filter_ids=candidate_ids
        )
        results.extend(segment_results)

    # Step 3: 병합 및 정렬
    results.sort(key=lambda x: x.distance)

    # Step 4: Top-K
    return results[:top_k]


"""
성능 특성:

1. Write 성능:
   - WAL: O(1) append
   - Mutable segment: O(log n) insert
   - 전체: 수천 QPS

2. Read 성능:
   - HNSW: O(log n) search
   - Segment 병렬 검색
   - 전체: 수만 QPS

3. Memory-mapped I/O:
   - OS page cache 활용
   - 자주 접근하는 벡터만 메모리에
   - 대규모 데이터셋 지원 (RAM > 데이터 크기 불필요)

4. Compaction:
   - 백그라운드 동작
   - 검색 성능에 영향 최소화
   - LSM-tree와 유사
"""
```

---

## 🚀 Qdrant 완전 가이드

### 설치 및 시작

```bash
# Docker로 Qdrant 실행
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

# Python client 설치
pip install qdrant-client sentence-transformers
```

### 기본 사용법

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# 1. Qdrant 클라이언트 생성
# 의도: Vector DB에 연결
client = QdrantClient(host="localhost", port=6333)

# 2. Embedding 모델
# 의도: 텍스트를 벡터로 변환할 모델
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 3. Collection 생성
# 의도: 벡터를 저장할 "테이블" 생성
collection_name = "my_documents"

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,  # 벡터 차원 (모델에 따라 다름)
        distance=Distance.COSINE  # 거리 측정 방법
    )
)

print(f"Collection '{collection_name}' created!")
```

### 데이터 추가 (Create)

```python
# 문서 준비
documents = [
    {
        "id": str(uuid.uuid4()),
        "text": "Qdrant는 고성능 벡터 데이터베이스입니다.",
        "metadata": {"category": "기술", "author": "Alice"}
    },
    {
        "id": str(uuid.uuid4()),
        "text": "Python은 데이터 과학에 최적화된 언어입니다.",
        "metadata": {"category": "프로그래밍", "author": "Bob"}
    },
    {
        "id": str(uuid.uuid4()),
        "text": "Machine Learning은 AI의 핵심 분야입니다.",
        "metadata": {"category": "AI", "author": "Alice"}
    }
]

# 벡터 생성 및 업로드
# 의도: 문서를 벡터로 변환 후 DB에 저장
points = []
for doc in documents:
    # 텍스트 → 벡터
    vector = encoder.encode(doc["text"]).tolist()

    # Point 생성 (ID + Vector + Metadata)
    # 의도: 벡터와 메타데이터를 함께 저장
    point = PointStruct(
        id=doc["id"],
        vector=vector,
        payload={
            "text": doc["text"],
            "category": doc["metadata"]["category"],
            "author": doc["metadata"]["author"]
        }
    )
    points.append(point)

# Batch upload
# 의도: 여러 벡터를 한 번에 업로드 (효율성)
client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"{len(points)} documents uploaded!")
```

### 검색 (Read)

```python
# 기본 검색
def search_documents(query_text, top_k=3):
    """
    Semantic search: 의미 기반 문서 검색

    Args:
        query_text: 검색 쿼리
        top_k: 반환할 결과 수

    Returns:
        검색 결과 리스트
    """
    # 쿼리 텍스트를 벡터로 변환
    # 의도: 검색어를 문서와 같은 벡터 공간에 매핑
    query_vector = encoder.encode(query_text).tolist()

    # Vector search
    # 의도: 쿼리 벡터와 가장 유사한 벡터 찾기
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    return search_result


# 검색 실행
query = "데이터베이스 기술"
results = search_documents(query, top_k=3)

print(f"Query: '{query}'\n")
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"  Score: {result.score:.4f}")
    print(f"  Text: {result.payload['text']}")
    print(f"  Category: {result.payload['category']}")
    print()

# Output:
# Query: '데이터베이스 기술'
#
# Result 1:
#   Score: 0.7234
#   Text: Qdrant는 고성능 벡터 데이터베이스입니다.
#   Category: 기술
```

### 필터링 검색

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

def search_with_filter(query_text, category=None, author=None, top_k=3):
    """
    Filtered vector search

    핵심: 벡터 유사도 + 메타데이터 필터링
    의도: "Alice가 쓴 AI 관련 문서" 같은 복합 검색
    """
    query_vector = encoder.encode(query_text).tolist()

    # 필터 조건 생성
    # 의도: SQL의 WHERE 절과 유사
    filter_conditions = []

    if category:
        filter_conditions.append(
            FieldCondition(
                key="category",
                match=MatchValue(value=category)
            )
        )

    if author:
        filter_conditions.append(
            FieldCondition(
                key="author",
                match=MatchValue(value=author)
            )
        )

    # 검색 (벡터 + 필터)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=Filter(must=filter_conditions) if filter_conditions else None,
        limit=top_k
    )

    return search_result


# 필터링 검색
results = search_with_filter(
    query_text="기술에 대해",
    author="Alice",  # Alice가 쓴 문서만
    top_k=5
)

print(f"Alice가 쓴 문서 중 '기술'과 관련된 것:")
for result in results:
    print(f"  - {result.payload['text']}")
```

### 업데이트 및 삭제

```python
# 특정 문서 업데이트
def update_document(doc_id, new_text=None, new_metadata=None):
    """
    문서 업데이트

    의도: 기존 문서의 내용이나 메타데이터 수정
    """
    if new_text:
        # 새 텍스트로 벡터 재생성
        new_vector = encoder.encode(new_text).tolist()

        # Payload 업데이트
        payload = {"text": new_text}
        if new_metadata:
            payload.update(new_metadata)

        # Upsert (update + insert)
        # 의도: 있으면 업데이트, 없으면 삽입
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=doc_id,
                    vector=new_vector,
                    payload=payload
                )
            ]
        )


# 문서 삭제
def delete_document(doc_id):
    """
    문서 삭제

    의도: 특정 ID의 문서를 DB에서 제거
    """
    client.delete(
        collection_name=collection_name,
        points_selector=[doc_id]
    )


# 조건부 삭제
def delete_by_filter(category):
    """
    필터 조건으로 삭제

    의도: 특정 카테고리의 모든 문서 삭제
    """
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category)
                )
            ]
        )
    )
```

---

### Embedding 모델 내부 동작

#### Sentence Transformers 아키텍처

```python
"""
Sentence Transformer 내부 구조

텍스트: "강아지가 좋아요"
  ↓
Tokenizer: ["강", "##아지", "##가", "좋", "##아", "##요"]
  ↓
Token IDs: [4521, 2341, 1234, 8765, 3421, 5678]
  ↓
Embedding Layer: (6, 768) - 각 token을 768차원 벡터로
  ↓
Transformer Layers (12층):
  - Self-Attention: token 간 관계 학습
  - Feed-Forward: 비선형 변환
  ↓
Token embeddings: (6, 768)
  ↓
Pooling (Mean/CLS):
  - Mean: 모든 token의 평균
  - CLS: [CLS] token만 사용
  ↓
Sentence embedding: (768,) 또는 (384,) 또는 (1024,)
  ↓
L2 Normalization (선택): 벡터를 단위 벡터로
  ↓
Final embedding: (384,)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SentenceTransformerInternals:
    """
    Sentence Transformer 내부 구현 설명

    실제로 sentence-transformers 라이브러리가 하는 일
    """

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        모델 로드

        all-MiniLM-L6-v2:
        - Base: BERT-like transformer
        - Layers: 6 (L6)
        - Hidden: 384
        - Parameters: 22M
        - 훈련: Contrastive learning (유사한 문장끼리 가깝게)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode_detailed(self, text):
        """
        인코딩 과정을 단계별로 설명

        의도: 각 단계에서 무슨 일이 일어나는지 이해
        """

        # ===== Step 1: Tokenization =====
        # 의도: 텍스트를 모델이 이해할 수 있는 숫자로 변환
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        print(f"Input text: {text}")
        print(f"Token IDs: {tokens['input_ids']}")
        print(f"Token IDs shape: {tokens['input_ids'].shape}")  # (1, seq_len)

        # ===== Step 2: Transformer Forward Pass =====
        # 의도: Token 간 관계를 학습하여 contextual embedding 생성
        with torch.no_grad():
            outputs = self.model(**tokens)

        # outputs.last_hidden_state: (batch, seq_len, hidden_dim)
        # 각 token의 contextualized embedding
        token_embeddings = outputs.last_hidden_state
        print(f"Token embeddings shape: {token_embeddings.shape}")  # (1, seq_len, 384)

        # ===== Step 3: Pooling =====
        # 의도: 여러 token embedding을 하나의 sentence embedding으로 집약

        # 방법 1: Mean Pooling (가장 일반적)
        # 모든 token의 평균 (padding 제외)
        attention_mask = tokens['attention_mask']  # (1, seq_len)

        # Attention mask 확장: (1, seq_len) → (1, seq_len, 384)
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Token embeddings에 mask 적용 후 sum
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)

        # 실제 token 수로 나누기 (평균)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        print(f"Mean pooled shape: {mean_pooled.shape}")  # (1, 384)

        # 방법 2: CLS Token Pooling
        # [CLS] token (첫 번째 token)만 사용
        cls_pooled = token_embeddings[:, 0, :]
        print(f"CLS pooled shape: {cls_pooled.shape}")  # (1, 384)

        # 방법 3: Max Pooling
        # 각 차원에서 최대값 선택
        max_pooled = torch.max(token_embeddings, dim=1)[0]
        print(f"Max pooled shape: {max_pooled.shape}")  # (1, 384)

        # ===== Step 4: Normalization =====
        # 의도: Cosine similarity 사용 시 크기 통일
        sentence_embedding = mean_pooled

        # L2 normalization: 벡터를 단위 벡터로
        # 의도: 벡터의 방향만 중요, 크기 무시
        normalized = nn.functional.normalize(sentence_embedding, p=2, dim=1)

        print(f"Final embedding shape: {normalized.shape}")  # (1, 384)
        print(f"Embedding norm: {torch.norm(normalized, p=2, dim=1)}")  # ~1.0

        return normalized.squeeze(0).numpy()

    def why_contrastive_learning(self):
        """
        Sentence Transformer 훈련 방법

        핵심: Contrastive Learning (대조 학습)

        데이터:
        - Positive pairs: (문장1, 문장2) - 의미가 유사
        - Negative pairs: (문장1, 문장3) - 의미가 다름

        예:
        Anchor:   "강아지가 귀여워요"
        Positive: "개가 예뻐요"        (유사)
        Negative: "자동차가 빨라요"    (다름)

        Loss: Triplet Loss 또는 Contrastive Loss

        목표:
        - anchor와 positive의 거리 ↓
        - anchor와 negative의 거리 ↑
        """

        # Triplet Loss
        def triplet_loss(anchor, positive, negative, margin=0.5):
            """
            Triplet Loss

            수식: max(0, d(a,p) - d(a,n) + margin)

            의도:
            - d(a,p): anchor와 positive 거리 → 작게
            - d(a,n): anchor와 negative 거리 → 크게
            - margin: 최소 거리 차이
            """
            d_pos = torch.norm(anchor - positive, p=2)
            d_neg = torch.norm(anchor - negative, p=2)

            loss = torch.clamp(d_pos - d_neg + margin, min=0.0)

            return loss

        # 예제
        anchor = torch.randn(384)
        positive = anchor + 0.1 * torch.randn(384)  # 유사
        negative = torch.randn(384)  # 다름

        loss = triplet_loss(anchor, positive, negative)
        print(f"Triplet loss: {loss.item():.4f}")

        """
        실제 훈련:
        1. 대규모 데이터셋 (NLI, STS, QA pairs)
        2. Hard negative mining (어려운 negative 선택)
        3. Batch 내 negative 활용
        4. 수백만 ~ 수십억 pair로 훈련

        결과:
        - 의미가 유사한 문장은 벡터 공간에서 가까이
        - Cross-lingual도 가능 (다국어 학습 시)
        """


# 사용 예제
st = SentenceTransformerInternals()

text = "강아지가 좋아요"
embedding = st.encode_detailed(text)

print(f"\nFinal embedding (first 10 dims): {embedding[:10]}")
```

#### Embedding 품질 향상 기법

```python
class EmbeddingQuality:
    """
    Embedding 품질을 높이는 기법들

    의도: 더 나은 semantic search를 위해
    """

    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def domain_adaptation(self, domain_texts):
        """
        Domain Adaptation

        문제: 일반 도메인으로 훈련된 모델은 특정 도메인에서 성능 저하

        예:
        - 의료 문서: "MI"는 "Myocardial Infarction" (심근경색)
        - 일반 문서: "MI"는 "Michigan" 또는 "Military Intelligence"

        해결: Domain-specific data로 fine-tuning
        """

        # 1. Domain corpus에서 positive pairs 생성
        # 방법: 같은 문서의 문장들은 유사
        pairs = []
        for doc in domain_texts:
            sentences = doc.split('.')
            for i in range(len(sentences) - 1):
                pairs.append((sentences[i], sentences[i+1]))

        # 2. Contrastive learning으로 fine-tuning
        # (실제 구현은 sentence-transformers 라이브러리 사용)

        return "Domain-adapted model"

    def query_document_asymmetry(self):
        """
        Query-Document Asymmetry 처리

        문제:
        - Query: "강아지 사료 추천" (짧음, 질문)
        - Document: "저희 회사는 프리미엄 강아지 사료를 판매합니다..." (길음, 설명)

        → 같은 embedding 모델로는 매칭 어려움

        해결: Asymmetric model
        - Query encoder: 짧은 텍스트 특화
        - Document encoder: 긴 텍스트 특화
        """

        # 예: DPR (Dense Passage Retrieval)
        query = "강아지 사료"
        document = "프리미엄 강아지 사료를 판매합니다. 영양소가 풍부하고..."

        # 서로 다른 encoder 사용
        query_emb = query_encoder(query)      # 질문 특화
        doc_emb = document_encoder(document)   # 문서 특화

        # Cosine similarity로 매칭
        similarity = cosine_similarity(query_emb, doc_emb)

        return similarity

    def hard_negative_mining(self):
        """
        Hard Negative Mining

        목적: 더 어려운 negative로 훈련하여 성능 향상

        예:
        Query: "강아지 사료 추천"

        Easy negative: "자동차 정비" (너무 다름)
        Hard negative: "고양이 사료 추천" (비슷하지만 다름)

        의도: Hard negative로 훈련하면 미묘한 차이를 구분
        """

        # 1. Batch 내 in-batch negatives
        # (A, B, C) 3개 문장 있으면
        # A-B positive, A-C negative, B-A positive, B-C negative, ...

        # 2. Top-K retrieval negatives
        # 검색 결과 상위이지만 실제로는 관련 없는 것들

        return "Improved model"


"""
Embedding 모델 선택 가이드:

1. all-MiniLM-L6-v2 (384 dim, 22M params)
   - 용도: 일반 목적, 빠른 속도
   - 성능: 좋음
   - 속도: 매우 빠름

2. all-mpnet-base-v2 (768 dim, 110M params)
   - 용도: 높은 정확도 필요
   - 성능: 최고
   - 속도: 중간

3. multi-qa-MiniLM-L6-cos-v1 (384 dim)
   - 용도: QA pairs (question-answer)
   - 성능: QA 특화
   - 속도: 빠름

4. paraphrase-multilingual-MiniLM-L12-v2
   - 용도: 다국어 지원
   - 성능: 50+ languages
   - 속도: 중간

선택 기준:
- 속도 중요 → MiniLM
- 정확도 중요 → mpnet
- QA → multi-qa 모델
- 다국어 → multilingual 모델
"""
```

---

## 🤖 RAG (Retrieval-Augmented Generation)

### RAG란?

**"LLM에게 관련 문서를 제공하여 더 정확한 답변 생성"**

```
전통적 LLM:
  User Question → LLM → Answer

  문제:
  - 훈련 데이터에 없는 정보는 모름
  - 최신 정보 부족
  - Hallucination (잘못된 정보 생성)

RAG:
  User Question
    ↓
  Vector DB Search (관련 문서 검색)
    ↓
  LLM (문서 + 질문) → Answer

  장점:
  - 최신 정보 활용
  - 특정 도메인 지식 활용
  - Hallucination 감소
  - 출처 제공 가능
```

### RAG 구현 (기본)

```python
from openai import OpenAI

# OpenAI client (또는 다른 LLM)
openai_client = OpenAI(api_key="your-api-key")

def rag_query(question, top_k=3):
    """
    RAG: Retrieval-Augmented Generation

    Process:
    1. 질문을 벡터로 변환
    2. 관련 문서 검색 (Vector DB)
    3. 문서 + 질문을 LLM에 전달
    4. LLM이 문서 기반으로 답변 생성

    의도: LLM의 답변에 외부 지식 주입
    """

    # Step 1: Vector search로 관련 문서 검색
    # 의도: 질문과 관련된 context 찾기
    query_vector = encoder.encode(question).tolist()

    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    # Step 2: 검색된 문서를 context로 결합
    # 의도: LLM에게 제공할 배경 지식
    context = "\n\n".join([
        f"Document {i+1}: {result.payload['text']}"
        for i, result in enumerate(search_results)
    ])

    # Step 3: Prompt 구성
    # 의도: 문서를 먼저 제공하고, 그 기반으로 답변하도록 유도
    prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

[관련 문서]
{context}

[질문]
{question}

[답변]
위 문서들을 바탕으로 정확하고 구체적으로 답변해주세요.
문서에 없는 내용은 추측하지 말고 "문서에서 확인할 수 없습니다"라고 말해주세요.
"""

    # Step 4: LLM으로 답변 생성
    # 의도: 문서 기반 답변 생성
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 주어진 문서를 기반으로 정확하게 답변하는 도우미입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3  # 낮은 temperature로 factual한 답변 유도
    )

    answer = response.choices[0].message.content

    # 출처 함께 반환
    # 의도: 답변의 근거 제공 (신뢰성 향상)
    return {
        "answer": answer,
        "sources": [
            {
                "text": result.payload['text'],
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k != 'text'}
            }
            for result in search_results
        ]
    }


# RAG 실행
question = "벡터 데이터베이스란 무엇인가요?"
result = rag_query(question, top_k=2)

print(f"Question: {question}\n")
print(f"Answer: {result['answer']}\n")
print("Sources:")
for i, source in enumerate(result['sources']):
    print(f"  {i+1}. {source['text']} (score: {source['score']:.4f})")
```

### 고급 RAG 패턴

```python
class AdvancedRAG:
    """
    Advanced RAG with multiple improvements

    개선사항:
    1. Reranking: 검색 결과 재정렬
    2. Hybrid Search: 벡터 + 키워드 검색 결합
    3. Query Rewriting: 질문 개선
    4. Citation: 출처 명시
    """

    def __init__(self, vector_db_client, llm_client, encoder):
        self.vdb = vector_db_client
        self.llm = llm_client
        self.encoder = encoder

    def rewrite_query(self, query):
        """
        Query Rewriting: 질문을 검색에 최적화

        예:
        - "그거 뭐야?" → "Qdrant의 기능은 무엇인가요?"
        - "어떻게 써?" → "Qdrant 사용 방법을 알려주세요"

        의도: 모호한 질문을 구체적으로 만들어 검색 품질 향상
        """
        prompt = f"""다음 질문을 더 구체적이고 검색에 적합하게 다시 작성해주세요.
단, 원래 의미는 유지하세요.

원래 질문: {query}
개선된 질문:"""

        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )

        return response.choices[0].message.content.strip()

    def hybrid_search(self, query, top_k=10):
        """
        Hybrid Search: Vector + Keyword 검색 결합

        의도: 의미 검색과 정확한 키워드 매칭을 모두 활용
        """
        # 1. Vector search
        query_vector = self.encoder.encode(query).tolist()
        vector_results = self.vdb.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        # 2. Keyword search (full-text search)
        # Qdrant의 full-text search 기능 사용
        from qdrant_client.models import FieldCondition, MatchText

        keyword_results = self.vdb.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="text",
                        match=MatchText(text=query)
                    )
                ]
            ),
            limit=top_k
        )

        # 3. 결과 결합 및 중복 제거
        # 의도: 벡터와 키워드 검색 결과를 합쳐 더 포괄적인 결과 제공
        combined = {}

        for result in vector_results:
            combined[result.id] = {
                'vector_score': result.score,
                'keyword_score': 0,
                'payload': result.payload
            }

        for point, _ in keyword_results:
            if point.id in combined:
                combined[point.id]['keyword_score'] = 1.0
            else:
                combined[point.id] = {
                    'vector_score': 0,
                    'keyword_score': 1.0,
                    'payload': point.payload
                }

        # 4. Hybrid score 계산
        # 의도: 벡터와 키워드 점수를 결합하여 최종 순위 결정
        for item in combined.values():
            item['hybrid_score'] = (
                0.7 * item['vector_score'] +  # 벡터 검색에 70% 가중치
                0.3 * item['keyword_score']    # 키워드 검색에 30% 가중치
            )

        # 정렬
        sorted_results = sorted(
            combined.items(),
            key=lambda x: x[1]['hybrid_score'],
            reverse=True
        )[:top_k]

        return sorted_results

    def rerank(self, query, documents, top_k=3):
        """
        Reranking: LLM으로 검색 결과 재정렬

        의도: 초기 검색 결과를 LLM이 다시 평가하여 품질 향상
        """
        # LLM에게 각 문서의 relevance 평가 요청
        reranked = []

        for doc_id, doc_info in documents[:10]:  # 상위 10개만 rerank
            prompt = f"""질문: {query}
문서: {doc_info['payload']['text']}

이 문서가 질문에 답변하는데 얼마나 관련있나요?
0-10 사이의 점수로만 답변하세요. (숫자만)"""

            response = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )

            try:
                relevance_score = float(response.choices[0].message.content.strip())
            except:
                relevance_score = doc_info['hybrid_score'] * 10

            reranked.append((doc_id, doc_info, relevance_score))

        # Relevance score로 재정렬
        reranked.sort(key=lambda x: x[2], reverse=True)

        return reranked[:top_k]

    def generate_with_citations(self, query, top_k=3):
        """
        답변 + 출처 명시

        의도: 답변의 각 부분이 어느 문서에서 왔는지 명확히 표시
        """
        # 1. Query rewriting
        improved_query = self.rewrite_query(query)

        # 2. Hybrid search
        search_results = self.hybrid_search(improved_query, top_k=10)

        # 3. Reranking
        reranked = self.rerank(improved_query, search_results, top_k=top_k)

        # 4. Context 생성 (출처 번호 포함)
        context = ""
        sources = []

        for i, (doc_id, doc_info, score) in enumerate(reranked):
            context += f"\n[{i+1}] {doc_info['payload']['text']}\n"
            sources.append({
                'id': doc_id,
                'text': doc_info['payload']['text'],
                'score': score
            })

        # 5. Prompt (citation 요청)
        prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.
각 정보의 출처를 [번호] 형식으로 명시해주세요.

{context}

질문: {query}

답변 (출처 번호 포함):"""

        # 6. LLM 생성
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "주어진 문서를 기반으로 답변하고, 각 정보의 출처를 [번호]로 명시하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content

        return {
            'original_query': query,
            'improved_query': improved_query,
            'answer': answer,
            'sources': sources
        }


# 사용 예제
rag = AdvancedRAG(client, openai_client, encoder)

result = rag.generate_with_citations("벡터 DB 어떻게 써?")

print(f"Original Query: {result['original_query']}")
print(f"Improved Query: {result['improved_query']}\n")
print(f"Answer:\n{result['answer']}\n")
print("Sources:")
for i, source in enumerate(result['sources']):
    print(f"  [{i+1}] {source['text']} (relevance: {source['score']:.2f})")
```

---

## 💻 실전 예제: 문서 챗봇

```python
import os
from pathlib import Path

class DocumentChatbot:
    """
    문서 기반 챗봇

    기능:
    1. 문서 로딩 (PDF, TXT, Markdown 등)
    2. Chunking (문서를 작은 조각으로 분할)
    3. Embedding & Indexing
    4. RAG 기반 질의응답
    """

    def __init__(self, collection_name="documents"):
        self.collection_name = collection_name
        self.client = QdrantClient(host="localhost", port=6333)
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Collection 생성
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        except:
            pass  # 이미 존재하면 무시

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """
        텍스트를 작은 chunk로 분할

        의도: 긴 문서를 작은 단위로 나눠 검색 정확도 향상

        Args:
            chunk_size: 각 chunk의 글자 수
            overlap: chunk 간 중복 글자 수 (문맥 유지)
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # 문장 중간에서 자르지 않도록
            # 의도: 의미 단위 유지
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')

                cut_point = max(last_period, last_newline)
                if cut_point > chunk_size * 0.5:  # 너무 짧지 않으면
                    end = start + cut_point + 1
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start = end - overlap  # Overlap

        return chunks

    def load_document(self, file_path, metadata=None):
        """
        문서 로딩 및 인덱싱

        Process:
        1. 파일 읽기
        2. Chunking
        3. Embedding
        4. Vector DB에 저장
        """
        # 1. 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 2. Chunking
        chunks = self.chunk_text(text)

        # 3. Embedding & Upload
        points = []

        for i, chunk in enumerate(chunks):
            vector = self.encoder.encode(chunk).tolist()

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "source": file_path,
                    "chunk_id": i,
                    **(metadata or {})
                }
            )
            points.append(point)

        # Batch upload
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Loaded {len(chunks)} chunks from {file_path}")

        return len(chunks)

    def load_directory(self, directory_path):
        """
        디렉터리의 모든 문서 로딩

        의도: 여러 파일을 한 번에 인덱싱
        """
        total_chunks = 0

        for file_path in Path(directory_path).rglob("*.txt"):
            chunks = self.load_document(
                str(file_path),
                metadata={"filename": file_path.name}
            )
            total_chunks += chunks

        print(f"Total: {total_chunks} chunks from directory")

        return total_chunks

    def chat(self, question, conversation_history=None):
        """
        대화형 질의응답

        의도: 이전 대화 맥락을 고려한 답변
        """
        # 1. 관련 문서 검색
        query_vector = self.encoder.encode(question).tolist()

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=5
        )

        # 2. Context 구성
        context = "\n\n".join([
            result.payload['text']
            for result in search_results
        ])

        # 3. 대화 기록 포함
        messages = [
            {"role": "system", "content": "당신은 주어진 문서를 기반으로 정확하게 답변하는 도우미입니다."}
        ]

        if conversation_history:
            messages.extend(conversation_history)

        # 4. 현재 질문 + context
        user_message = f"""[관련 문서]
{context}

[질문]
{question}"""

        messages.append({"role": "user", "content": user_message})

        # 5. LLM 응답
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5
        )

        answer = response.choices[0].message.content

        return {
            'answer': answer,
            'sources': [
                {
                    'text': result.payload['text'],
                    'source': result.payload['source'],
                    'score': result.score
                }
                for result in search_results
            ]
        }


# 사용 예제
chatbot = DocumentChatbot(collection_name="my_docs")

# 문서 로딩
chatbot.load_directory("./documents")

# 질의응답
conversation = []

question1 = "이 문서에서 주요 개념은 무엇인가요?"
response1 = chatbot.chat(question1, conversation)
print(f"Q: {question1}")
print(f"A: {response1['answer']}\n")

# 대화 기록 업데이트
conversation.append({"role": "user", "content": question1})
conversation.append({"role": "assistant", "content": response1['answer']})

# 후속 질문 (이전 맥락 고려)
question2 = "그것에 대해 더 자세히 설명해주세요."
response2 = chatbot.chat(question2, conversation)
print(f"Q: {question2}")
print(f"A: {response2['answer']}")
```

---

## 📊 성능 최적화

### Indexing 최적화

```python
# 1. Batch size 조정
# 의도: 한 번에 많은 벡터를 업로드하여 네트워크 오버헤드 감소
BATCH_SIZE = 1000

def batch_upload(points, batch_size=BATCH_SIZE):
    """대량 데이터 효율적 업로드"""
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"Uploaded batch {i//batch_size + 1}")


# 2. Parallel indexing
# 의도: 여러 스레드로 embedding 생성 속도 향상
from concurrent.futures import ThreadPoolExecutor

def parallel_encode(texts, num_workers=4):
    """병렬 encoding"""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        vectors = list(executor.map(encoder.encode, texts))
    return vectors
```

### 검색 최적화

```python
# 1. HNSW 파라미터 튜닝
from qdrant_client.models import HnswConfigDiff

client.update_collection(
    collection_name=collection_name,
    hnsw_config=HnswConfigDiff(
        m=16,              # 연결 수 (높을수록 정확, 메모리 많이 사용)
        ef_construct=100,  # 인덱싱 정확도 (높을수록 느림, 정확)
    )
)

# 2. Search 파라미터
search_results = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=10,
    search_params={
        "hnsw_ef": 128,    # 검색 정확도 (높을수록 느림, 정확)
        "exact": False     # True면 완전 탐색 (느리지만 100% 정확)
    }
)
```

---

## 📝 Summary

### Key Takeaways

1. **Vector Embeddings**
   - 텍스트 → 고차원 벡터
   - 의미가 유사하면 벡터도 유사
   - Cosine similarity로 유사도 측정

2. **Vector Database**
   - ANN (Approximate Nearest Neighbor)
   - HNSW: O(log n) 검색
   - Qdrant: 고성능, 풍부한 필터링

3. **RAG (Retrieval-Augmented Generation)**
   - Vector search로 관련 문서 검색
   - LLM에게 context 제공
   - Hallucination 감소, 최신 정보 활용

4. **Advanced Patterns**
   - Query rewriting
   - Hybrid search (vector + keyword)
   - Reranking
   - Citation

5. **실전 팁**:
   - Chunking: 500-1000 글자, 50-100 overlap
   - Top-k: 3-5개 문서
   - Temperature: 0.3-0.5 (factual 답변)
   - Metadata filtering 적극 활용

---

## 📚 References

**Papers**:
1. **REALM**: Retrieval-Augmented Language Model Pre-Training
2. **RAG**: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
3. **HNSW**: Efficient and robust approximate nearest neighbor search

**Tools**:
- Qdrant: https://qdrant.tech/
- Sentence Transformers: https://www.sbert.net/
- LangChain: https://python.langchain.com/

**Blogs**:
- Pinecone Learning Center
- Qdrant Documentation
- OpenAI Cookbook (RAG examples)

---

## ⏭️ Next Steps

Vector DB와 RAG를 마스터했으니:

1. **Advanced RAG** → Multi-query, Self-query
2. **Fine-tuning Embeddings** → Domain-specific embeddings
3. **Production Deployment** → Scaling, Monitoring

👉 Continue to **프로덕션 배포**

**Vector Database & RAG 마스터 완료!** 🎉
