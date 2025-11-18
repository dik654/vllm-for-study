# 차원 축소 & 시각화 완벽 가이드

## 목차
1. [개요](#개요)
2. [선형 차원 축소](#선형-차원-축소)
3. [비선형 차원 축소](#비선형-차원-축소)
4. [시각화 기법](#시각화-기법)
5. [실전 선택 가이드](#실전-선택-가이드)
6. [프로덕션 활용](#프로덕션-활용)

---

## 개요

### 차원 축소란?

**정의**: 고차원 데이터를 저차원으로 변환하면서 중요한 정보는 최대한 보존

**왜 필요한가?**
```python
"""
문제: Curse of Dimensionality (차원의 저주)

1. 시각화 불가능
   - 3차원 이상은 그림으로 못 그림
   - 768차원 임베딩을 어떻게 이해?

2. 계산 비용
   - 차원이 높으면 거리 계산, 클러스터링 등이 느림
   - 메모리 사용량 증가

3. 노이즈
   - 불필요한 차원이 많으면 노이즈 증가
   - 모델 성능 저하

해결: 차원 축소
   - 768차원 → 2차원 (시각화)
   - 1000차원 → 50차원 (모델 입력)
   - 중요한 정보만 유지!
"""
```

### 주요 기법 분류

```python
"""
차원 축소 기법 분류

1. 선형 (Linear)
   - PCA (Principal Component Analysis)
   - LDA (Linear Discriminant Analysis)
   - MDS (Multidimensional Scaling)
   - Factor Analysis

   특징: 빠름, 해석 용이, 선형 관계만 포착

2. 비선형 (Nonlinear)
   - t-SNE (t-Distributed Stochastic Neighbor Embedding)
   - UMAP (Uniform Manifold Approximation and Projection)
   - Isomap
   - LLE (Locally Linear Embedding)
   - Autoencoders

   특징: 느림, 복잡한 구조 포착, 시각화에 강력

3. 딥러닝 기반
   - Autoencoder (AE)
   - Variational Autoencoder (VAE)
   - Contrastive Learning (SimCLR, MoCo)

   특징: 유연함, 큰 데이터셋에 적합
"""
```

---

## 선형 차원 축소

### 1. PCA (Principal Component Analysis)

**핵심 아이디어**: 데이터의 분산이 최대인 방향을 찾아 투영

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PCAExplained:
    """
    PCA 완벽 이해

    목적: 데이터의 주성분(Principal Component)을 찾아 차원 축소

    주성분: 데이터 분산이 최대인 방향
    - 1st PC: 가장 많이 퍼진 방향
    - 2nd PC: 1st에 직교하면서 다음으로 많이 퍼진 방향
    - ...

    수학:
    1. 데이터 중심화 (평균 = 0)
    2. 공분산 행렬 계산
    3. 고유값 분해 (Eigenvalue Decomposition)
    4. 고유값이 큰 순서대로 고유벡터 선택
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None

    def fit(self, X):
        """
        PCA 학습

        X: (n_samples, n_features)
        """
        # Step 1: 중심화 (각 특징의 평균을 0으로)
        # 의도: PCA는 원점을 중심으로 회전하므로 데이터를 중심에 맞춤
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: 공분산 행렬 계산
        # 의도: 변수 간 상관관계를 파악
        # Cov = (X^T X) / (n-1)
        cov_matrix = np.cov(X_centered.T)

        # Step 3: 고유값 분해 (Eigenvalue Decomposition)
        # 의도: 공분산 행렬의 고유벡터 = 주성분 방향
        #       고유값 = 그 방향으로의 분산
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Step 4: 고유값 기준 내림차순 정렬
        # 의도: 분산이 큰 순서대로 중요한 주성분부터 선택
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 5: 상위 n_components개 선택
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]

        return self

    def transform(self, X):
        """
        데이터를 저차원으로 투영

        아이디어: 고차원 데이터를 주성분 방향으로 사영(projection)
        """
        # 중심화
        X_centered = X - self.mean_

        # 주성분 방향으로 투영
        # (n_samples, n_features) @ (n_features, n_components)
        # = (n_samples, n_components)
        X_reduced = X_centered @ self.components_

        return X_reduced

    def inverse_transform(self, X_reduced):
        """
        저차원 → 고차원 복원

        주의: 정보 손실로 완벽히 복원 불가
        """
        # (n_samples, n_components) @ (n_components, n_features)
        X_reconstructed = X_reduced @ self.components_.T + self.mean_

        return X_reconstructed

# ========== 실전 예시 ==========

# 고차원 데이터 생성 (100차원)
np.random.seed(42)
n_samples = 1000
n_features = 100

# 실제로는 저차원 구조 (2차원)를 가짐
# 의도: PCA가 이 숨겨진 2차원 구조를 찾아낼 수 있는지 테스트
latent_dim = 2
latent_data = np.random.randn(n_samples, latent_dim)

# 100차원으로 투영 (random projection)
projection_matrix = np.random.randn(latent_dim, n_features)
X_high_dim = latent_data @ projection_matrix

# 노이즈 추가
X_high_dim += np.random.randn(n_samples, n_features) * 0.1

print(f"원본 데이터: {X_high_dim.shape}")  # (1000, 100)

# PCA 적용
pca = PCAExplained(n_components=2)
pca.fit(X_high_dim)
X_reduced = pca.transform(X_high_dim)

print(f"축소 데이터: {X_reduced.shape}")  # (1000, 2)

# 설명된 분산 비율
explained_var_ratio = pca.explained_variance_ / np.sum(pca.explained_variance_)
print(f"설명된 분산 비율: {explained_var_ratio}")
print(f"누적 설명 비율: {np.sum(explained_var_ratio):.2%}")

# 시각화
plt.figure(figsize=(15, 5))

# 1. Scree Plot (고유값 그래프)
# 의도: 몇 개의 주성분을 사용할지 결정 (elbow method)
plt.subplot(1, 3, 1)
plt.plot(range(1, len(pca.explained_variance_) + 1),
         pca.explained_variance_, 'bo-')
plt.xlabel('주성분 번호')
plt.ylabel('고유값 (분산)')
plt.title('Scree Plot')
plt.grid(True)

# 2. 누적 설명 분산
plt.subplot(1, 3, 2)
cumsum_var = np.cumsum(explained_var_ratio)
plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'ro-')
plt.axhline(y=0.95, color='k', linestyle='--', label='95%')
plt.xlabel('주성분 개수')
plt.ylabel('누적 설명 분산 비율')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True)

# 3. 2D 시각화
plt.subplot(1, 3, 3)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title('PCA Projection')
plt.grid(True)

plt.tight_layout()
plt.show()

"""
PCA 활용 팁:

1. 전처리:
   - 스케일링 필수! (StandardScaler)
   - 각 변수의 스케일이 다르면 분산 큰 변수가 지배

2. 주성분 개수 선택:
   - Scree plot에서 elbow 찾기
   - 누적 설명 분산 95% 이상
   - Cross-validation으로 최적값 찾기

3. 장점:
   - 빠름 (O(n * d^2))
   - 해석 가능 (각 주성분의 의미)
   - 노이즈 제거 효과

4. 단점:
   - 선형 관계만 포착
   - 이상치에 민감
   - 비선형 구조는 못 찾음
"""

# ========== Scikit-learn 버전 ==========

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 데이터 스케일링 (중요!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_high_dim)

# PCA
pca_sklearn = PCA(n_components=0.95)  # 95% 분산 설명
X_pca = pca_sklearn.fit_transform(X_scaled)

print(f"\n95% 분산 설명하는 주성분 개수: {pca_sklearn.n_components_}")
print(f"각 주성분의 분산 비율: {pca_sklearn.explained_variance_ratio_}")

# 주성분 해석
# components_: (n_components, n_features)
# 각 행 = 주성분, 각 열 = 원본 특징의 기여도
print(f"\n1st PC의 주요 특징:")
pc1_weights = pca_sklearn.components_[0]
top_features = np.argsort(np.abs(pc1_weights))[-5:][::-1]
for idx in top_features:
    print(f"  Feature {idx}: {pc1_weights[idx]:.3f}")
```

---

### 2. LDA (Linear Discriminant Analysis)

**핵심 아이디어**: 클래스를 최대한 잘 분리하는 방향 찾기

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDAExplained:
    """
    LDA (Linear Discriminant Analysis)

    PCA vs LDA:
    - PCA: 분산 최대화 (비지도)
    - LDA: 클래스 간 분리 최대화 (지도)

    목적: 클래스 내 분산은 작게, 클래스 간 분산은 크게

    수식:
    J(w) = (w^T S_B w) / (w^T S_W w)

    - S_B: Between-class scatter (클래스 간 분산)
    - S_W: Within-class scatter (클래스 내 분산)
    - 목표: J(w) 최대화
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, y):
        """
        LDA 학습

        X: (n_samples, n_features)
        y: (n_samples,) - 클래스 레이블
        """
        n_features = X.shape[1]
        classes = np.unique(y)

        # 전체 평균
        mean_overall = np.mean(X, axis=0)

        # 클래스별 평균
        mean_classes = {}
        for c in classes:
            mean_classes[c] = np.mean(X[y == c], axis=0)

        # Step 1: Within-class scatter matrix (S_W)
        # 의도: 각 클래스 내에서 데이터가 얼마나 흩어져 있는지
        S_W = np.zeros((n_features, n_features))
        for c in classes:
            X_c = X[y == c]
            # 클래스 c 내 공분산
            S_W += (X_c - mean_classes[c]).T @ (X_c - mean_classes[c])

        # Step 2: Between-class scatter matrix (S_B)
        # 의도: 클래스 평균들이 전체 평균에서 얼마나 떨어져 있는지
        S_B = np.zeros((n_features, n_features))
        for c in classes:
            n_c = np.sum(y == c)
            mean_diff = (mean_classes[c] - mean_overall).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        # Step 3: S_W^{-1} S_B의 고유값 분해
        # 의도: J(w) = w^T S_B w / w^T S_W w를 최대화하는 w 찾기
        # 이는 S_W^{-1} S_B의 고유벡터 문제로 변환됨
        eigen_matrix = np.linalg.inv(S_W) @ S_B
        eigenvalues, eigenvectors = np.linalg.eig(eigen_matrix)

        # 고유값 기준 정렬
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 상위 n_components 선택
        # 최대: min(n_features, n_classes - 1)
        max_components = min(n_features, len(classes) - 1)
        n_comp = min(self.n_components, max_components)

        self.components_ = eigenvectors[:, :n_comp].real

        return self

    def transform(self, X):
        """저차원 투영"""
        return X @ self.components_

# ========== 실전 예시: Iris 데이터셋 ==========

from sklearn.datasets import load_iris

# 데이터 로드
iris = load_iris()
X_iris = iris.data  # (150, 4)
y_iris = iris.target  # 3개 클래스

# LDA 적용 (최대 2개 성분: n_classes - 1 = 3 - 1 = 2)
lda = LDAExplained(n_components=2)
lda.fit(X_iris, y_iris)
X_lda = lda.transform(X_iris)

# 시각화
plt.figure(figsize=(15, 5))

# PCA vs LDA 비교
# 1. PCA
plt.subplot(1, 3, 1)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_iris)
for c in np.unique(y_iris):
    plt.scatter(X_pca[y_iris == c, 0], X_pca[y_iris == c, 1],
                label=iris.target_names[c], alpha=0.6)
plt.xlabel('1st PC')
plt.ylabel('2nd PC')
plt.title('PCA (Unsupervised)')
plt.legend()
plt.grid(True)

# 2. LDA
plt.subplot(1, 3, 2)
for c in np.unique(y_iris):
    plt.scatter(X_lda[y_iris == c, 0], X_lda[y_iris == c, 1],
                label=iris.target_names[c], alpha=0.6)
plt.xlabel('1st LD')
plt.ylabel('2nd LD')
plt.title('LDA (Supervised)')
plt.legend()
plt.grid(True)

# 3. 원본 데이터 (처음 2개 특징)
plt.subplot(1, 3, 3)
for c in np.unique(y_iris):
    plt.scatter(X_iris[y_iris == c, 0], X_iris[y_iris == c, 1],
                label=iris.target_names[c], alpha=0.6)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Original (First 2 Features)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

"""
관찰:
- PCA: 분산이 큰 방향으로 투영 → 클래스 분리가 명확하지 않을 수 있음
- LDA: 클래스 분리를 명시적으로 최적화 → 더 잘 분리됨

LDA 활용 팁:

1. 사용 시기:
   - 분류 문제에서 시각화
   - 클래스 레이블이 있을 때
   - 차원 축소 + 분류 성능 향상

2. 제약:
   - 최대 (n_classes - 1)개 성분만 가능
   - 클래스 개수가 적으면 차원 많이 못 줄임
   - 클래스 내 공분산이 비슷하다고 가정

3. PCA vs LDA:
   - 비지도 작업: PCA
   - 분류 작업: LDA
   - 둘 다 시도해서 비교 추천
"""
```

---

## 비선형 차원 축소

### 1. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**핵심 아이디어**: 고차원에서 가까운 점들은 저차원에서도 가깝게, 먼 점들은 멀게

```python
from sklearn.manifold import TSNE
import time

class TSNEExplained:
    """
    t-SNE 완벽 이해

    목적: 고차원 데이터의 지역적 구조를 저차원에 보존

    아이디어:
    1. 고차원에서 점 i와 j의 유사도를 확률 p_ij로 표현
    2. 저차원에서도 유사한 확률 분포 q_ij를 만듦
    3. p_ij와 q_ij의 차이(KL divergence)를 최소화

    특징:
    - 비선형 구조 잘 포착
    - 클러스터 시각화에 강력
    - 계산 비용 높음 (O(n^2))
    - 재현성 낮음 (랜덤 초기화)
    - 전역 구조 보존 약함

    하이퍼파라미터:
    - perplexity: 각 점의 이웃 개수 (5-50)
    - learning_rate: 학습률 (10-1000)
    - n_iter: 반복 횟수 (최소 250, 권장 1000+)
    """

    def __init__(self, n_components=2, perplexity=30, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter

    def fit_transform(self, X):
        """
        t-SNE 적용 (sklearn 사용)

        주의: fit과 transform 분리 불가!
        새 데이터에 적용 못 함 (시각화 전용)
        """
        tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            n_iter=self.n_iter,
            random_state=42,
            verbose=1
        )

        X_embedded = tsne.fit_transform(X)

        return X_embedded

# ========== 실전 예시: MNIST ==========

from sklearn.datasets import load_digits

# Digits 데이터셋 로드 (8x8 = 64차원)
digits = load_digits()
X_digits = digits.data  # (1797, 64)
y_digits = digits.target  # 0-9

print(f"원본 데이터: {X_digits.shape}")

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_digits)

# ========== 1. Perplexity 영향 ==========

perplexities = [5, 30, 50, 100]

fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.ravel()

for i, perp in enumerate(perplexities):
    print(f"\nt-SNE with perplexity={perp}")

    # t-SNE 적용
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, verbose=0)
    X_tsne = tsne.fit_transform(X_scaled[:500])  # 시간 단축을 위해 500개만

    # 시각화
    ax = axes[i]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                        c=y_digits[:500], cmap='tab10', alpha=0.6)
    ax.set_title(f't-SNE (perplexity={perp})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()

"""
Perplexity 영향:

- 낮은 perplexity (5):
  * 지역적 구조에 집중
  * 작은 클러스터 여러 개
  * 노이즈에 민감

- 중간 perplexity (30):
  * 균형잡힌 결과
  * 대부분 이 값 사용

- 높은 perplexity (50-100):
  * 전역적 구조 강조
  * 큰 클러스터
  * 계산 시간 증가

권장: 데이터 크기에 따라 조정
- 작은 데이터 (< 1000): perplexity = 5-30
- 중간 데이터 (1000-10000): perplexity = 30-50
- 큰 데이터 (> 10000): perplexity = 50-100
"""

# ========== 2. PCA vs t-SNE 비교 ==========

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled[:1000])

axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                c=y_digits[:1000], cmap='tab10', alpha=0.6)
axes[0].set_title('PCA (Linear)')
axes[0].set_xlabel('PC 1')
axes[0].set_ylabel('PC 2')

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled[:1000])

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=y_digits[:1000], cmap='tab10', alpha=0.6)
axes[1].set_title('t-SNE (Nonlinear)')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()

"""
관찰:
- PCA: 일부 클래스가 겹침 (선형 관계만 포착)
- t-SNE: 클래스들이 명확히 분리됨 (비선형 구조 포착)

t-SNE 활용 팁:

1. 전처리:
   - 스케일링 필수
   - PCA로 먼저 50-100차원으로 축소 (속도 향상)
   - 이상치 제거

2. 해석 주의:
   - 클러스터 간 거리는 의미 없음
   - 클러스터 크기도 의미 없음
   - 오직 점들의 그룹화만 의미 있음

3. 재현성:
   - random_state 고정
   - 여러 번 실행해서 안정적인 결과 확인

4. 속도 개선:
   - 샘플링 (큰 데이터는 일부만)
   - PCA 전처리
   - early_exaggeration 조정

5. 사용 사례:
   - 고차원 데이터 탐색
   - 클러스터링 결과 시각화
   - 임베딩 품질 평가

주의:
- 새 데이터에 적용 불가 (시각화 전용)
- 클러스터 개수 추정용으로만
- 정량적 분석에는 부적합
"""

# ========== 3. t-SNE 최적화 파이프라인 ==========

def optimized_tsne(X, y=None, n_samples=1000):
    """
    최적화된 t-SNE 파이프라인

    의도: 대용량 데이터에서 빠르고 좋은 시각화
    """
    import time

    print("Step 1: 샘플링")
    if X.shape[0] > n_samples:
        # 층화 샘플링 (클래스별 비율 유지)
        if y is not None:
            from sklearn.model_selection import train_test_split
            X_sample, _, y_sample, _ = train_test_split(
                X, y, train_size=n_samples, stratify=y, random_state=42
            )
        else:
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sample = X[indices]
            y_sample = y[indices] if y is not None else None
    else:
        X_sample = X
        y_sample = y

    print(f"샘플 크기: {X_sample.shape}")

    print("\nStep 2: 스케일링")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    print("\nStep 3: PCA 전처리 (50차원)")
    # 의도: 차원이 너무 높으면 t-SNE가 느리고 불안정
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA 설명 분산: {pca.explained_variance_ratio_.sum():.2%}")

    print("\nStep 4: t-SNE")
    start = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42,
        verbose=1
    )
    X_embedded = tsne.fit_transform(X_pca)
    elapsed = time.time() - start
    print(f"t-SNE 완료: {elapsed:.2f}초")

    return X_embedded, y_sample

# 사용
# X_tsne, y_sample = optimized_tsne(X_digits, y_digits, n_samples=1000)
```

---

### 2. UMAP (Uniform Manifold Approximation and Projection)

**핵심 아이디어**: t-SNE의 개선 버전 - 더 빠르고, 전역 구조도 보존

```python
# pip install umap-learn
import umap

class UMAPExplained:
    """
    UMAP 완벽 이해

    t-SNE의 단점 보완:
    1. 속도: 훨씬 빠름 (O(n log n) vs O(n^2))
    2. 전역 구조: 더 잘 보존
    3. 새 데이터: transform 가능 (학습 후 적용)
    4. 재현성: 더 안정적

    아이디어:
    - Riemannian geometry와 topology 기반
    - 고차원의 manifold 구조를 저차원으로 투영
    - t-SNE보다 수학적으로 더 엄밀

    하이퍼파라미터:
    - n_neighbors: 지역 구조 크기 (5-50)
    - min_dist: 저차원에서 점 간 최소 거리 (0.0-0.99)
    - metric: 거리 메트릭 (euclidean, cosine, etc.)
    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.model = None

    def fit(self, X):
        """UMAP 학습"""
        self.model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=42,
            verbose=True
        )
        self.model.fit(X)
        return self

    def transform(self, X):
        """
        새 데이터 변환

        장점: t-SNE와 달리 학습 후 새 데이터에 적용 가능!
        """
        return self.model.transform(X)

    def fit_transform(self, X):
        """학습 + 변환"""
        self.fit(X)
        return self.model.embedding_

# ========== 실전 예시: t-SNE vs UMAP ==========

# 데이터
X_scaled = StandardScaler().fit_transform(X_digits)

# ========== 1. 속도 비교 ==========

print("=" * 50)
print("속도 비교 (1000 샘플)")
print("=" * 50)

X_test = X_scaled[:1000]

# t-SNE
start = time.time()
tsne = TSNE(n_components=2, random_state=42, verbose=0)
X_tsne = tsne.fit_transform(X_test)
tsne_time = time.time() - start
print(f"t-SNE: {tsne_time:.2f}초")

# UMAP
start = time.time()
reducer_umap = umap.UMAP(n_components=2, random_state=42, verbose=False)
X_umap = reducer_umap.fit_transform(X_test)
umap_time = time.time() - start
print(f"UMAP: {umap_time:.2f}초")

print(f"속도 향상: {tsne_time / umap_time:.1f}배")

# ========== 2. 시각화 비교 ==========

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                c=y_digits[:1000], cmap='tab10', alpha=0.6, s=10)
axes[0].set_title('PCA (Linear, Fast)')
axes[0].set_xlabel('PC 1')
axes[0].set_ylabel('PC 2')

# t-SNE
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=y_digits[:1000], cmap='tab10', alpha=0.6, s=10)
axes[1].set_title('t-SNE (Nonlinear, Slow)')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

# UMAP
axes[2].scatter(X_umap[:, 0], X_umap[:, 1],
                c=y_digits[:1000], cmap='tab10', alpha=0.6, s=10)
axes[2].set_title('UMAP (Nonlinear, Fast)')
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')

plt.tight_layout()
plt.show()

# ========== 3. 하이퍼파라미터 영향 ==========

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# n_neighbors 영향
n_neighbors_list = [5, 15, 50]
for i, n_neighbors in enumerate(n_neighbors_list):
    reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42, verbose=False)
    X_reduced = reducer.fit_transform(X_test)

    axes[0, i].scatter(X_reduced[:, 0], X_reduced[:, 1],
                      c=y_digits[:1000], cmap='tab10', alpha=0.6, s=10)
    axes[0, i].set_title(f'n_neighbors={n_neighbors}')

# min_dist 영향
min_dist_list = [0.0, 0.1, 0.5]
for i, min_dist in enumerate(min_dist_list):
    reducer = umap.UMAP(min_dist=min_dist, random_state=42, verbose=False)
    X_reduced = reducer.fit_transform(X_test)

    axes[1, i].scatter(X_reduced[:, 0], X_reduced[:, 1],
                      c=y_digits[:1000], cmap='tab10', alpha=0.6, s=10)
    axes[1, i].set_title(f'min_dist={min_dist}')

plt.tight_layout()
plt.show()

"""
하이퍼파라미터 영향:

n_neighbors (이웃 개수):
- 낮은 값 (5): 지역 구조 강조, 작은 클러스터
- 중간 값 (15): 기본값, 균형
- 높은 값 (50): 전역 구조 강조, 큰 클러스터

min_dist (최소 거리):
- 0.0: 점들이 빽빽하게, 클러스터 명확
- 0.1: 기본값, 적당한 간격
- 0.5: 점들이 퍼짐, 전역 구조 보존

UMAP 활용 팁:

1. 사용 시기:
   - t-SNE보다 빠른 속도 필요
   - 전역 구조도 중요
   - 새 데이터에 적용 필요
   - 대규모 데이터 (> 10,000)

2. 하이퍼파라미터 튜닝:
   - n_neighbors: 데이터 크기에 비례
   - min_dist: 시각화 목적에 따라
   - metric: 데이터 타입에 맞게 (cosine for text)

3. 장점:
   - 빠름 (t-SNE보다 10-100배)
   - 전역 구조 보존
   - 새 데이터 변환 가능
   - 재현성 좋음
   - 다양한 metric 지원

4. 단점:
   - t-SNE보다 복잡한 알고리즘
   - 하이퍼파라미터 영향 큼
   - 극도로 작은 데이터에는 t-SNE가 나을 수 있음
"""

# ========== 4. UMAP의 새 데이터 변환 ==========

# 학습 데이터
X_train = X_scaled[:1000]
y_train = y_digits[:1000]

# 테스트 데이터
X_test_new = X_scaled[1000:1500]
y_test_new = y_digits[1000:1500]

# UMAP 학습
reducer = umap.UMAP(n_components=2, random_state=42, verbose=False)
reducer.fit(X_train)

# 학습 데이터 임베딩
X_train_umap = reducer.embedding_

# 새 데이터 변환 (이게 가능!)
X_test_umap = reducer.transform(X_test_new)

# 시각화
plt.figure(figsize=(10, 8))
plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1],
            c=y_train, cmap='tab10', alpha=0.3, s=20, label='Train')
plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1],
            c=y_test_new, cmap='tab10', alpha=0.8, s=50,
            edgecolors='black', linewidths=1, label='Test (transformed)')
plt.title('UMAP: Train vs Test')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend()
plt.colorbar()
plt.show()

print("✓ UMAP은 새 데이터를 동일한 공간으로 변환 가능!")
```

---

### 3. Autoencoder

**핵심 아이디어**: 신경망으로 데이터를 압축했다가 복원

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    """
    Autoencoder for Dimensionality Reduction

    아키텍처:
    Input (784) → Encoder → Latent (2) → Decoder → Output (784)

    목적: 입력을 저차원으로 압축했다가 복원
    - Encoder: 784 → 2 (차원 축소)
    - Decoder: 2 → 784 (복원)

    학습: Reconstruction loss 최소화
    Loss = MSE(Input, Output)

    장점:
    - 비선형 차원 축소
    - 새 데이터 변환 가능
    - 복원 가능

    의도:
    - Latent space에 의미있는 representation 학습
    - 2D latent로 시각화 가능
    """

    def __init__(self, input_dim=784, latent_dim=2):
        super().__init__()

        # Encoder: 784 → 128 → 64 → 2
        # 의도: 점진적으로 차원을 줄여 중요한 정보만 남김
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)  # Bottleneck
        )

        # Decoder: 2 → 64 → 128 → 784
        # 의도: 압축된 정보로부터 원본 복원
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # [0, 1] 범위로 출력
        )

    def forward(self, x):
        """순전파"""
        # Encoding
        z = self.encoder(x)

        # Decoding
        x_reconstructed = self.decoder(z)

        return x_reconstructed, z

    def encode(self, x):
        """인코딩만 (차원 축소)"""
        return self.encoder(x)

# ========== 학습 ==========

# MNIST 데이터 준비
from sklearn.datasets import fetch_openml

# MNIST 로드 (시간이 걸릴 수 있음)
# mnist = fetch_openml('mnist_784', version=1)
# X_mnist = mnist.data / 255.0  # [0, 1] 정규화
# y_mnist = mnist.target.astype(int)

# 여기서는 Digits 사용 (빠른 데모)
X_norm = X_digits / 16.0  # [0, 1] 정규화
X_tensor = torch.FloatTensor(X_norm)
y_tensor = torch.LongTensor(y_digits)

# Train/Test split
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 모델 초기화
model = Autoencoder(input_dim=64, latent_dim=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
epochs = 50
losses = []

print("Autoencoder 학습 시작...")
for epoch in range(epochs):
    epoch_loss = 0
    for batch_x, _ in train_loader:
        # Forward
        x_recon, z = model(batch_x)
        loss = criterion(x_recon, batch_x)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

# ========== 시각화 ==========

# 전체 데이터 인코딩
model.eval()
with torch.no_grad():
    _, z_all = model(X_tensor)
    z_all = z_all.numpy()

# 시각화
plt.figure(figsize=(15, 5))

# 1. Loss curve
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.grid(True)

# 2. Latent space visualization
plt.subplot(1, 3, 2)
scatter = plt.scatter(z_all[:, 0], z_all[:, 1],
                     c=y_digits, cmap='tab10', alpha=0.6)
plt.xlabel('Latent Dim 1')
plt.ylabel('Latent Dim 2')
plt.title('Autoencoder Latent Space')
plt.colorbar(scatter)
plt.grid(True)

# 3. Reconstruction 예시
plt.subplot(1, 3, 3)
with torch.no_grad():
    sample_idx = 0
    original = X_tensor[sample_idx:sample_idx+1]
    reconstructed, _ = model(original)

    original_img = original.numpy().reshape(8, 8)
    recon_img = reconstructed.numpy().reshape(8, 8)

    combined = np.hstack([original_img, recon_img])
    plt.imshow(combined, cmap='gray')
    plt.title('Original (left) vs Reconstructed (right)')
    plt.axis('off')

plt.tight_layout()
plt.show()

"""
Autoencoder 활용 팁:

1. 아키텍처 설계:
   - Bottleneck이 너무 작으면: 정보 손실 크고, 클러스터 뭉침
   - Bottleneck이 너무 크면: 차원 축소 효과 적음
   - 점진적 감소: 784 → 256 → 64 → 2

2. Regularization:
   - Dropout 추가 (overfitting 방지)
   - L2 regularization
   - Batch Normalization

3. 변형:
   - Variational Autoencoder (VAE): latent space가 더 smooth
   - Denoising Autoencoder: 노이즈 제거 학습
   - Sparse Autoencoder: 희소성 강제

4. 장점:
   - 비선형 관계 학습
   - 새 데이터 변환 가능
   - 복원 가능 (생성 모델로 활용)
   - 레이블 불필요 (비지도)

5. 단점:
   - 학습 시간 김
   - 하이퍼파라미터 튜닝 필요
   - 작은 데이터에는 과적합 위험
"""
```

---

## 시각화 기법

### 1. 고급 시각화

```python
import seaborn as sns

class AdvancedVisualization:
    """
    차원 축소 결과 고급 시각화

    목적: 데이터의 패턴, 클러스터, 이상치를 효과적으로 표현
    """

    @staticmethod
    def plot_2d_scatter(X_2d, y=None, title='2D Projection',
                        alpha=0.6, s=20, cmap='tab10'):
        """
        기본 2D scatter plot

        의도: 간단하고 명확한 시각화
        """
        plt.figure(figsize=(10, 8))

        if y is not None:
            scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                                 c=y, cmap=cmap, alpha=alpha, s=s)
            plt.colorbar(scatter, label='Class')
        else:
            plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=alpha, s=s)

        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_density(X_2d, y=None, title='Density Plot'):
        """
        밀도 기반 시각화

        의도: 점이 많을 때 겹침 문제 해결
        """
        plt.figure(figsize=(12, 5))

        # 1. Hexbin plot
        plt.subplot(1, 2, 1)
        if y is not None:
            for label in np.unique(y):
                mask = y == label
                plt.hexbin(X_2d[mask, 0], X_2d[mask, 1],
                          gridsize=30, alpha=0.6, label=f'Class {label}')
        else:
            plt.hexbin(X_2d[:, 0], X_2d[:, 1], gridsize=30, cmap='Blues')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Hexbin Density')
        plt.colorbar(label='Count')

        # 2. KDE plot
        plt.subplot(1, 2, 2)
        if y is not None:
            for label in np.unique(y):
                mask = y == label
                sns.kdeplot(x=X_2d[mask, 0], y=X_2d[mask, 1],
                           label=f'Class {label}', alpha=0.5)
        else:
            sns.kdeplot(x=X_2d[:, 0], y=X_2d[:, 1])
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('KDE Density')
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_interactive(X_2d, y=None, labels=None, title='Interactive Plot'):
        """
        인터랙티브 시각화 (Plotly)

        의도: 줌, 호버 등으로 세밀한 탐색 가능
        """
        import plotly.express as px
        import pandas as pd

        # DataFrame 생성
        df = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            'class': y if y is not None else ['Unknown'] * len(X_2d)
        })

        if labels is not None:
            df['label'] = labels

        # 인터랙티브 플롯
        fig = px.scatter(df, x='x', y='y', color='class',
                        hover_data=['label'] if labels is not None else None,
                        title=title)

        fig.show()

    @staticmethod
    def plot_3d(X_3d, y=None, title='3D Projection'):
        """
        3D 시각화

        의도: 3차원까지는 직접 볼 수 있음
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if y is not None:
            scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                               c=y, cmap='tab10', alpha=0.6)
            plt.colorbar(scatter, label='Class')
        else:
            ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], alpha=0.6)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title(title)

        plt.show()

    @staticmethod
    def plot_comparison(X, y, methods=['PCA', 't-SNE', 'UMAP']):
        """
        여러 방법 한번에 비교

        의도: 어떤 방법이 데이터에 적합한지 빠르게 파악
        """
        from sklearn.preprocessing import StandardScaler

        # 스케일링
        X_scaled = StandardScaler().fit_transform(X)

        fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
        if len(methods) == 1:
            axes = [axes]

        for i, method in enumerate(methods):
            print(f"Processing {method}...")

            if method == 'PCA':
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X_scaled)
            elif method == 't-SNE':
                # PCA 전처리
                pca_pre = PCA(n_components=50, random_state=42)
                X_pca = pca_pre.fit_transform(X_scaled)
                reducer = TSNE(n_components=2, random_state=42, verbose=0)
                X_reduced = reducer.fit_transform(X_pca)
            elif method == 'UMAP':
                reducer = umap.UMAP(n_components=2, random_state=42, verbose=False)
                X_reduced = reducer.fit_transform(X_scaled)

            # 시각화
            scatter = axes[i].scatter(X_reduced[:, 0], X_reduced[:, 1],
                                     c=y, cmap='tab10', alpha=0.6, s=20)
            axes[i].set_title(method)
            axes[i].set_xlabel('Component 1')
            axes[i].set_ylabel('Component 2')
            axes[i].grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=axes, label='Class')
        plt.tight_layout()
        plt.show()

# ========== 실전 사용 ==========

viz = AdvancedVisualization()

# 예시 데이터
X_scaled = StandardScaler().fit_transform(X_digits)

# 1. 여러 방법 비교
viz.plot_comparison(X_scaled[:1000], y_digits[:1000],
                    methods=['PCA', 't-SNE', 'UMAP'])

# 2. t-SNE 결과
tsne = TSNE(n_components=2, random_state=42, verbose=0)
X_tsne = tsne.fit_transform(X_scaled[:1000])

# 밀도 플롯
viz.plot_density(X_tsne, y_digits[:1000], title='t-SNE Density')

# 3. 3D UMAP
reducer_3d = umap.UMAP(n_components=3, random_state=42, verbose=False)
X_umap_3d = reducer_3d.fit_transform(X_scaled[:1000])

viz.plot_3d(X_umap_3d, y_digits[:1000], title='UMAP 3D')

"""
시각화 팁:

1. 컬러맵 선택:
   - 정성적 (클래스): tab10, Set1, Paired
   - 정량적 (연속): viridis, plasma, coolwarm
   - 색맹 고려: colorblind-friendly palettes

2. 점 크기와 투명도:
   - 데이터 많을 때: s=5, alpha=0.3
   - 데이터 적을 때: s=50, alpha=0.8

3. 인터랙티브:
   - Jupyter: plotly, bokeh
   - 대시보드: Dash, Streamlit
   - 큰 데이터: datashader

4. 애니메이션:
   - 학습 과정 시각화
   - 파라미터 변화에 따른 결과
"""
```

---

## 실전 선택 가이드

### 방법 선택 플로우차트

```python
"""
차원 축소 방법 선택 가이드

┌─────────────────────────────────────────────────────────────┐
│ 질문 1: 목적이 무엇인가?                                    │
├─────────────────────────────────────────────────────────────┤
│ A. 시각화 (2D/3D)                                           │
│    → 질문 2로                                               │
│                                                             │
│ B. 차원 축소 + 모델 입력                                     │
│    → PCA or Autoencoder                                     │
│                                                             │
│ C. 특징 추출 (해석 필요)                                     │
│    → PCA or LDA                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 질문 2: 데이터 크기는?                                      │
├─────────────────────────────────────────────────────────────┤
│ A. 작음 (< 1,000)                                           │
│    → t-SNE or PCA                                           │
│                                                             │
│ B. 중간 (1,000 - 10,000)                                    │
│    → UMAP or t-SNE                                          │
│                                                             │
│ C. 큼 (> 10,000)                                            │
│    → UMAP or PCA → t-SNE                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 질문 3: 구조는?                                             │
├─────────────────────────────────────────────────────────────┤
│ A. 선형 구조                                                │
│    → PCA                                                    │
│                                                             │
│ B. 비선형 구조                                              │
│    → t-SNE or UMAP                                          │
│                                                             │
│ C. 모르겠음                                                 │
│    → 둘 다 시도해서 비교                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 질문 4: 레이블 있음?                                        │
├─────────────────────────────────────────────────────────────┤
│ A. 있음 + 분류 목적                                         │
│    → LDA                                                    │
│                                                             │
│ B. 있음 + 시각화 목적                                       │
│    → t-SNE or UMAP (색으로 레이블 표시)                     │
│                                                             │
│ C. 없음                                                     │
│    → PCA, t-SNE, UMAP 중 선택                               │
└─────────────────────────────────────────────────────────────┘
"""

# 실전 의사결정 트리
def recommend_method(n_samples, has_labels=False, purpose='visualization'):
    """
    차원 축소 방법 추천

    의도: 데이터 특성에 맞는 최적 방법 선택
    """
    recommendations = []

    # 시각화 목적
    if purpose == 'visualization':
        if n_samples < 1000:
            recommendations.append(('t-SNE', 'high'))
            recommendations.append(('PCA', 'medium'))
        elif n_samples < 10000:
            recommendations.append(('UMAP', 'high'))
            recommendations.append(('t-SNE', 'medium'))
        else:
            recommendations.append(('UMAP', 'high'))
            recommendations.append(('PCA then t-SNE', 'medium'))

    # 모델 입력용
    elif purpose == 'feature_extraction':
        recommendations.append(('PCA', 'high'))
        if n_samples > 5000:
            recommendations.append(('Autoencoder', 'medium'))

    # 분류 + 레이블 있음
    if has_labels and purpose in ['visualization', 'classification']:
        recommendations.append(('LDA', 'high'))

    # 정렬 (우선순위 순)
    recommendations.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x[1]])

    return recommendations

# 예시
rec = recommend_method(n_samples=5000, has_labels=True, purpose='visualization')
print("추천 방법:")
for method, priority in rec:
    print(f"  {method} (우선순위: {priority})")
```

---

### 성능 비교표

```python
import pandas as pd

comparison = pd.DataFrame({
    '방법': ['PCA', 'LDA', 't-SNE', 'UMAP', 'Autoencoder'],
    '속도': ['매우 빠름', '빠름', '느림', '빠름', '중간'],
    '확장성': ['높음', '중간', '낮음', '높음', '높음'],
    '비선형': ['✗', '✗', '✓', '✓', '✓'],
    '지도학습': ['✗', '✓', '✗', '✗', '✗'],
    '새 데이터': ['✓', '✓', '✗', '✓', '✓'],
    '해석성': ['높음', '높음', '낮음', '중간', '낮음'],
    '전역 구조': ['✓', '✓', '✗', '✓', '△'],
    '지역 구조': ['△', '△', '✓', '✓', '✓'],
    '최적 용도': ['특징 추출', '분류', '시각화', '시각화+분석', '복원+생성']
})

print(comparison.to_string(index=False))

"""
출력:

     방법      속도  확장성 비선형 지도학습 새 데이터 해석성 전역 구조 지역 구조      최적 용도
      PCA  매우 빠름   높음    ✗      ✗      ✓   높음      ✓      △      특징 추출
      LDA      빠름   중간    ✗      ✓      ✓   높음      ✓      △          분류
    t-SNE      느림   낮음    ✓      ✗      ✗   낮음      ✗      ✓        시각화
     UMAP      빠름   높음    ✓      ✗      ✓   중간      ✓      ✓  시각화+분석
Autoencoder   중간   높음    ✓      ✗      ✓   낮음      △      ✓    복원+생성
"""
```

---

## 프로덕션 활용

### 파이프라인 구축

```python
class DimensionalityReductionPipeline:
    """
    프로덕션 차원 축소 파이프라인

    특징:
    - 여러 방법 시도
    - 자동 파라미터 튜닝
    - 결과 저장/로드
    - 시각화 자동화
    """

    def __init__(self, config=None):
        self.config = config or self.default_config()
        self.results = {}

    @staticmethod
    def default_config():
        return {
            'methods': ['PCA', 'UMAP', 't-SNE'],
            'n_components': 2,
            'random_state': 42,
            'scale': True,
            'pca_preprocessing': True,
            'save_results': True
        }

    def fit_transform(self, X, y=None):
        """
        여러 방법으로 차원 축소

        의도: 한 번에 모든 방법 시도하고 비교
        """
        from sklearn.preprocessing import StandardScaler

        # 전처리
        if self.config['scale']:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X)
        else:
            X_processed = X

        # PCA 전처리 (선택적)
        if self.config['pca_preprocessing'] and X.shape[1] > 50:
            print("PCA 전처리 (50차원)")
            pca_pre = PCA(n_components=50, random_state=self.config['random_state'])
            X_pca_pre = pca_pre.fit_transform(X_processed)
        else:
            X_pca_pre = X_processed

        # 각 방법 적용
        for method in self.config['methods']:
            print(f"\n{'='*50}")
            print(f"Method: {method}")
            print(f"{'='*50}")

            start_time = time.time()

            if method == 'PCA':
                reducer = PCA(n_components=self.config['n_components'],
                            random_state=self.config['random_state'])
                X_reduced = reducer.fit_transform(X_processed)

            elif method == 't-SNE':
                reducer = TSNE(n_components=self.config['n_components'],
                             random_state=self.config['random_state'],
                             verbose=1)
                X_reduced = reducer.fit_transform(X_pca_pre)

            elif method == 'UMAP':
                reducer = umap.UMAP(n_components=self.config['n_components'],
                                  random_state=self.config['random_state'],
                                  verbose=False)
                X_reduced = reducer.fit_transform(X_processed)

            elapsed = time.time() - start_time

            # 결과 저장
            self.results[method] = {
                'embedding': X_reduced,
                'reducer': reducer,
                'time': elapsed
            }

            print(f"완료: {elapsed:.2f}초")

        return self.results

    def visualize_all(self, y=None, save_path=None):
        """
        모든 결과 시각화

        의도: 한 눈에 비교 가능하게
        """
        n_methods = len(self.results)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))

        if n_methods == 1:
            axes = [axes]

        for i, (method, result) in enumerate(self.results.items()):
            X_reduced = result['embedding']
            elapsed = result['time']

            ax = axes[i]

            if y is not None:
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
                                   c=y, cmap='tab10', alpha=0.6, s=20)
                if i == n_methods - 1:  # 마지막 플롯에만 컬러바
                    plt.colorbar(scatter, ax=ax, label='Class')
            else:
                ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=20)

            ax.set_title(f'{method}\n({elapsed:.2f}s)')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"저장됨: {save_path}")

        plt.show()

    def save(self, path):
        """결과 저장"""
        import joblib
        joblib.dump(self.results, path)
        print(f"결과 저장: {path}")

    def load(self, path):
        """결과 로드"""
        import joblib
        self.results = joblib.load(path)
        print(f"결과 로드: {path}")
        return self.results

# ========== 실전 사용 ==========

# 파이프라인 설정
config = {
    'methods': ['PCA', 't-SNE', 'UMAP'],
    'n_components': 2,
    'random_state': 42,
    'scale': True,
    'pca_preprocessing': True
}

pipeline = DimensionalityReductionPipeline(config)

# 실행
results = pipeline.fit_transform(X_digits, y_digits)

# 시각화
pipeline.visualize_all(y_digits, save_path='dimensionality_reduction.png')

# 저장
# pipeline.save('dr_results.pkl')

# 나중에 로드
# pipeline.load('dr_results.pkl')
# pipeline.visualize_all(y_digits)

"""
프로덕션 체크리스트:

[  ] 데이터 스케일링
[  ] PCA 전처리 (고차원 → 50차원)
[  ] 여러 방법 비교
[  ] 하이퍼파라미터 튜닝
[  ] 결과 저장 (재현성)
[  ] 시각화 자동화
[  ] 로깅 (시간, 파라미터)
[  ] 에러 핸들링
[  ] 문서화
"""
```

---

## 핵심 요약

### 방법별 요약

| 방법 | 핵심 개념 | 장점 | 단점 | 사용 시기 |
|------|----------|------|------|----------|
| **PCA** | 분산 최대화 | 빠름, 해석 가능 | 선형만 | 특징 추출, 노이즈 제거 |
| **LDA** | 클래스 분리 | 분류 성능 향상 | 레이블 필요 | 분류 전처리 |
| **t-SNE** | 지역 구조 보존 | 클러스터 명확 | 느림, 재현성 낮음 | 작은 데이터 시각화 |
| **UMAP** | Manifold 학습 | 빠름, 전역+지역 | 파라미터 민감 | 대규모 데이터 시각화 |
| **Autoencoder** | 신경망 압축 | 유연함, 복원 가능 | 학습 필요 | 복잡한 비선형 |

### 실전 추천 워크플로우

```python
"""
데이터 탐색 워크플로우:

1. 빠른 확인 (PCA)
   → 전체적인 구조 파악
   → 이상치 탐지

2. 상세 분석 (UMAP)
   → 클러스터 구조
   → 레이블 확인

3. 세밀한 검증 (t-SNE)
   → 특정 영역 확대
   → 파라미터 조정

4. 모델 입력 (PCA or Autoencoder)
   → 차원 축소 후 분류/회귀
   → 성능 향상
"""
```

---

**작성일**: 2024-11-18
**업데이트**: Phase 6 - Current Trends
