# SVM, KNN, Naive Bayes

## 1. Support Vector Machine (SVM)

### 1.1 핵심 아이디어: 최대 마진

```python
# 의도: SVM의 기본 원리 이해
# 아이디어: "가장 안전한" 결정 경계 찾기

"""
                  |
    o  o  o       |       x  x  x
       o  o       |     x  x
          o       |   x
        [margin]  | [margin]
─────────────────────────────────── Decision Boundary (초평면)

목표: Margin을 최대화하는 초평면 찾기

Margin = 가장 가까운 샘플(Support Vector)까지의 거리
→ Margin이 클수록 일반화 성능 좋음
"""

from sklearn import svm
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

# 2D 데이터 생성
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, random_state=1, n_clusters_per_class=1)

# Linear SVM 학습
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# Support Vectors 확인
print(f"Support Vectors 개수: {len(clf.support_vectors_)}")
print(f"전체 샘플 중 {len(clf.support_vectors_)/len(X)*100:.1f}%만 사용")

# 시각화
def plot_svm_decision_boundary(clf, X, y):
    """SVM 결정 경계 시각화"""

    plt.figure(figsize=(10, 6))

    # 결정 경계
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 그리드 생성
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # 결정 경계와 마진
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # 데이터 포인트
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # Support Vectors 강조
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='k')

    plt.title('SVM: Support Vectors (검은 테두리)')
    plt.show()

plot_svm_decision_boundary(clf, X, y)

# 실전 팁: Support Vector만으로 결정 경계 결정
# → 나머지 샘플은 제거해도 동일한 결과!
```

### 1.2 Kernel Trick: 비선형 분류

```python
# 의도: Kernel Trick으로 비선형 경계 학습
# 아이디어: 고차원 공간으로 변환 → 선형 분리

"""
2D에서 선형 분리 불가능:
    x x x x x
  x           x
 x      o o    x
 x    o o o    x
  x   o o     x
    x x x x x

3D로 변환 (e.g., z = x^2 + y^2):
  z
  ↑
  |    o o o (높이 낮음)
  |  x x x x x (높이 높음)
  └──────────→ x, y

→ 3D에서는 평면으로 분리 가능!
"""

from sklearn.datasets import make_circles

# 원형 데이터 (선형 분리 불가능)
X_circles, y_circles = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=42)

# 1. Linear Kernel (실패)
linear_svm = svm.SVC(kernel='linear')
linear_svm.fit(X_circles, y_circles)
linear_score = linear_svm.score(X_circles, y_circles)
print(f"Linear SVM 정확도: {linear_score:.2%}")  # ~50% (랜덤과 동일)

# 2. RBF Kernel (성공)
rbf_svm = svm.SVC(kernel='rbf', gamma='scale')
rbf_svm.fit(X_circles, y_circles)
rbf_score = rbf_svm.score(X_circles, y_circles)
print(f"RBF SVM 정확도: {rbf_score:.2%}")  # ~100%

# 3. Polynomial Kernel
poly_svm = svm.SVC(kernel='poly', degree=3)
poly_svm.fit(X_circles, y_circles)
poly_score = poly_svm.score(X_circles, y_circles)
print(f"Polynomial SVM 정확도: {poly_score:.2%}")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, clf, title in zip(axes,
                          [linear_svm, rbf_svm, poly_svm],
                          ['Linear Kernel', 'RBF Kernel', 'Poly Kernel']):
    ax.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap=plt.cm.Paired)

    # 결정 경계
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
    ax.set_title(title)

plt.tight_layout()
plt.show()
```

### 1.3 Kernel 종류와 선택

```python
# 의도: 각 Kernel의 특징과 사용 시기
# 아이디어: 문제에 맞는 Kernel 선택

"""
1. Linear Kernel: K(x, x') = x · x'
   - 선형 분리 가능한 데이터
   - 빠름, 해석 쉬움
   - 고차원 데이터에 유리 (텍스트, 이미지 특징)

2. RBF (Radial Basis Function) Kernel: K(x, x') = exp(-γ||x - x'||^2)
   - 비선형 데이터
   - 가장 범용적
   - γ (gamma) 조정 필요

3. Polynomial Kernel: K(x, x') = (γx · x' + r)^d
   - 특정 차수의 다항식 관계
   - degree (d) 선택 중요
   - 잘 안 쓰임 (RBF가 대부분 더 좋음)

4. Sigmoid Kernel: K(x, x') = tanh(γx · x' + r)
   - Neural Network와 유사
   - 거의 안 쓰임
"""

from sklearn.model_selection import cross_val_score

# 다양한 Kernel 비교
kernels = {
    'Linear': svm.SVC(kernel='linear'),
    'RBF': svm.SVC(kernel='rbf', gamma='scale'),
    'Poly (d=2)': svm.SVC(kernel='poly', degree=2),
    'Poly (d=3)': svm.SVC(kernel='poly', degree=3),
}

print("Kernel 성능 비교 (Cross-Validation):")
for name, clf in kernels.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name:12s}: {scores.mean():.4f} ± {scores.std():.4f}")

# 실전 가이드:
"""
데이터 특징          추천 Kernel
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
선형 분리 가능       Linear
고차원 (>1000)       Linear
비선형, 저차원       RBF (기본값)
특정 다항식 관계     Polynomial
잘 모르겠다          RBF로 시작
"""
```

### 1.4 하이퍼파라미터: C와 gamma

```python
# 의도: C, gamma가 모델에 미치는 영향
# 아이디어: C는 Regularization, gamma는 Kernel 폭

"""
C (Regularization):
  - 작을수록: Margin 넓게 (Underfitting)
  - 클수록: Margin 좁게, 모든 샘플 정확히 분류 (Overfitting)
  - 기본값: 1.0
  - 추천 범위: [0.1, 1, 10, 100]

gamma (RBF Kernel):
  - 작을수록: 멀리 있는 샘플도 영향 (Smooth boundary)
  - 클수록: 가까운 샘플만 영향 (Complex boundary)
  - 기본값: 'scale' = 1 / (n_features * X.var())
  - 추천 범위: [0.001, 0.01, 0.1, 1]
"""

from sklearn.model_selection import GridSearchCV

# Grid Search로 최적 C, gamma 찾기
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(
    svm.SVC(kernel='rbf'),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(X, y)

print(f"Best Params: {grid.best_params_}")
print(f"Best Score: {grid.best_score_:.4f}")

# 시각화: C, gamma 영향
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

params_list = [
    {'C': 0.1, 'gamma': 0.1},
    {'C': 1, 'gamma': 0.1},
    {'C': 0.1, 'gamma': 1},
    {'C': 1, 'gamma': 1}
]

for ax, params in zip(axes.flatten(), params_list):
    clf = svm.SVC(kernel='rbf', **params)
    clf.fit(X_circles, y_circles)

    ax.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap=plt.cm.Paired)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.predict(xy).reshape(XX.shape)

    ax.contourf(XX, YY, Z, alpha=0.3, cmap=plt.cm.Paired)
    ax.set_title(f"C={params['C']}, gamma={params['gamma']}")

plt.tight_layout()
plt.show()
```

---

## 2. K-Nearest Neighbors (KNN)

### 2.1 기본 원리

```python
# 의도: KNN의 단순하지만 강력한 아이디어
# 아이디어: "가까운 이웃들의 다수결"

"""
새 샘플 (?) 분류:

        o o o
      o   o   o
    o     ?     o      x x x
      o o o          x   x   x
                   x       x   x

K=3: 가장 가까운 3개 → o, o, o → 예측: o
K=7: 가장 가까운 7개 → o*5, x*2 → 예측: o
"""

from sklearn.neighbors import KNeighborsClassifier

# KNN 학습
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# 예측
new_sample = [[0, 0]]
prediction = knn.predict(new_sample)
neighbors = knn.kneighbors(new_sample, return_distance=True)

print(f"새 샘플: {new_sample}")
print(f"예측: {prediction[0]}")
print(f"가장 가까운 {knn.n_neighbors}개 이웃:")
for dist, idx in zip(neighbors[0][0], neighbors[1][0]):
    print(f"  거리 {dist:.2f}, 클래스 {y[idx]}")

# 실전 팁: KNN은 "lazy learning" - 학습 단계에서는 데이터만 저장
# → 예측 시 모든 샘플과 거리 계산 (느림)
```

### 2.2 K 선택의 영향

```python
# 의도: K가 모델 복잡도에 미치는 영향
# 아이디어: K 작으면 복잡(Overfitting), K 크면 단순(Underfitting)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K값에 따른 성능 변화
k_values = range(1, 51)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, label='Train Accuracy')
plt.plot(k_values, test_scores, label='Test Accuracy')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN: K값에 따른 성능')
plt.legend()
plt.grid(True)
plt.show()

# 최적 K 찾기
best_k = k_values[np.argmax(test_scores)]
print(f"최적 K: {best_k}")
print(f"Test Accuracy: {max(test_scores):.4f}")

# 실전 팁:
"""
- K=1: 매우 복잡, Overfitting
- K=√n: 일반적 경험칙
- K=n: 모든 샘플의 다수결 (매우 단순)
- Cross-validation으로 최적 K 찾기
"""
```

### 2.3 거리 척도 (Distance Metrics)

```python
# 의도: 다양한 거리 척도의 영향
# 아이디어: 데이터 특성에 맞는 거리 선택

"""
1. Euclidean Distance (유클리드 거리):
   d = √((x1-x2)^2 + (y1-y2)^2)
   - 기본값, 가장 일반적
   - 크기에 민감 → Scaling 필요

2. Manhattan Distance (맨해튼 거리):
   d = |x1-x2| + |y1-y2|
   - 격자형 이동 거리
   - Outlier에 덜 민감

3. Minkowski Distance (민코프스키 거리):
   d = (Σ|xi-yi|^p)^(1/p)
   - p=1: Manhattan
   - p=2: Euclidean
   - p=∞: Chebyshev

4. Cosine Distance:
   d = 1 - (x·y) / (||x|| ||y||)
   - 방향만 고려 (크기 무시)
   - 텍스트, 추천시스템에 유용
"""

from sklearn.neighbors import KNeighborsClassifier

metrics = ['euclidean', 'manhattan', 'minkowski', 'cosine']

print("거리 척도별 성능:")
for metric in metrics:
    if metric == 'minkowski':
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric, p=3)
    else:
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)

    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    print(f"{metric:12s}: {score:.4f}")

# 실전 팁:
"""
데이터 유형             추천 거리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
일반 수치 데이터        Euclidean (기본값)
Outlier 많음            Manhattan
텍스트, 고차원 sparse   Cosine
범주형 데이터           Hamming
"""
```

### 2.4 Feature Scaling의 중요성

```python
# 의도: KNN은 거리 기반 → Scaling 필수
# 아이디어: Feature 스케일이 다르면 큰 feature가 지배

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

# Wine 데이터 (Feature 스케일 차이 큼)
wine = load_wine()
X_wine = wine.data
y_wine = wine.target

print("Feature 스케일:")
for i, name in enumerate(wine.feature_names[:3]):
    print(f"{name:20s}: min={X_wine[:, i].min():.2f}, max={X_wine[:, i].max():.2f}")

# 1. Scaling 없이
knn_no_scale = KNeighborsClassifier(n_neighbors=5)
score_no_scale = cross_val_score(knn_no_scale, X_wine, y_wine, cv=5).mean()
print(f"\nScaling 없이: {score_no_scale:.4f}")

# 2. Scaling 적용
scaler = StandardScaler()
X_wine_scaled = scaler.fit_transform(X_wine)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
score_scaled = cross_val_score(knn_scaled, X_wine_scaled, y_wine, cv=5).mean()
print(f"Scaling 적용: {score_scaled:.4f}")

print(f"성능 향상: {(score_scaled - score_no_scale)*100:.1f}%p")

# 실전 팁: KNN은 반드시 Scaling 필요!
```

### 2.5 Curse of Dimensionality (차원의 저주)

```python
# 의도: 고차원에서 KNN의 한계
# 아이디어: 차원이 높아지면 모든 점이 멀어짐

"""
차원의 저주:

1D: ─o──o─o─o─o──o─  (거리 차이 명확)

2D:  o    o
       o   o
     o   o

100D: 모든 점이 비슷한 거리에 위치
      → "가까운 이웃" 개념이 무의미

해결책:
1. 차원 축소 (PCA, t-SNE)
2. Feature Selection
3. 다른 모델 사용 (Tree, Neural Network)
"""

# 시뮬레이션: 차원에 따른 거리 변화
from scipy.spatial.distance import pdist

dimensions = [2, 10, 50, 100, 500]

print("차원에 따른 평균 거리 (표준화 후):")
for dim in dimensions:
    # 랜덤 데이터 생성
    X_random = np.random.randn(100, dim)

    # 모든 쌍의 거리
    distances = pdist(X_random)

    print(f"Dim {dim:3d}: 평균={distances.mean():.2f}, 표준편차={distances.std():.2f}")
    print(f"         변동계수 (CV)={distances.std()/distances.mean():.4f}")

# 출력 예시:
# Dim   2: 평균=2.01, 표준편차=0.58, CV=0.29
# Dim  10: 평균=4.47, 표준편차=0.45, CV=0.10
# Dim 100: 평균=14.14, 표준편차=0.45, CV=0.03
# → 차원이 높아질수록 거리 차이가 줄어듦 (CV 감소)

# 실전 팁: KNN은 저차원(<20)에서만 효과적
```

---

## 3. Naive Bayes

### 3.1 Bayes' Theorem 기반 분류

```python
# 의도: 확률 기반 분류
# 아이디어: P(Class|Features) 계산

"""
Bayes' Theorem:
P(Y|X) = P(X|Y) * P(Y) / P(X)

예시: 스팸 메일 분류
P(spam|"free money") = P("free money"|spam) * P(spam) / P("free money")

Naive 가정:
- Feature들이 독립
- P(X1, X2, ..., Xn|Y) = P(X1|Y) * P(X2|Y) * ... * P(Xn|Y)
- 현실에서는 거의 성립 안 하지만, 의외로 잘 작동!
"""

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import load_iris

# Gaussian Naive Bayes (연속형 변수)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_score = gnb.score(X_test, y_test)
print(f"Gaussian NB: {gnb_score:.4f}")

# 확률 예측
proba = gnb.predict_proba(X_test[:5])
print("\n확률 예측 (상위 5개):")
for i, prob in enumerate(proba):
    print(f"Sample {i}: Class 0={prob[0]:.2%}, Class 1={prob[1]:.2%}")

# 실전 팁: 매우 빠름 (거의 계산 없음), 실시간 분류에 유리
```

### 3.2 Naive Bayes 종류

```python
# 의도: 데이터 타입에 맞는 Naive Bayes 선택
# 아이디어: 각 타입에 최적화된 확률 분포

"""
1. Gaussian NB:
   - 연속형 변수
   - 정규분포 가정
   - 예: 키, 몸무게, 온도

2. Multinomial NB:
   - 이산형 카운트 (0, 1, 2, ...)
   - 텍스트 분류 (단어 빈도)
   - 예: 문서에 "free" 3번, "money" 5번

3. Bernoulli NB:
   - 이진 변수 (0 or 1)
   - 텍스트 분류 (단어 유무)
   - 예: "free" 있음(1), "money" 있음(1)
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 텍스트 데이터
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, random_state=42)

# Bag of Words (카운트)
vectorizer_count = CountVectorizer()
X_count = vectorizer_count.fit_transform(newsgroups.data)

# Multinomial NB
mnb = MultinomialNB()
mnb_score = cross_val_score(mnb, X_count, newsgroups.target, cv=5).mean()
print(f"Multinomial NB (카운트): {mnb_score:.4f}")

# Bernoulli NB (이진)
X_binary = (X_count > 0).astype(int)
bnb = BernoulliNB()
bnb_score = cross_val_score(bnb, X_binary, newsgroups.target, cv=5).mean()
print(f"Bernoulli NB (이진): {bnb_score:.4f}")

# 실전 팁:
"""
텍스트 분류:
- 문서 길이 비슷: Multinomial NB
- 문서 길이 다양: Bernoulli NB or TF-IDF
- 성능: Multinomial NB > Bernoulli NB (대부분)
"""
```

### 3.3 Naive Bayes의 장단점

```python
"""
장점:
✅ 매우 빠름 (학습, 예측 모두)
✅ 메모리 효율적
✅ 고차원 데이터에 강함 (텍스트)
✅ Feature 독립 가정 → 각 feature 기여도 명확
✅ 실시간 분류에 적합
✅ 작은 데이터셋에서도 작동

단점:
❌ Feature 독립 가정이 현실과 안 맞음
❌ 연속형 변수에는 약함 (Gaussian 가정)
❌ 확률 추정이 부정확할 수 있음
❌ Tree/SVM/NN보다 성능 낮음 (대부분)

언제 쓸까?
✅ 텍스트 분류 (스팸 필터, 감정 분석)
✅ Baseline 모델
✅ 실시간 분류 필요
✅ 해석 쉬운 모델 필요
"""

# 실전 예시: 스팸 필터
spam_data = [
    ("free money now", 1),  # 스팸
    ("win lottery", 1),
    ("meeting at 3pm", 0),  # 정상
    ("project update", 0),
    ("free gift", 1),
    ("lunch tomorrow?", 0)
]

texts, labels = zip(*spam_data)

# Vectorize
vectorizer = CountVectorizer()
X_spam = vectorizer.fit_transform(texts)

# Train
mnb = MultinomialNB()
mnb.fit(X_spam, labels)

# Predict
new_emails = ["free offer", "meeting rescheduled"]
X_new = vectorizer.transform(new_emails)
predictions = mnb.predict(X_new)
probas = mnb.predict_proba(X_new)

for email, pred, proba in zip(new_emails, predictions, probas):
    print(f"'{email}': {'스팸' if pred == 1 else '정상'} (스팸 확률: {proba[1]:.2%})")
```

---

## 4. 모델 비교: SVM vs KNN vs Naive Bayes

### 4.1 종합 벤치마크

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time

# 데이터 생성
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, random_state=42)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    'SVM (Linear)': svm.SVC(kernel='linear'),
    'SVM (RBF)': svm.SVC(kernel='rbf'),
    'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (K=15)': KNeighborsClassifier(n_neighbors=15),
    'Naive Bayes': GaussianNB()
}

results = []

for name, model in models.items():
    # 학습 시간
    start = time.time()
    model.fit(X_scaled, y)
    train_time = time.time() - start

    # CV 성능
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, n_jobs=-1)

    # 추론 시간
    start = time.time()
    _ = model.predict(X_scaled[:1000])
    inference_time = time.time() - start

    results.append({
        'Model': name,
        'CV Accuracy': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        'Train Time': f"{train_time:.2f}s",
        'Inference (1k)': f"{inference_time*1000:.1f}ms"
    })

import pandas as pd
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

"""
예상 출력:
           Model       CV Accuracy Train Time Inference (1k)
     SVM (Linear)  0.9234 ± 0.0045      8.5s           25ms
        SVM (RBF)  0.9456 ± 0.0032     15.2s           45ms
       KNN (K=5)  0.9123 ± 0.0067      0.1s          350ms  ← 느림!
      KNN (K=15)  0.9087 ± 0.0055      0.1s          340ms
     Naive Bayes  0.8876 ± 0.0078      0.05s            5ms  ← 빠름!
"""
```

### 4.2 선택 가이드

```python
"""
모델 선택 플로우차트:

데이터 크기는?
├─ <1k samples → KNN or Naive Bayes
├─ 1k-10k → SVM or KNN
└─ >10k → SVM (Linear) or Tree models

차원 수는?
├─ <20 features → KNN 가능
├─ 20-100 → SVM
└─ >100 (텍스트 등) → SVM (Linear) or Naive Bayes

실시간 추론 필요?
├─ Yes → Naive Bayes or Linear SVM
└─ No → SVM (RBF) or KNN

데이터 타입은?
├─ 텍스트 → Naive Bayes or Linear SVM
├─ 이미지 특징 → SVM (RBF)
└─ 테이블 → Tree models (XGBoost)

해석 필요?
├─ Yes → Naive Bayes or Tree
└─ No → SVM (RBF)

결론:
- SVM: 중간 크기, 성능 중요, 비선형
- KNN: 작은 데이터, 저차원, 빠른 프로토타입
- Naive Bayes: 텍스트, 실시간, Baseline
"""

def recommend_model(n_samples, n_features, need_realtime, data_type):
    """모델 추천"""

    if data_type == 'text':
        return "Naive Bayes (텍스트 특화)"

    if need_realtime:
        return "Naive Bayes or Linear SVM (빠른 추론)"

    if n_features > 100:
        return "Linear SVM (고차원)"

    if n_samples < 1000:
        return "KNN (소규모 데이터)"

    if n_features < 20:
        return "SVM (RBF) or KNN"

    return "SVM (범용적)"

# 예시
print(recommend_model(n_samples=5000, n_features=50, need_realtime=False, data_type='tabular'))
# → "SVM (범용적)"

print(recommend_model(n_samples=10000, n_features=5000, need_realtime=True, data_type='text'))
# → "Naive Bayes (텍스트 특화)"
```

---

## 핵심 정리

### 각 모델의 특징

```
┌──────────────┬─────────────┬──────────┬───────────┬─────────────┐
│   Model      │  성능       │  속도    │ 메모리    │  해석성     │
├──────────────┼─────────────┼──────────┼───────────┼─────────────┤
│ SVM (Linear) │ ★★★★☆       │ ★★★☆☆    │ ★★★☆☆     │ ★★★☆☆       │
│ SVM (RBF)    │ ★★★★★       │ ★★☆☆☆    │ ★★☆☆☆     │ ★★☆☆☆       │
│ KNN          │ ★★★☆☆       │ ★☆☆☆☆    │ ★★★★★     │ ★★★★★       │
│ Naive Bayes  │ ★★☆☆☆       │ ★★★★★    │ ★★★★★     │ ★★★★☆       │
└──────────────┴─────────────┴──────────┴───────────┴─────────────┘
```

### 실무 사용 빈도

```
Tree Models (XGBoost, Random Forest): 70%
SVM:                                  15%
KNN:                                   5%
Naive Bayes:                          10%
```

### 언제 이 모델들을 쓸까?

```python
# ✅ SVM을 쓰는 경우
- 중간 크기 데이터 (1k-100k)
- 비선형 경계
- 성능 중요
- 이미지 특징 분류

# ✅ KNN을 쓰는 경우
- 프로토타이핑
- 추천 시스템 (유사도 기반)
- 이상치 탐지
- 저차원 데이터

# ✅ Naive Bayes를 쓰는 경우
- 텍스트 분류 (스팸 필터, 감정 분석)
- 실시간 분류
- Baseline 모델
- 작은 데이터셋

# ❌ 이 모델들을 안 쓰는 경우
- Tabular 데이터 → XGBoost/LightGBM
- 이미지/비디오 → CNN
- 텍스트 (고성능) → Transformer
- 대규모 데이터 (>1M) → Neural Network or LightGBM
```

다음: Feature Engineering →
