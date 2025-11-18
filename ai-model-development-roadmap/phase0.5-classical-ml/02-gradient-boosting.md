# Gradient Boosting: XGBoost, LightGBM, CatBoost

## 1. Boosting vs Bagging

### 1.1 핵심 차이점

```python
# 의도: Boosting과 Bagging의 근본적 차이 이해
# 아이디어: Sequential(순차) vs Parallel(병렬)

"""
Bagging (Random Forest):
├── Tree 1 (독립적) ────┐
├── Tree 2 (독립적) ────┼──→ 다수결/평균
├── Tree 3 (독립적) ────┘
└── 병렬 학습 가능

특징:
- 각 트리는 독립적
- Variance 감소
- Overfitting 방지


Boosting (XGBoost, LightGBM):
Tree 1 (weak) → 오류 찾기
    ↓
Tree 2 (이전 오류 집중) → 오류 찾기
    ↓
Tree 3 (이전 오류 집중) → 오류 찾기
    ↓
최종 예측 = 가중합

특징:
- 순차 학습 (이전 트리 결과 필요)
- Bias 감소
- 성능이 더 높음 (대부분의 경우)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 데이터 생성
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bagging (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)

# Boosting (Gradient Boosting)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_score = gb.score(X_test, y_test)

print(f"Random Forest (Bagging): {rf_score:.4f}")
print(f"Gradient Boosting: {gb_score:.4f}")

# 일반적으로 Boosting이 2-5% 더 높음
```

---

## 2. Gradient Boosting 원리

### 2.1 Residual Fitting (잔차 학습)

```python
# 의도: Gradient Boosting의 핵심 아이디어 이해
# 아이디어: "이전 모델의 실수를 다음 모델로 보정"

import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingFromScratch:
    """Gradient Boosting 직접 구현"""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        """
        Gradient Boosting 알고리즘:

        1. 초기 예측: F_0(x) = mean(y)
        2. For m = 1 to M:
           a. 잔차 계산: r = y - F_{m-1}(x)
           b. 잔차에 트리 학습: h_m(x)
           c. 모델 업데이트: F_m(x) = F_{m-1}(x) + lr * h_m(x)
        """

        # 1. 초기 예측 = 평균
        self.initial_prediction = np.mean(y)
        F = np.full(len(y), self.initial_prediction)

        for m in range(self.n_estimators):
            # 2. 잔차 (Residual) 계산
            residual = y - F

            # 3. 잔차에 트리 학습
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residual)
            self.trees.append(tree)

            # 4. 예측 업데이트
            update = self.learning_rate * tree.predict(X)
            F += update

            # 진행 상황 출력 (10개마다)
            if (m + 1) % 10 == 0:
                mse = np.mean((y - F) ** 2)
                print(f"  Iteration {m+1}: MSE = {mse:.4f}")

    def predict(self, X):
        """예측"""
        # 초기 예측
        F = np.full(len(X), self.initial_prediction)

        # 모든 트리의 예측 합산
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)

        return F

# 회귀 데이터 생성
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# From Scratch Gradient Boosting
print("From Scratch Gradient Boosting:")
gb_scratch = GradientBoostingFromScratch(n_estimators=50, learning_rate=0.1, max_depth=3)
gb_scratch.fit(X_train, y_train)

y_pred_scratch = gb_scratch.predict(X_test)
mse_scratch = np.mean((y_test - y_pred_scratch) ** 2)
print(f"Test MSE (Scratch): {mse_scratch:.2f}\n")

# Sklearn Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gb_sklearn = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
gb_sklearn.fit(X_train, y_train)
mse_sklearn = np.mean((y_test - gb_sklearn.predict(X_test)) ** 2)
print(f"Test MSE (Sklearn): {mse_sklearn:.2f}")

# 거의 동일!
```

### 2.2 Learning Rate의 역할

```python
# 의도: Learning Rate가 성능에 미치는 영향
# 아이디어: 작을수록 천천히 학습 → Overfitting 방지

import matplotlib.pyplot as plt

learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
results = []

for lr in learning_rates:
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train, y_train)

    train_mse = np.mean((y_train - gb.predict(X_train)) ** 2)
    test_mse = np.mean((y_test - gb.predict(X_test)) ** 2)

    results.append({
        'learning_rate': lr,
        'train_mse': train_mse,
        'test_mse': test_mse
    })

    print(f"LR={lr:.2f}: Train MSE={train_mse:.2f}, Test MSE={test_mse:.2f}")

# 시각화
import pandas as pd
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
plt.plot(results_df['learning_rate'], results_df['train_mse'], marker='o', label='Train MSE')
plt.plot(results_df['learning_rate'], results_df['test_mse'], marker='s', label='Test MSE')
plt.xlabel('Learning Rate')
plt.ylabel('MSE')
plt.title('Learning Rate의 영향')
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.show()

# 실전 팁:
# - Learning Rate 작게 + n_estimators 크게 = 성능 좋음 (but 느림)
# - Learning Rate 0.01-0.1 추천
# - Early stopping으로 최적 iteration 찾기
```

---

## 3. XGBoost: Extreme Gradient Boosting

### 3.1 XGBoost의 혁신

```python
# 의도: XGBoost가 왜 Kaggle을 지배했는지 이해
# 아이디어: Regularization + 2차 미분 + 병렬화

"""
Gradient Boosting vs XGBoost:

1. 목적 함수에 Regularization 추가:
   Loss = Σ L(y_i, pred_i) + Σ Ω(f_k)
   Ω(f) = γT + (1/2)λΣw_j^2
   → Overfitting 방지

2. 2차 미분 사용 (Taylor Expansion):
   1차: Gradient Boosting
   2차: XGBoost → 더 정확한 최적화

3. 병렬 처리:
   - Feature별로 병렬로 best split 찾기
   - GPU 지원

4. Sparse data 최적화:
   - 결측치 자동 처리
   - 희소 행렬 효율적 처리
"""

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 데이터 로드
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# XGBoost 학습
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,            # Row sampling (80% 샘플)
    colsample_bytree=0.8,     # Column sampling (80% feature)
    reg_alpha=0.1,            # L1 regularization
    reg_lambda=1.0,           # L2 regularization
    gamma=0.1,                # 분할 최소 loss 감소
    min_child_weight=1,       # 리프 노드 최소 weight
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='logloss',
    verbose=10  # 10 iteration마다 출력
)

# 평가
from sklearn.metrics import accuracy_score, roc_auc_score

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

### 3.2 XGBoost 하이퍼파라미터 완벽 가이드

```python
# 의도: XGBoost 하이퍼파라미터 이해
# 아이디어: 각 파라미터의 역할과 튜닝 전략

"""
XGBoost 하이퍼파라미터 분류:

1. Tree 구조 (Overfitting 제어)
━━━━━━━━━━━━━━━━━━━━━━
max_depth: 트리 최대 깊이 (기본 6)
  - 클수록 복잡 → Overfitting
  - 추천: 3-10

min_child_weight: 리프 노드 최소 weight (기본 1)
  - 클수록 보수적 → Underfitting
  - 추천: 1-10

gamma: 분할 최소 loss 감소 (기본 0)
  - 클수록 보수적
  - 추천: 0-0.5

max_leaves: 최대 리프 노드 수
  - max_depth 대신 사용 가능

2. Sampling (다양성 증가)
━━━━━━━━━━━━━━━━━━━━━━
subsample: Row sampling 비율 (기본 1.0)
  - 0.8 추천 (80% 샘플만 사용)

colsample_bytree: Feature sampling (트리당) (기본 1.0)
  - 0.8 추천

colsample_bylevel: Feature sampling (레벨당)
  - 각 깊이마다 feature 비율

colsample_bynode: Feature sampling (노드당)
  - 각 split마다 feature 비율

3. Regularization (Overfitting 방지)
━━━━━━━━━━━━━━━━━━━━━━
reg_alpha: L1 정규화 (기본 0)
  - Sparse feature에 유용

reg_lambda: L2 정규화 (기본 1)
  - 기본 regularization

4. Learning Control
━━━━━━━━━━━━━━━━━━━━━━
n_estimators: 트리 개수 (기본 100)
  - 많을수록 좋지만 overfitting 위험
  - Early stopping으로 최적값 찾기

learning_rate (eta): 학습률 (기본 0.3)
  - 작을수록 천천히 학습
  - 추천: 0.01-0.1

5. 기타
━━━━━━━━━━━━━━━━━━━━━━
scale_pos_weight: 클래스 불균형 처리
  - (음성 샘플 수) / (양성 샘플 수)

tree_method: 트리 구축 알고리즘
  - 'auto', 'exact', 'approx', 'hist', 'gpu_hist'

predictor: 예측 알고리즘
  - 'cpu_predictor', 'gpu_predictor'
"""

# 실전 튜닝 전략 (3단계)
from sklearn.model_selection import GridSearchCV

# 1단계: Tree 구조 튜닝
param_grid_1 = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5]
}

grid_1 = GridSearchCV(
    xgb.XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=42),
    param_grid_1,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_1.fit(X_train, y_train)
print(f"Step 1 Best Params: {grid_1.best_params_}")
print(f"Step 1 Best Score: {grid_1.best_score_:.4f}\n")

# 2단계: Sampling 튜닝
param_grid_2 = {
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

grid_2 = GridSearchCV(
    xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        **grid_1.best_params_,  # Step 1 결과 사용
        random_state=42
    ),
    param_grid_2,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_2.fit(X_train, y_train)
print(f"Step 2 Best Params: {grid_2.best_params_}")
print(f"Step 2 Best Score: {grid_2.best_score_:.4f}\n")

# 3단계: Regularization 튜닝
param_grid_3 = {
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0.1, 1.0, 10.0]
}

grid_3 = GridSearchCV(
    xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        **grid_1.best_params_,
        **grid_2.best_params_,
        random_state=42
    ),
    param_grid_3,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_3.fit(X_train, y_train)
print(f"Step 3 Best Params: {grid_3.best_params_}")
print(f"Step 3 Best Score: {grid_3.best_score_:.4f}\n")

# 최종 모델
final_model = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    **grid_1.best_params_,
    **grid_2.best_params_,
    **grid_3.best_params_,
    random_state=42
)
final_model.fit(X_train, y_train)

# 실전 팁: Optuna로 자동 튜닝 (더 효율적)
```

### 3.3 Early Stopping & CV

```python
# 의도: Overfitting 방지 - Early Stopping
# 아이디어: Validation loss가 개선 안 되면 중단

from sklearn.model_selection import train_test_split

# Train/Val 분할
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Early Stopping
model = xgb.XGBClassifier(
    n_estimators=1000,  # 충분히 크게 설정
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='logloss',
    early_stopping_rounds=20,  # 20 iteration 동안 개선 없으면 중단
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score:.4f}")

# CV with Early Stopping
from xgboost import cv as xgb_cv

dtrain = xgb.DMatrix(X_train, label=y_train)

params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

cv_results = xgb_cv(
    params,
    dtrain,
    num_boost_round=1000,
    nfold=5,
    early_stopping_rounds=20,
    verbose_eval=50
)

print(f"\nBest iteration (CV): {len(cv_results)}")
print(f"Best score (CV): {cv_results['test-logloss-mean'].min():.4f}")
```

---

## 4. LightGBM: Light Gradient Boosting Machine

### 4.1 LightGBM의 핵심 혁신

```python
# 의도: LightGBM이 XGBoost보다 10-100배 빠른 이유
# 아이디어: Histogram-based + Leaf-wise 성장

"""
XGBoost vs LightGBM:

1. Split 알고리즘:
   XGBoost: Exact (정확하지만 느림)
   LightGBM: Histogram-based (빠름)
     - 연속값을 bins로 변환 (e.g., 256 bins)
     - 메모리: 1/8, 속도: 10배

2. Tree 성장 방식:
   XGBoost: Level-wise (깊이 우선)
     Depth 1:     [Root]
     Depth 2:   [  ][  ]
     Depth 3: [ ][ ][ ][ ]

   LightGBM: Leaf-wise (loss 감소 우선)
     [Root]
       ↓ (가장 큰 loss 감소)
     [Left]
       ↓ (다음으로 큰 loss 감소)
     [Left-Left]
     ...
     → 더 깊어질 수 있지만 성능 좋음

3. Categorical Feature 자동 처리:
   XGBoost: One-hot encoding 필요
   LightGBM: 자동 최적 분할

4. 메모리 효율:
   - Gradient-based One-Side Sampling (GOSS)
   - Exclusive Feature Bundling (EFB)
"""

import lightgbm as lgb

# LightGBM 학습
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,            # Leaf-wise → num_leaves 중요
    min_child_samples=20,     # 리프 노드 최소 샘플
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='logloss',
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(10)]
)

# 평가
y_pred_lgb = lgb_model.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")

# 속도 비교
import time

# XGBoost
start = time.time()
xgb_model = xgb.XGBClassifier(n_estimators=100, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start

# LightGBM
start = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=100, n_jobs=-1)
lgb_model.fit(X_train, y_train)
lgb_time = time.time() - start

print(f"\nXGBoost 학습 시간: {xgb_time:.2f}s")
print(f"LightGBM 학습 시간: {lgb_time:.2f}s")
print(f"속도 향상: {xgb_time/lgb_time:.1f}배")
```

### 4.2 LightGBM Categorical Feature 처리

```python
# 의도: LightGBM의 killer feature - categorical 자동 처리
# 아이디어: One-hot 없이 직접 최적 분할

import pandas as pd
import numpy as np

# 범주형 feature 포함 데이터
df = pd.DataFrame({
    'age': np.random.randint(20, 60, 1000),
    'income': np.random.randint(30000, 150000, 1000),
    'city': np.random.choice(['서울', '부산', '대구', '인천', '광주'], 1000),
    'education': np.random.choice(['고졸', '대졸', '석사', '박사'], 1000),
    'job': np.random.choice(['개발자', '디자이너', '마케터', '영업', '기획자'], 1000),
    'target': np.random.randint(0, 2, 1000)
})

# 범주형 변수 지정
categorical_features = ['city', 'education', 'job']

# Label Encoding (LightGBM 요구사항)
from sklearn.preprocessing import LabelEncoder

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('target', axis=1)
y = df['target']

# LightGBM with Categorical Features
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    random_state=42
)

lgb_model.fit(
    X, y,
    categorical_feature=categorical_features  # 범주형 feature 지정
)

print(f"Accuracy: {lgb_model.score(X, y):.4f}")

# 실전 팁:
"""
1. categorical_feature 지정 시 자동 최적 분할
   - One-hot보다 메모리 효율적
   - Cardinality 높아도 OK (e.g., 1000개 카테고리)

2. XGBoost는 one-hot 필요
   → 고차원 sparse matrix → 느림

3. 실무 벤치마크:
   - LightGBM: 1분
   - XGBoost (one-hot): 15분
"""
```

---

## 5. CatBoost: Categorical Boosting

### 5.1 CatBoost의 독특한 기능

```python
# 의도: CatBoost의 차별점
# 아이디어: Target Leakage 방지 + Ordered Boosting

"""
CatBoost 특징:

1. Categorical Features 자동 처리:
   - Label Encoding도 필요 없음!
   - 문자열 그대로 입력 가능

2. Target Encoding (Ordered Target Statistics):
   - Target leakage 방지
   - 순서대로 statistics 계산

3. Ordered Boosting:
   - Prediction shift 방지
   - 더 robust한 모델

4. GPU 최적화:
   - 대규모 데이터에 매우 빠름

5. 기본 하이퍼파라미터가 우수:
   - 튜닝 거의 필요 없음
"""

from catboost import CatBoostClassifier

# 범주형 feature 포함 데이터 (문자열 그대로)
df_cat = pd.DataFrame({
    'age': np.random.randint(20, 60, 1000),
    'income': np.random.randint(30000, 150000, 1000),
    'city': np.random.choice(['서울', '부산', '대구', '인천', '광주'], 1000),
    'education': np.random.choice(['고졸', '대졸', '석사', '박사'], 1000),
    'job': np.random.choice(['개발자', '디자이너', '마케터', '영업', '기획자'], 1000),
    'target': np.random.randint(0, 2, 1000)
})

X_cat = df_cat.drop('target', axis=1)
y_cat = df_cat['target']

# CatBoost 학습
cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    cat_features=['city', 'education', 'job'],  # 범주형 feature 지정 (문자열 그대로)
    random_state=42,
    verbose=10
)

cat_model.fit(X_cat, y_cat)

print(f"CatBoost Accuracy: {cat_model.score(X_cat, y_cat):.4f}")

# 실전 팁: Encoding 없이 바로 학습 가능!
```

### 5.2 Ordered Target Statistics (Target Encoding)

```python
# 의도: CatBoost의 Target Encoding 이해
# 아이디어: Target leakage 방지

"""
일반 Target Encoding (문제 있음):
City='서울'인 샘플 10개, Target=[1,1,1,0,0,0,0,0,0,0]
→ 서울 평균 target = 0.3
→ 모든 서울 샘플에 0.3 할당

문제: Target leakage!
→ 현재 샘플의 target이 encoding에 포함됨

CatBoost Ordered Target Statistics:
순서대로 계산:
  Sample 1 (서울): prior (0.5)
  Sample 2 (서울): 1개 이전 샘플 평균 = 1.0
  Sample 3 (서울): 2개 이전 샘플 평균 = 1.0
  Sample 4 (서울): 3개 이전 샘플 평균 = 1.0
  Sample 5 (서울): 4개 이전 샘플 평균 = 0.75
  ...

→ 현재 샘플 제외하고 이전 샘플만 사용!
"""

# Target Encoding 직접 구현 (CatBoost 방식)
class OrderedTargetEncoder:
    """CatBoost-style Target Encoder"""

    def __init__(self, prior=0.5):
        self.prior = prior
        self.statistics = {}

    def fit_transform(self, X, y, categorical_cols):
        """순서대로 target statistics 계산"""

        X_encoded = X.copy()

        for col in categorical_cols:
            running_stats = {}  # {category: [target_sum, count]}
            encoded_values = []

            for idx in range(len(X)):
                category = X.iloc[idx][col]

                if category not in running_stats:
                    # 처음 보는 카테고리 → prior 사용
                    encoded_values.append(self.prior)
                    running_stats[category] = [y.iloc[idx], 1]
                else:
                    # 이전 샘플들의 평균
                    target_sum, count = running_stats[category]
                    mean_target = target_sum / count
                    encoded_values.append(mean_target)

                    # 현재 샘플 추가
                    running_stats[category][0] += y.iloc[idx]
                    running_stats[category][1] += 1

            X_encoded[col] = encoded_values

        return X_encoded

# 테스트
encoder = OrderedTargetEncoder(prior=0.5)
X_encoded = encoder.fit_transform(df_cat[['city']], df_cat['target'], ['city'])

print("Ordered Target Encoding:")
print(X_encoded.head(20))
```

---

## 6. XGBoost vs LightGBM vs CatBoost 비교

### 6.1 종합 벤치마크

```python
import time
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# 대규모 데이터 생성
X, y = make_classification(
    n_samples=100000,
    n_features=100,
    n_informative=50,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# 1. XGBoost
print("Training XGBoost...")
start = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start

xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_pred)

results.append({
    'Model': 'XGBoost',
    'Train Time (s)': round(xgb_time, 2),
    'ROC-AUC': round(xgb_auc, 4),
    'Memory (MB)': 'High'
})

# 2. LightGBM
print("Training LightGBM...")
start = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_time = time.time() - start

lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
lgb_auc = roc_auc_score(y_test, lgb_pred)

results.append({
    'Model': 'LightGBM',
    'Train Time (s)': round(lgb_time, 2),
    'ROC-AUC': round(lgb_auc, 4),
    'Memory (MB)': 'Low'
})

# 3. CatBoost
print("Training CatBoost...")
start = time.time()
cat_model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)
cat_model.fit(X_train, y_train)
cat_time = time.time() - start

cat_pred = cat_model.predict_proba(X_test)[:, 1]
cat_auc = roc_auc_score(y_test, cat_pred)

results.append({
    'Model': 'CatBoost',
    'Train Time (s)': round(cat_time, 2),
    'ROC-AUC': round(cat_auc, 4),
    'Memory (MB)': 'Medium'
})

# 결과 출력
results_df = pd.DataFrame(results)
print("\n=== Benchmark Results ===")
print(results_df.to_string(index=False))

print(f"\n속도 비교:")
print(f"  LightGBM이 XGBoost보다 {xgb_time/lgb_time:.1f}배 빠름")
print(f"  LightGBM이 CatBoost보다 {cat_time/lgb_time:.1f}배 빠름")

"""
예상 출력:
      Model  Train Time (s)  ROC-AUC Memory (MB)
   XGBoost           12.5   0.9234        High
  LightGBM            2.3   0.9241         Low
  CatBoost            8.7   0.9238      Medium

속도 비교:
  LightGBM이 XGBoost보다 5.4배 빠름
  LightGBM이 CatBoost보다 3.8배 빠름
"""
```

### 6.2 선택 가이드

```python
"""
언제 어떤 모델을 쓸까?

1. LightGBM 추천:
   ✅ 대규모 데이터 (>100k samples)
   ✅ 빠른 학습 필요
   ✅ 메모리 제약
   ✅ Categorical feature 많음 (but Label Encoding 필요)

   실무 점유율: 60%

2. XGBoost 추천:
   ✅ 중소규모 데이터 (<100k samples)
   ✅ Kaggle 앙상블 (다양성)
   ✅ Sparse data (NLP, 추천시스템)

   실무 점유율: 30%

3. CatBoost 추천:
   ✅ Categorical feature 매우 많음
   ✅ 튜닝 시간 없음 (기본값 우수)
   ✅ Robust한 모델 필요
   ✅ Target leakage 우려

   실무 점유율: 10%

성능 (ROC-AUC):
  XGBoost:  ★★★★☆
  LightGBM: ★★★★★
  CatBoost: ★★★★★

속도:
  XGBoost:  ★★☆☆☆
  LightGBM: ★★★★★
  CatBoost: ★★★☆☆

메모리:
  XGBoost:  ★★☆☆☆
  LightGBM: ★★★★★
  CatBoost: ★★★☆☆

사용 편의성:
  XGBoost:  ★★★☆☆
  LightGBM: ★★★★☆
  CatBoost: ★★★★★ (Categorical 자동 처리)
"""

# 실무 결정 트리
def choose_model(data_size, n_categorical, time_budget, need_interpretation):
    """모델 선택 가이드"""

    if n_categorical > 10 and time_budget == 'low':
        return "CatBoost (Categorical 자동 처리 + 빠른 튜닝)"

    if data_size > 100000:
        return "LightGBM (대규모 데이터 특화)"

    if need_interpretation and n_categorical < 5:
        return "XGBoost (안정적 + Feature Importance 신뢰성)"

    return "LightGBM (범용적으로 가장 좋음)"

# 예시
print(choose_model(data_size=200000, n_categorical=15, time_budget='low', need_interpretation=False))
# → "LightGBM (대규모 데이터 특화)"

print(choose_model(data_size=10000, n_categorical=20, time_budget='low', need_interpretation=True))
# → "CatBoost (Categorical 자동 처리 + 빠른 튜닝)"
```

---

## 7. 실전 최적화 기법

### 7.1 Optuna로 자동 하이퍼파라미터 튜닝

```python
import optuna
from sklearn.model_selection import cross_val_score

# 의도: GridSearch보다 10-100배 빠른 Bayesian Optimization
# 아이디어: 좋은 영역을 집중 탐색

def objective(trial):
    """Optuna objective function"""

    # 하이퍼파라미터 샘플링
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # LightGBM 모델
    model = lgb.LGBMClassifier(**param, random_state=42, n_jobs=-1, verbose=-1)

    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)

    return scores.mean()

# Optuna 최적화
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best Score: {study.best_value:.4f}")
print(f"Best Params: {study.best_params}")

# 최적 모델 학습
best_model = lgb.LGBMClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)

# 실전 팁: n_trials=50이면 GridSearch 1000번보다 좋은 결과
```

### 7.2 Feature Engineering for Tree Models

```python
# 의도: Tree 모델에 최적화된 Feature Engineering
# 아이디어: Tree는 선형 관계 못 배움 → 명시적 특징 필요

import pandas as pd
import numpy as np

def engineer_features_for_trees(df):
    """Tree 모델용 Feature Engineering"""

    df_new = df.copy()

    # 1. Binning (연속형 → 범주형)
    #    Tree는 threshold 기반이므로 binning이 도움됨
    df_new['age_group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 100],
                                   labels=['10s', '20s', '30s', '40s', '50s+'])

    # 2. Interaction Features (상호작용)
    #    Tree가 자동으로 찾지만, 명시적으로 주면 더 빠름
    df_new['income_per_age'] = df['income'] / (df['age'] + 1)
    df_new['high_income_young'] = ((df['income'] > 100000) & (df['age'] < 30)).astype(int)

    # 3. Aggregation Features (그룹별 통계)
    #    city별 평균 income
    city_income_mean = df.groupby('city')['income'].mean()
    df_new['city_income_mean'] = df['city'].map(city_income_mean)

    # 4. Count Encoding (범주 빈도)
    #    빈도가 target과 상관있을 때 유용
    city_counts = df['city'].value_counts()
    df_new['city_count'] = df['city'].map(city_counts)

    # 5. Ratio Features
    df_new['income_to_city_mean_ratio'] = df['income'] / df_new['city_income_mean']

    return df_new

# 실전 팁:
"""
Tree 모델은 Feature Scaling 불필요!
❌ StandardScaler, MinMaxScaler → 필요 없음
✅ Binning, Interaction, Aggregation → 매우 유용
"""
```

---

## 핵심 정리

### 모델 선택 플로우차트

```
데이터 크기는?
├─ <10k samples → XGBoost
├─ 10k-100k samples → XGBoost or LightGBM
└─ >100k samples → LightGBM

Categorical Features 많은가? (>10개)
├─ Yes → CatBoost or LightGBM
└─ No → LightGBM

GPU 있는가?
├─ Yes → LightGBM (gpu_hist) or CatBoost
└─ No → LightGBM (cpu)

튜닝 시간 있는가?
├─ Yes → XGBoost or LightGBM + Optuna
└─ No → CatBoost (기본값 우수)

결론: 80%의 경우 LightGBM이 정답!
```

### 하이퍼파라미터 기본값 추천

```python
# LightGBM 기본 설정 (실무용)
lgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,  # num_leaves로 제어
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

# Early stopping 필수
model = lgb.LGBMClassifier(**lgb_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50)]
)
```

다음: SVM, KNN, Naive Bayes →
