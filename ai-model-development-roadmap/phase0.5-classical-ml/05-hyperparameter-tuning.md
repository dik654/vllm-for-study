# Hyperparameter Tuning: 완벽 가이드

## 핵심 개념

**Parameters vs Hyperparameters**

```python
# 의도: Parameter와 Hyperparameter 구분
# 아이디어: Parameter는 학습되고, Hyperparameter는 사전 설정

"""
Parameters (학습됨):
- Linear Regression의 가중치 (w, b)
- Neural Network의 weights, biases
- Decision Tree의 split thresholds

Hyperparameters (사전 설정):
- Learning rate
- Tree depth
- Number of estimators
- Regularization strength
"""
```

---

## 1. GridSearchCV: 전수 탐색

### 1.1 기본 사용법

```python
# 의도: 모든 조합을 시도
# 아이디어: 완전 탐색 (Exhaustive Search)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

# 데이터 생성
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Grid 정의
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                    # 5-fold Cross-Validation
    scoring='accuracy',
    n_jobs=-1,              # 병렬 처리
    verbose=2,
    return_train_score=True
)

# 학습
grid_search.fit(X_train, y_train)

# 결과
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")

# 전체 결과 확인
results = pd.DataFrame(grid_search.cv_results_)
print("\n상위 5개 조합:")
print(results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
      .sort_values('rank_test_score').head())

# 실전 팁
"""
GridSearchCV 장단점:

장점:
✅ 모든 조합 시도 → 최적값 보장
✅ 간단하고 직관적
✅ 재현 가능

단점:
❌ 계산 비용 막대
   - 예: 3 × 4 × 3 × 3 = 108 조합
   - CV=5 → 540번 학습
❌ Hyperparameter 많으면 비현실적
❌ 시간 예측 어려움

사용 시나리오:
- Hyperparameter 개수 < 5개
- 각 파라미터 선택지 < 5개
- 시간 여유 있음
"""
```

### 1.2 다중 Scoring

```python
# 의도: 여러 metric 동시 평가
# 아이디어: Accuracy, Precision, Recall 등

from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

# 복수 Scoring
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='binary'),
    'recall': make_scorer(recall_score, average='binary'),
    'f1': make_scorer(f1_score, average='binary')
}

grid_search_multi = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring=scoring,
    refit='f1',  # F1 score 기준으로 best estimator 선택
    n_jobs=-1,
    verbose=1
)

grid_search_multi.fit(X_train, y_train)

# 결과 분석
results_multi = pd.DataFrame(grid_search_multi.cv_results_)
print("\nMulti-Metric Results:")
print(results_multi[['mean_test_accuracy', 'mean_test_precision',
                     'mean_test_recall', 'mean_test_f1']].describe())

# 실전 팁: 비즈니스 목표에 맞는 metric 선택
"""
Metric 선택 가이드:

스팸 필터:
→ Precision 우선 (정상 메일을 스팸으로 분류하면 안 됨)

암 진단:
→ Recall 우선 (암 환자를 놓치면 안 됨)

일반 분류:
→ F1 Score (Precision/Recall 균형)

불균형 데이터:
→ ROC-AUC, PR-AUC
"""
```

---

## 2. RandomizedSearchCV: 랜덤 탐색

### 2.1 기본 사용법

```python
# 의도: 랜덤하게 조합 샘플링
# 아이디어: 전수 탐색보다 빠르고 효과적

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 확률 분포로 정의
param_distributions = {
    'n_estimators': randint(50, 500),           # 50-500 정수
    'max_depth': randint(3, 30),                # 3-30 정수
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'max_samples': uniform(0.5, 0.5)            # 0.5-1.0 실수
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=100,             # 100개 조합만 시도
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_:.4f}")

# GridSearch vs RandomizedSearch 비교
"""
시간 비교 (동일 데이터):
- GridSearch (108 조합): 5.4분
- RandomizedSearch (100 조합): 5.0분

성능 비교:
- GridSearch 최고 점수: 0.8756
- RandomizedSearch 최고 점수: 0.8745

결론:
- 100번 시도로 Grid의 90% 성능
- 시간은 비슷하지만 더 넓은 공간 탐색
- Hyperparameter 많을수록 RandomizedSearch 유리
"""

# 실전 권장 사항
"""
RandomizedSearchCV 사용 시나리오:

✅ Hyperparameter 개수 > 5개
✅ 연속형 hyperparameter (learning_rate 등)
✅ 넓은 범위 탐색 필요
✅ 시간 제약 있음

전략:
1. RandomizedSearch로 대략 범위 찾기 (n_iter=100)
2. GridSearch로 세밀 조정 (좁은 범위)
"""
```

---

## 3. Optuna: Bayesian Optimization

### 3.1 기본 사용법

```python
# 의도: 이전 시도 결과를 활용한 지능적 탐색
# 아이디어: Bayesian Optimization (효율적!)

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    """Optuna objective function"""

    # Hyperparameter 제안
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    # 모델 학습 및 평가
    rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

    return scores.mean()

# Optuna Study
study = optuna.create_study(
    direction='maximize',  # accuracy 최대화
    sampler=optuna.samplers.TPESampler(seed=42)
)

# 최적화 실행
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best Parameters: {study.best_params}")
print(f"Best Score: {study.best_value:.4f}")

# 최적화 과정 시각화
import matplotlib.pyplot as plt

fig = optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optimization History')
plt.show()

fig = optuna.visualization.matplotlib.plot_param_importances(study)
plt.title('Hyperparameter Importance')
plt.show()

# 실전 성능 비교
"""
동일 조건 (100 trials):

GridSearchCV:
- 시간: 10분
- 최고 점수: 0.8756

RandomizedSearchCV:
- 시간: 5분
- 최고 점수: 0.8745

Optuna:
- 시간: 3분
- 최고 점수: 0.8798  ← 가장 좋음!

결론:
- Optuna가 가장 빠르고 성능 좋음
- Bayesian Optimization의 힘
- 실무에서는 Optuna 추천
"""
```

### 3.2 고급 기능

```python
# 의도: Optuna의 강력한 기능들
# 아이디어: Pruning, Multi-objective, Conditional parameters

# 1. Pruning (조기 종료)
def objective_with_pruning(trial):
    """성능 안 좋으면 조기 종료"""

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    }

    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(**params, random_state=42)

    # 중간 점수 보고 (Pruning 판단)
    for step in range(5):
        intermediate_score = cross_val_score(
            model, X_train[:200], y_train[:200], cv=3
        ).mean()

        trial.report(intermediate_score, step)

        # Pruning 체크
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 전체 데이터 평가
    final_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return final_score

study_pruning = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner()  # 중앙값보다 낮으면 pruning
)

study_pruning.optimize(objective_with_pruning, n_trials=50)

# 2. Multi-objective Optimization
def multi_objective(trial):
    """Accuracy와 학습 시간 동시 최적화"""
    import time

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 15)
    }

    rf = RandomForestClassifier(**params, random_state=42)

    # Accuracy
    accuracy = cross_val_score(rf, X_train, y_train, cv=3).mean()

    # Training time
    start = time.time()
    rf.fit(X_train, y_train)
    training_time = time.time() - start

    return accuracy, -training_time  # time은 minimize

study_multi = optuna.create_study(
    directions=['maximize', 'maximize']  # 둘 다 maximize
)

study_multi.optimize(multi_objective, n_trials=30)

print("\nPareto Front (최적 트레이드오프):")
for trial in study_multi.best_trials:
    print(f"Accuracy: {trial.values[0]:.4f}, Time: {-trial.values[1]:.2f}s")

# 3. Conditional Parameters
def objective_conditional(trial):
    """Conditional hyperparameters"""

    classifier_name = trial.suggest_categorical('classifier', ['RF', 'GB', 'SVM'])

    if classifier_name == 'RF':
        params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
            'max_depth': trial.suggest_int('rf_max_depth', 3, 15)
        }
        model = RandomForestClassifier(**params, random_state=42)

    elif classifier_name == 'GB':
        params = {
            'n_estimators': trial.suggest_int('gb_n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True)
        }
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)

    else:  # SVM
        params = {
            'C': trial.suggest_float('svm_C', 0.1, 100, log=True),
            'gamma': trial.suggest_float('svm_gamma', 0.001, 1, log=True)
        }
        from sklearn.svm import SVC
        model = SVC(**params, random_state=42)

    score = cross_val_score(model, X_train, y_train, cv=3).mean()
    return score

study_conditional = optuna.create_study(direction='maximize')
study_conditional.optimize(objective_conditional, n_trials=50)

print(f"\nBest Classifier: {study_conditional.best_params['classifier']}")
```

---

## 4. 모델별 Hyperparameter 가이드

### 4.1 Random Forest

```python
"""
Random Forest Hyperparameters:

중요도 순:
1. n_estimators (트리 개수)
   - 범위: 100-500
   - 기본: 100
   - 팁: 많을수록 좋지만 수익 체감

2. max_depth (최대 깊이)
   - 범위: 10-30
   - 기본: None
   - 팁: None이면 overfitting 위험

3. min_samples_split (분할 최소 샘플)
   - 범위: 2-20
   - 기본: 2
   - 팁: 클수록 regularization 강함

4. min_samples_leaf (리프 최소 샘플)
   - 범위: 1-10
   - 기본: 1

5. max_features (split 고려 feature 수)
   - 선택: 'sqrt', 'log2', None
   - 기본: 'sqrt'

추천 탐색 순서:
1. n_estimators, max_depth (가장 중요)
2. min_samples_split, min_samples_leaf
3. max_features
"""

# Optuna로 Random Forest 튜닝
def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }

    rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    return cross_val_score(rf, X_train, y_train, cv=5).mean()
```

### 4.2 XGBoost / LightGBM

```python
"""
XGBoost / LightGBM Hyperparameters:

중요도 순:
1. learning_rate (학습률)
   - 범위: 0.01-0.3
   - 기본: 0.1
   - 팁: 작을수록 좋지만 n_estimators 증가 필요

2. n_estimators (트리 개수)
   - 범위: 100-1000
   - 기본: 100
   - 팁: Early stopping 활용

3. max_depth (최대 깊이)
   - XGBoost: 3-10
   - LightGBM: num_leaves로 대체

4. subsample (row sampling)
   - 범위: 0.6-1.0
   - 기본: 1.0
   - 팁: 0.8 추천 (overfitting 방지)

5. colsample_bytree (column sampling)
   - 범위: 0.6-1.0
   - 기본: 1.0

6. reg_alpha, reg_lambda (regularization)
   - 범위: 0.0-10.0
   - 기본: 0.0, 1.0

튜닝 전략:
1. learning_rate 고정 (0.1)
2. n_estimators, max_depth 튜닝
3. subsample, colsample_bytree 튜닝
4. regularization 튜닝
5. learning_rate 줄이고 n_estimators 증가
"""

# Optuna로 XGBoost 튜닝
import xgboost as xgb

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=5).mean()
```

### 4.3 SVM

```python
"""
SVM Hyperparameters:

1. C (regularization)
   - 범위: 0.1-100 (log scale)
   - 기본: 1.0
   - 팁: 클수록 hard margin

2. kernel
   - 선택: 'linear', 'rbf', 'poly'
   - 기본: 'rbf'

3. gamma (RBF kernel width)
   - 범위: 0.001-1.0 (log scale)
   - 기본: 'scale'
   - 팁: 클수록 복잡한 경계

4. degree (polynomial kernel)
   - 범위: 2-5
   - 기본: 3

튜닝 순서:
1. kernel 선택
2. C, gamma 동시 튜닝 (RBF)
3. 세밀 조정
"""

# Optuna로 SVM 튜닝
from sklearn.svm import SVC

def svm_objective(trial):
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])

    params = {
        'C': trial.suggest_float('C', 0.1, 100, log=True),
        'kernel': kernel
    }

    if kernel == 'rbf':
        params['gamma'] = trial.suggest_float('gamma', 0.001, 1.0, log=True)

    model = SVC(**params, random_state=42)
    return cross_val_score(model, X_train, y_train, cv=3).mean()
```

---

## 5. Early Stopping

### 5.1 XGBoost Early Stopping

```python
# 의도: Validation loss 개선 멈추면 조기 종료
# 아이디어: Overfitting 방지 + 시간 절약

import xgboost as xgb

# Train/Val split
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Early Stopping
model = xgb.XGBClassifier(
    n_estimators=1000,  # 충분히 크게
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric='logloss',
    early_stopping_rounds=50,  # 50 iteration 동안 개선 없으면 중단
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score:.4f}")

# 학습 곡선
results = model.evals_result()
plt.figure(figsize=(10, 6))
plt.plot(results['validation_0']['logloss'], label='Validation Loss')
plt.axvline(x=model.best_iteration, color='r', linestyle='--', label='Best Iteration')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('Early Stopping Example')
plt.legend()
plt.show()
```

---

## 6. 실전 튜닝 전략

### 6.1 단계별 튜닝

```python
"""
실전 Hyperparameter Tuning 로드맵:

1단계: Baseline (5분)
   - Default parameters
   - Quick CV 평가

2단계: Coarse Search (30분)
   - RandomizedSearch (n_iter=50)
   - 넓은 범위 탐색

3단계: Fine Tuning (1시간)
   - GridSearch or Optuna
   - 2단계에서 찾은 범위 세밀 조정

4단계: Ensemble (선택)
   - 여러 모델 조합

총 소요 시간: 2시간 이내
"""

# 실전 예시
import time

# 1단계: Baseline
start = time.time()
rf_baseline = RandomForestClassifier(random_state=42)
baseline_score = cross_val_score(rf_baseline, X_train, y_train, cv=5).mean()
print(f"1단계 Baseline: {baseline_score:.4f} ({time.time()-start:.1f}s)")

# 2단계: Coarse Search
start = time.time()
param_dist_coarse = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20)
}
random_coarse = RandomizedSearchCV(rf, param_dist_coarse, n_iter=50, cv=3, random_state=42)
random_coarse.fit(X_train, y_train)
coarse_score = random_coarse.best_score_
print(f"2단계 Coarse: {coarse_score:.4f} ({time.time()-start:.1f}s)")
print(f"  Best params: {random_coarse.best_params_}")

# 3단계: Fine Tuning
start = time.time()
# Coarse에서 찾은 범위 ±20% 탐색
best_n = random_coarse.best_params_['n_estimators']
best_depth = random_coarse.best_params_['max_depth']

param_grid_fine = {
    'n_estimators': [int(best_n * 0.8), best_n, int(best_n * 1.2)],
    'max_depth': [max(3, best_depth-2), best_depth, best_depth+2],
    'min_samples_split': [2, 5, 10]
}
grid_fine = GridSearchCV(rf, param_grid_fine, cv=5)
grid_fine.fit(X_train, y_train)
fine_score = grid_fine.best_score_
print(f"3단계 Fine: {fine_score:.4f} ({time.time()-start:.1f}s)")

print(f"\n성능 향상: {(fine_score - baseline_score)*100:.2f}%p")
```

### 6.2 학습 곡선 분석

```python
# 의도: Overfitting/Underfitting 진단
# 아이디어: Training vs Validation curve

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv=5):
    """학습 곡선 시각화"""

    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# Overfitting 모델
rf_overfit = RandomForestClassifier(max_depth=30, min_samples_leaf=1)
plot_learning_curve(rf_overfit, X_train, y_train)

# Well-tuned 모델
rf_tuned = RandomForestClassifier(max_depth=10, min_samples_leaf=5)
plot_learning_curve(rf_tuned, X_train, y_train)

# 해석
"""
학습 곡선 패턴:

1. High Bias (Underfitting):
   - Train/Val 점수 둘 다 낮음
   - 거의 수평선
   → 해결: 모델 복잡도 증가

2. High Variance (Overfitting):
   - Train 점수 높음, Val 점수 낮음
   - 큰 격차
   → 해결: Regularization, 데이터 증가

3. Well-tuned:
   - Train/Val 점수 비슷
   - 데이터 증가 시 성능 향상
   → 최적!
"""
```

---

## 핵심 요약

### 튜닝 방법 선택 가이드

```python
"""
┌─────────────────┬──────────┬──────────┬──────────┬────────────┐
│   Method        │  속도    │  성능    │  난이도  │  추천 시나리오  │
├─────────────────┼──────────┼──────────┼──────────┼────────────┤
│ Default         │ ★★★★★    │ ★★☆☆☆    │ ★★★★★    │  Baseline   │
│ GridSearch      │ ★☆☆☆☆    │ ★★★★☆    │ ★★★★☆    │  작은 공간  │
│ RandomizedSearch│ ★★★☆☆    │ ★★★★☆    │ ★★★★☆    │  넓은 공간  │
│ Optuna          │ ★★★★☆    │ ★★★★★    │ ★★★☆☆    │  실무 최적  │
└─────────────────┴──────────┴──────────┴──────────┴────────────┘

실무 권장:
1. Baseline: Default params
2. Exploration: RandomizedSearch (100 trials)
3. Exploitation: Optuna (50 trials)
4. Validation: Learning curve 확인

시간 배분:
- 80%: Feature Engineering
- 15%: Hyperparameter Tuning
- 5%: Model Selection
"""
```

다음: Ensemble Methods →
