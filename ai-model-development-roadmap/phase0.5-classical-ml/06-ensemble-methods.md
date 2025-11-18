# Ensemble Methods: 앙상블 기법 완벽 가이드

## 핵심 개념

**"여러 개의 약한 모델 > 하나의 강한 모델"**

```python
# 의도: Ensemble의 힘
# 아이디어: Wisdom of Crowds (집단 지성)

"""
단일 모델:
- 정확도: 85%
- 불안정함
- 편향 가능

앙상블 (5개 모델):
- 정확도: 92%
- 안정적
- 편향 감소

실무 사례:
- Kaggle 우승 솔루션의 95%가 앙상블 사용
- Netflix Prize: 앙상블로 10% 성능 향상
- 구글 검색: 수백 개 모델 앙상블
"""
```

---

## 1. Voting (투표 방식)

### 1.1 Hard Voting (다수결)

```python
# 의도: 가장 단순한 앙상블
# 아이디어: 다수결 투표

from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터 생성
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 개별 모델들
model1 = DecisionTreeClassifier(max_depth=10, random_state=42)
model2 = LogisticRegression(random_state=42)
model3 = SVC(kernel='rbf', random_state=42)

# Hard Voting Ensemble
voting_hard = VotingClassifier(
    estimators=[
        ('dt', model1),
        ('lr', model2),
        ('svc', model3)
    ],
    voting='hard'  # 다수결
)

voting_hard.fit(X_train, y_train)

# 개별 모델 vs 앙상블 성능 비교
print("개별 모델 성능:")
model1.fit(X_train, y_train)
print(f"  Decision Tree: {model1.score(X_test, y_test):.4f}")

model2.fit(X_train, y_train)
print(f"  Logistic Regression: {model2.score(X_test, y_test):.4f}")

model3.fit(X_train, y_train)
print(f"  SVM: {model3.score(X_test, y_test):.4f}")

print(f"\nHard Voting Ensemble: {voting_hard.score(X_test, y_test):.4f}")

# 실전 팁
"""
Hard Voting:

장점:
✅ 단순하고 직관적
✅ 다양한 모델 조합 가능
✅ 과적합 방지

단점:
❌ 확률 정보 미사용
❌ 모든 모델의 영향력 동일

사용 시나리오:
- 빠른 프로토타이핑
- 이진 분류
- 모델 성능이 비슷할 때
"""
```

### 1.2 Soft Voting (가중 평균)

```python
# 의도: 확률 기반 투표
# 아이디어: 각 모델의 신뢰도 반영

# Soft Voting
voting_soft = VotingClassifier(
    estimators=[
        ('dt', model1),
        ('lr', model2),
        ('svc', SVC(kernel='rbf', probability=True, random_state=42))
    ],
    voting='soft'  # 확률 평균
)

voting_soft.fit(X_train, y_train)
print(f"Soft Voting: {voting_soft.score(X_test, y_test):.4f}")

# 실전 팁
"""
Soft Voting:

장점:
✅ 확률 정보 활용
✅ Hard Voting보다 성능 좋음

단점:
❌ predict_proba 지원 필요

사용: 성능 중요할 때
"""
```

---

## 2. Bagging

### 2.1 기본 개념

```python
# 의도: Bootstrap + Aggregating
# 아이디어: 데이터 다양성으로 분산 감소

from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    max_samples=1.0,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

bagging.fit(X_train, y_train)

single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)

print(f"Single Tree: {single_tree.score(X_test, y_test):.4f}")
print(f"Bagging (100): {bagging.score(X_test, y_test):.4f}")

# 실전 팁
"""
Bagging:

장점:
✅ Variance 감소
✅ 안정적

단점:
❌ Bias 개선 안 됨

최적: 고분산 모델 (Decision Tree)
"""
```

---

## 3. Boosting

### 3.1 AdaBoost

```python
# 의도: 이전 실수에 집중
# 아이디어: 오분류 샘플에 가중치 증가

from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

adaboost.fit(X_train, y_train)
print(f"AdaBoost: {adaboost.score(X_test, y_test):.4f}")

# 실전 팁
"""
AdaBoost:

장점:
✅ Weak learner로 강력한 모델
✅ Overfitting 저항성

단점:
❌ Noise/Outlier에 민감
❌ XGBoost에 밀림
"""
```

---

## 4. Stacking

### 4.1 기본 Stacking

```python
# 의도: 메타 모델로 앙상블
# 아이디어: Base 예측을 Feature로

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
import xgboost as xgb

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
]

meta_model = LogisticRegression(random_state=42)

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

stacking.fit(X_train, y_train)

print("개별 모델:")
for name, model in base_models:
    model.fit(X_train, y_train)
    print(f"  {name}: {model.score(X_test, y_test):.4f}")

print(f"\nStacking: {stacking.score(X_test, y_test):.4f}")

# 실전 팁
"""
Stacking:

Base Models:
✅ 다양한 알고리즘
✅ 상관관계 낮게
✅ 각각 >0.7 성능

주의:
❌ Overfitting (CV 필수)
❌ 학습 시간 오래
❌ Kaggle 필수!
"""
```

---

## 5. 실전 Ensemble 전략

### 5.1 Kaggle 우승 전략

```python
class KaggleEnsemble:
    """Kaggle 앙상블 파이프라인"""

    def __init__(self):
        self.base_models = []
        self.meta_model = None
        self.oof_predictions = None

    def add_base_model(self, name, model):
        """Base 모델 추가"""
        self.base_models.append((name, model))

    def fit_cv(self, X, y, n_folds=5):
        """K-Fold CV로 OOF 생성"""
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        n_samples = len(X)
        n_models = len(self.base_models)

        self.oof_predictions = np.zeros((n_samples, n_models))

        for i, (name, model) in enumerate(self.base_models):
            print(f"Training {name}...")

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]

                model.fit(X_train_fold, y_train_fold)

                oof_pred = model.predict_proba(X_val_fold)[:, 1]
                self.oof_predictions[val_idx, i] = oof_pred

            model.fit(X, y)

        return self

    def fit_meta(self, y):
        """Meta 모델 학습"""
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(self.oof_predictions, y)

        oof_score = self.meta_model.score(self.oof_predictions, y)
        print(f"OOF Score: {oof_score:.4f}")

        return self

    def predict(self, X):
        """최종 예측"""
        base_preds = np.zeros((len(X), len(self.base_models)))

        for i, (name, model) in enumerate(self.base_models):
            base_preds[:, i] = model.predict_proba(X)[:, 1]

        return self.meta_model.predict(base_preds)

# 사용
ensemble = KaggleEnsemble()
ensemble.add_base_model('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42))
ensemble.add_base_model('rf', RandomForestClassifier(n_estimators=100, random_state=42))

ensemble.fit_cv(X_train, y_train, n_folds=5)
ensemble.fit_meta(y_train)

y_pred = ensemble.predict(X_test)
print(f"Test Score: {np.mean(y_pred == y_test):.4f}")
```

---

## 핵심 요약

### Ensemble 방법 비교

```python
"""
┌─────────────┬──────────┬──────────┬──────────┬────────────┐
│   Method    │  성능    │  속도    │  난이도  │  사용 빈도  │
├─────────────┼──────────┼──────────┼──────────┼────────────┤
│ Voting      │ ★★★☆☆    │ ★★★★☆    │ ★★★★★    │  20%       │
│ Bagging     │ ★★★★☆    │ ★★★☆☆    │ ★★★★☆    │  30%       │
│ Boosting    │ ★★★★★    │ ★★☆☆☆    │ ★★★☆☆    │  40%       │
│ Stacking    │ ★★★★★    │ ★☆☆☆☆    │ ★★☆☆☆    │  10%       │
└─────────────┴──────────┴──────────┴──────────┴────────────┘

실무 선택:
- 프로토타입: Voting
- 프로덕션: Boosting (XGBoost, LightGBM)
- Kaggle: Stacking
- 소규모: Bagging

성능 향상 기대:
- Voting: +2-5%
- Bagging: +5-10%
- Boosting: +10-20%
- Stacking: +3-7%
"""
```

Phase 0.5: Classical ML 완성!
