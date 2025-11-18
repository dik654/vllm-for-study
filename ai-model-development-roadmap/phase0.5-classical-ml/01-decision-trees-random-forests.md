# Decision Trees & Random Forests

## 1. Decision Tree 기본 개념

### 1.1 의사결정 나무란?

```python
# 의도: 의사결정 과정을 나무 구조로 표현
# 아이디어: if-else 규칙을 자동으로 학습

"""
예시: 타이타닉 생존 예측

                    [성별]
                   /      \
              남성/        \여성
                /            \
           [나이]             [객실등급]
          /    \             /        \
    <10세/      \>=10세  1등급/        \3등급
       /          \         /            \
  생존(80%)   사망(90%)  생존(95%)    생존(50%)

규칙 추출:
- IF 성별=남성 AND 나이<10 THEN 생존 확률 80%
- IF 성별=여성 AND 객실등급=1 THEN 생존 확률 95%
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# 타이타닉 데이터 (간소화)
X = np.array([
    # [성별(0=여, 1=남), 나이, 객실등급]
    [1, 25, 3],  # 남성, 25세, 3등급 → 사망
    [0, 30, 1],  # 여성, 30세, 1등급 → 생존
    [1, 5, 2],   # 남성, 5세, 2등급 → 생존 (어린이)
    [0, 45, 3],  # 여성, 45세, 3등급 → 생존
    # ... 더 많은 데이터
])

y = np.array([0, 1, 1, 1])  # 0=사망, 1=생존

# Decision Tree 학습
clf = DecisionTreeClassifier(
    max_depth=3,           # 최대 깊이 (overfitting 방지)
    min_samples_split=2,   # 분할 최소 샘플 수
    min_samples_leaf=1,    # 리프 노드 최소 샘플 수
    random_state=42
)

clf.fit(X, y)

# 트리 시각화
plt.figure(figsize=(15, 10))
tree.plot_tree(
    clf,
    feature_names=['성별', '나이', '객실등급'],
    class_names=['사망', '생존'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()

# 예측
new_passenger = [[1, 8, 3]]  # 남성, 8세, 3등급
prediction = clf.predict(new_passenger)
probability = clf.predict_proba(new_passenger)

print(f"예측: {'생존' if prediction[0] == 1 else '사망'}")
print(f"생존 확률: {probability[0][1]:.2%}")
```

---

## 2. Split 기준: Gini Impurity vs Entropy

### 2.1 Gini Impurity (지니 불순도)

```python
# 의도: 노드의 "불순도" 측정
# 아이디어: 0에 가까울수록 순수 (한 클래스로 구성)

def gini_impurity(labels):
    """
    Gini Impurity = 1 - Σ(p_i^2)

    예시:
    - [0, 0, 0, 0]: Gini = 0 (완전 순수)
    - [0, 0, 1, 1]: Gini = 0.5 (최대 불순)
    - [0, 0, 0, 1]: Gini = 0.375
    """
    if len(labels) == 0:
        return 0

    # 각 클래스 비율 계산
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    # Gini = 1 - Σ(p_i^2)
    gini = 1 - np.sum(probabilities ** 2)

    return gini

# 예시 1: 완전 순수
labels_pure = [0, 0, 0, 0]
print(f"순수 노드 Gini: {gini_impurity(labels_pure)}")  # 0.0

# 예시 2: 최대 불순 (50:50)
labels_impure = [0, 0, 1, 1]
print(f"불순 노드 Gini: {gini_impurity(labels_impure)}")  # 0.5

# 예시 3: 중간
labels_mid = [0, 0, 0, 1]
print(f"중간 노드 Gini: {gini_impurity(labels_mid)}")  # 0.375

# 실전 팁: Gini가 0에 가까울수록 좋은 분할!
```

### 2.2 Information Gain (정보 이득)

```python
# 의도: Entropy 기반 분할 기준
# 아이디어: 분할 전후 불확실성 감소량

def entropy(labels):
    """
    Entropy = -Σ(p_i * log2(p_i))

    - 0: 완전 순수
    - 1: 최대 불확실 (이진 분류)
    """
    if len(labels) == 0:
        return 0

    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    # Entropy = -Σ(p_i * log2(p_i))
    # 0 * log(0) = 0으로 처리
    entropy_val = -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    return entropy_val

def information_gain(parent, left_child, right_child):
    """
    Information Gain = Entropy(parent) - Weighted Average Entropy(children)
    """
    n = len(parent)
    n_left = len(left_child)
    n_right = len(right_child)

    # 부모 엔트로피
    parent_entropy = entropy(parent)

    # 자식 엔트로피 가중 평균
    weighted_child_entropy = (
        (n_left / n) * entropy(left_child) +
        (n_right / n) * entropy(right_child)
    )

    # 정보 이득
    ig = parent_entropy - weighted_child_entropy

    return ig

# 예시: 나이로 분할
parent = [0, 0, 0, 1, 1, 1, 1, 1]  # 3명 사망, 5명 생존
left = [0, 0, 0, 1]     # 나이 < 30: 3명 사망, 1명 생존
right = [1, 1, 1, 1]    # 나이 >= 30: 0명 사망, 4명 생존

ig = information_gain(parent, left, right)
print(f"나이로 분할 시 Information Gain: {ig:.4f}")
# → 높을수록 좋은 분할!

# Gini vs Entropy 비교
print(f"\n부모 노드:")
print(f"  Gini: {gini_impurity(parent):.4f}")
print(f"  Entropy: {entropy(parent):.4f}")

# 실전 팁: Gini가 더 빠름 (제곱만 계산), Entropy는 log 계산 필요
# 성능 차이는 미미 → sklearn 기본값은 Gini
```

---

## 3. 트리 분할 알고리즘 (From Scratch)

### 3.1 Best Split 찾기

```python
class DecisionTreeFromScratch:
    """의사결정 나무 from scratch 구현"""

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def gini_impurity(self, y):
        """Gini 불순도 계산"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def find_best_split(self, X, y):
        """
        모든 feature, 모든 threshold에 대해
        Gini Impurity 감소량이 가장 큰 split 찾기
        """
        best_gini_gain = -1
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape
        parent_gini = self.gini_impurity(y)

        # 각 feature에 대해
        for feature_idx in range(n_features):
            # 해당 feature의 모든 unique 값을 threshold 후보로
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                # 분할
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue  # 한쪽이 비면 skip

                # 자식 노드 Gini
                left_gini = self.gini_impurity(y[left_mask])
                right_gini = self.gini_impurity(y[right_mask])

                # 가중 평균 Gini
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_gini = (n_left / n_samples) * left_gini + (n_right / n_samples) * right_gini

                # Gini Gain
                gini_gain = parent_gini - weighted_gini

                # 최대 gain 업데이트
                if gini_gain > best_gini_gain:
                    best_gini_gain = gini_gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini_gain

    def build_tree(self, X, y, depth=0):
        """재귀적으로 트리 구축"""

        n_samples = len(y)
        n_classes = len(np.unique(y))

        # 종료 조건
        if (depth >= self.max_depth or
            n_classes == 1 or
            n_samples < self.min_samples_split):
            # 리프 노드: 다수결로 클래스 결정
            leaf_value = np.bincount(y).argmax()
            return {'leaf': True, 'value': leaf_value}

        # Best split 찾기
        best_feature, best_threshold, best_gain = self.find_best_split(X, y)

        if best_feature is None:
            # Split 불가능 → 리프 노드
            leaf_value = np.bincount(y).argmax()
            return {'leaf': True, 'value': leaf_value}

        # 분할
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # 재귀적으로 자식 노드 생성
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        """트리 학습"""
        self.tree = self.build_tree(X, y)
        return self

    def predict_sample(self, x, tree):
        """단일 샘플 예측"""
        if tree['leaf']:
            return tree['value']

        if x[tree['feature']] <= tree['threshold']:
            return self.predict_sample(x, tree['left'])
        else:
            return self.predict_sample(x, tree['right'])

    def predict(self, X):
        """전체 데이터 예측"""
        return np.array([self.predict_sample(x, self.tree) for x in X])

# 테스트
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# From Scratch 모델
custom_tree = DecisionTreeFromScratch(max_depth=3)
custom_tree.fit(X_train, y_train)
y_pred_custom = custom_tree.predict(X_test)
accuracy_custom = np.mean(y_pred_custom == y_test)

# Sklearn 모델
sklearn_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
sklearn_tree.fit(X_train, y_train)
y_pred_sklearn = sklearn_tree.predict(X_test)
accuracy_sklearn = np.mean(y_pred_sklearn == y_test)

print(f"Custom Tree 정확도: {accuracy_custom:.4f}")
print(f"Sklearn Tree 정확도: {accuracy_sklearn:.4f}")
# 거의 동일!
```

---

## 4. Overfitting 방지 기법

### 4.1 Pruning (가지치기)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# 의도: 과적합 방지 - 트리 복잡도 제한
# 아이디어: max_depth, min_samples 등으로 제어

# Overfitting 예시
overfit_tree = DecisionTreeClassifier(max_depth=None)  # 제한 없음
overfit_tree.fit(X_train, y_train)

train_score = overfit_tree.score(X_train, y_train)
test_score = overfit_tree.score(X_test, y_test)

print("Overfitting 트리:")
print(f"  Train 정확도: {train_score:.4f}")  # 1.0 (완벽)
print(f"  Test 정확도: {test_score:.4f}")    # 0.85 (낮음)
print(f"  트리 깊이: {overfit_tree.get_depth()}")  # 매우 깊음

# Pruning 적용
pruned_tree = DecisionTreeClassifier(
    max_depth=5,             # 최대 깊이 제한
    min_samples_split=10,    # 분할 최소 샘플: 10개 미만이면 분할 안 함
    min_samples_leaf=5,      # 리프 노드 최소 샘플: 5개 이상
    max_leaf_nodes=20,       # 최대 리프 노드 수
    min_impurity_decrease=0.01  # Gini 감소량이 0.01 이상이어야 분할
)
pruned_tree.fit(X_train, y_train)

train_score_pruned = pruned_tree.score(X_train, y_train)
test_score_pruned = pruned_tree.score(X_test, y_test)

print("\nPruning 적용 트리:")
print(f"  Train 정확도: {train_score_pruned:.4f}")  # 0.95 (약간 낮음)
print(f"  Test 정확도: {test_score_pruned:.4f}")    # 0.92 (높음!)
print(f"  트리 깊이: {pruned_tree.get_depth()}")    # 5

# 실전 팁: GridSearch로 최적 하이퍼파라미터 찾기
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\n최적 하이퍼파라미터: {grid_search.best_params_}")
print(f"최고 CV 정확도: {grid_search.best_score_:.4f}")
```

---

## 5. Random Forest: Bagging + Random Feature Selection

### 5.1 Random Forest 핵심 아이디어

```python
from sklearn.ensemble import RandomForestClassifier

"""
Random Forest = Bagging + Random Feature Selection

아이디어:
1. Bagging: Bootstrap Aggregating
   - 원본 데이터에서 복원 추출로 N개 데이터셋 생성
   - 각 데이터셋으로 N개 트리 학습
   - 예측: 다수결 (분류) 또는 평균 (회귀)

2. Random Feature Selection:
   - 각 split마다 전체 feature 중 √d개만 랜덤 선택
   - 다양성 증가 → 과적합 방지

예시:
원본 데이터: 1000 samples, 50 features

Tree 1: Bootstrap 샘플 1000개 (중복 허용)
        각 split마다 √50 ≈ 7개 feature만 고려
Tree 2: 다른 Bootstrap 샘플 1000개
        각 split마다 다른 7개 feature
...
Tree 100: ...

최종 예측: 100개 트리의 다수결
"""

# Random Forest 학습
rf = RandomForestClassifier(
    n_estimators=100,        # 트리 개수
    max_depth=10,            # 각 트리 최대 깊이
    max_features='sqrt',     # 각 split마다 √d개 feature 사용
    min_samples_split=5,
    min_samples_leaf=2,
    bootstrap=True,          # Bootstrap 샘플링
    n_jobs=-1,               # 병렬 처리
    random_state=42
)

rf.fit(X_train, y_train)

# 예측
y_pred = rf.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print(f"Random Forest 정확도: {accuracy:.4f}")

# 단일 Decision Tree와 비교
single_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
single_tree.fit(X_train, y_train)
single_tree_accuracy = single_tree.score(X_test, y_test)

print(f"단일 Tree 정확도: {single_tree_accuracy:.4f}")
print(f"성능 향상: {(accuracy - single_tree_accuracy)*100:.2f}%p")

# 일반적으로 Random Forest가 5-10%p 더 높음
```

### 5.2 Bootstrap Aggregating (Bagging) 원리

```python
# 의도: Bagging이 왜 성능을 높이는지 이해
# 아이디어: Variance 감소 (Bias-Variance Tradeoff)

import matplotlib.pyplot as plt

def bootstrap_sample(X, y, random_state=None):
    """Bootstrap 샘플링: 복원 추출"""
    if random_state:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.random.choice(n_samples, size=n_samples, replace=True)

    return X[indices], y[indices]

# 100개 트리 학습 (각각 다른 Bootstrap 샘플)
n_trees = 100
trees = []

for i in range(n_trees):
    X_boot, y_boot = bootstrap_sample(X_train, y_train, random_state=i)

    tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    tree.fit(X_boot, y_boot)
    trees.append(tree)

# 각 트리의 예측
predictions = np.array([tree.predict(X_test) for tree in trees])

# 다수결 투표
final_prediction = np.apply_along_axis(
    lambda x: np.bincount(x).argmax(),
    axis=0,
    arr=predictions
)

# 개별 트리 정확도 vs Ensemble 정확도
individual_accuracies = [np.mean(pred == y_test) for pred in predictions]
ensemble_accuracy = np.mean(final_prediction == y_test)

print(f"개별 트리 평균 정확도: {np.mean(individual_accuracies):.4f}")
print(f"Ensemble 정확도: {ensemble_accuracy:.4f}")
print(f"성능 향상: {(ensemble_accuracy - np.mean(individual_accuracies))*100:.2f}%p")

# 시각화: 트리 개수에 따른 정확도 변화
accuracies_by_n_trees = []
for n in range(1, 101):
    # n개 트리의 앙상블
    ensemble_pred = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=predictions[:n]
    )
    acc = np.mean(ensemble_pred == y_test)
    accuracies_by_n_trees.append(acc)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), accuracies_by_n_trees)
plt.axhline(y=np.mean(individual_accuracies), color='r', linestyle='--', label='개별 트리 평균')
plt.xlabel('트리 개수')
plt.ylabel('정확도')
plt.title('Random Forest: 트리 개수에 따른 성능')
plt.legend()
plt.grid(True)
plt.show()

# 실전 팁: 트리 50-100개면 충분 (그 이상은 성능 향상 미미)
```

---

## 6. Feature Importance 해석

### 6.1 Gini Importance

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Random Forest 학습
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature Importance 추출
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# 실전 예시: 고객 이탈 예측
"""
Feature Importance:
                  feature  importance
0    customer_service_calls      0.35  ← 가장 중요!
1               tenure_months      0.25
2            monthly_charges      0.18
3               contract_type      0.12
4              payment_method      0.10

인사이트:
- 고객센터 통화 횟수가 이탈에 가장 큰 영향
- → 고객센터 서비스 개선 필요
"""
```

### 6.2 Permutation Importance (더 정확한 방법)

```python
from sklearn.inspection import permutation_importance

# 의도: Feature를 랜덤하게 섞었을 때 성능 하락 측정
# 아이디어: 중요한 feature일수록 섞으면 성능이 크게 떨어짐

# Permutation Importance 계산
perm_importance = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# 결과
perm_importance_df = pd.DataFrame({
    'feature': iris.feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("Permutation Importance:")
print(perm_importance_df)

# Gini vs Permutation 비교
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Gini Importance
axes[0].barh(feature_importance['feature'], feature_importance['importance'])
axes[0].set_xlabel('Gini Importance')
axes[0].set_title('Gini Importance')
axes[0].invert_yaxis()

# Permutation Importance
axes[1].barh(perm_importance_df['feature'], perm_importance_df['importance_mean'])
axes[1].set_xlabel('Permutation Importance')
axes[1].set_title('Permutation Importance (더 정확)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# 실전 팁: Permutation Importance가 더 신뢰성 높음
```

---

## 7. 실전 활용: Kaggle 우승 전략

### 7.1 Random Forest Best Practices

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 의도: 실전에서 성능 최대화
# 아이디어: Hyperparameter tuning + Feature engineering

class RandomForestPipeline:
    """Random Forest 최적화 파이프라인"""

    def __init__(self):
        self.model = None
        self.best_params = None

    def tune_hyperparameters(self, X_train, y_train):
        """RandomizedSearch로 하이퍼파라미터 튜닝"""

        param_distributions = {
            'n_estimators': randint(50, 500),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'max_samples': uniform(0.5, 0.5)  # 0.5 ~ 1.0
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        random_search = RandomizedSearchCV(
            rf,
            param_distributions,
            n_iter=100,  # 100가지 조합 시도
            cv=5,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        self.best_params = random_search.best_params_
        self.model = random_search.best_estimator_

        print(f"Best Score: {random_search.best_score_:.4f}")
        print(f"Best Params: {self.best_params}")

        return self

    def get_feature_importance(self, feature_names):
        """Feature Importance 추출"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

# 실전 팁 모음
"""
1. n_estimators (트리 개수):
   - 많을수록 좋지만 학습 시간 증가
   - 실무: 100-500 사이 (성능 vs 속도 trade-off)

2. max_depth (최대 깊이):
   - 너무 깊으면 overfitting
   - 실무: 10-20 정도

3. max_features (split마다 고려할 feature 수):
   - 'sqrt': √d (분류 기본값)
   - 'log2': log2(d)
   - None: 모든 feature (overfitting 위험)

4. min_samples_split, min_samples_leaf:
   - 클수록 regularization 강함
   - 데이터 크기에 따라 조정

5. Bootstrap:
   - True: Bagging 적용 (기본값)
   - False: 전체 데이터 사용 (Pasting)

6. max_samples:
   - Bootstrap=True일 때 샘플 비율
   - 0.8 정도면 학습 속도 20% 향상, 성능 거의 동일
"""
```

### 7.2 실전 케이스: Titanic Kaggle

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Titanic 데이터 로드
train_df = pd.read_csv('titanic_train.csv')

# Feature Engineering
def engineer_features(df):
    """Feature 생성"""

    # 1. 결측치 처리
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # 2. 새 Feature 생성
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # 3. Title 그룹화
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                       'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # 4. 범주형 → 숫자형
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})

    # 5. Feature 선택
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
                'FamilySize', 'IsAlone', 'Title']

    return df[features]

X = engineer_features(train_df)
y = train_df['Survived']

# Random Forest 학습
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 최종 학습
rf.fit(X, y)

# Feature Importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# 예상 출력:
# feature       importance
# Title         0.28
# Sex           0.25
# Fare          0.18
# Age           0.15
# Pclass        0.08
# ...
```

---

## 8. Random Forest vs XGBoost

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time

# 성능 비교
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = []

for name, model in models.items():
    # 학습 시간
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # 추론 시간
    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start

    # 정확도
    accuracy = np.mean(y_pred == y_test)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Train Time': f"{train_time:.2f}s",
        'Inference Time': f"{inference_time*1000:.2f}ms"
    })

results_df = pd.DataFrame(results)
print(results_df)

"""
예상 출력:
          Model  Accuracy Train Time Inference Time
0  Random Forest     0.89      2.5s          15ms
1       XGBoost     0.92      1.8s          10ms

결론:
- XGBoost가 대부분 더 좋음 (특히 Tabular 데이터)
- Random Forest 장점: 해석 쉬움, 하이퍼파라미터 튜닝 덜 민감
- XGBoost 장점: 성능 더 높음, 속도 빠름
"""
```

---

## 핵심 정리

### Decision Tree
- **장점**: 해석 쉬움, 비선형 관계 학습 가능, Feature scaling 불필요
- **단점**: Overfitting 쉬움, 불안정 (데이터 조금만 바뀌어도 트리 구조 변함)
- **용도**: 규칙 기반 시스템, 빠른 프로토타이핑

### Random Forest
- **장점**: Overfitting 방지, 안정적, Feature importance 제공
- **단점**: 해석성 낮아짐, 메모리 많이 사용 (트리 100개 저장)
- **용도**: Tabular 데이터 baseline, Feature selection

### 실무 가이드라인
```python
# ✅ Random Forest를 쓰면 좋은 경우
- Tabular 데이터 (CSV, DB)
- Feature importance가 중요
- Robust한 baseline 필요
- 하이퍼파라미터 튜닝에 시간 쓰기 싫을 때

# ❌ Random Forest를 쓰면 안 되는 경우
- 고차원 sparse data (NLP, 추천시스템)
- 메모리 제약 심함
- 최고 성능 필요 (→ XGBoost, LightGBM)
```

다음: Gradient Boosting (XGBoost, LightGBM) →
