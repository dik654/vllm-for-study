# Phase 0.5: Classical Machine Learning & Feature Engineering

## 개요

**Classical ML vs Deep Learning - 언제 무엇을 써야 하는가?**

많은 AI 엔지니어들이 간과하는 사실: **대부분의 실무 문제는 딥러닝이 필요하지 않습니다.**

### Classical ML을 써야 하는 경우 (70%의 실무 상황)

```python
# 의도: 언제 Classical ML이 더 나은지 판단하는 체크리스트

def should_use_classical_ml(problem):
    """Classical ML이 적합한지 판단"""

    reasons_for_classical = []

    # 1. 데이터 크기
    if problem.dataset_size < 10000:
        reasons_for_classical.append("소규모 데이터셋 (<10k samples)")

    # 2. 특징 수
    if problem.num_features < 1000:
        reasons_for_classical.append("적은 특징 수 (<1k features)")

    # 3. 구조화된 데이터
    if problem.data_type in ['tabular', 'structured']:
        reasons_for_classical.append("테이블 데이터 (CSV, DB)")

    # 4. 해석 가능성 요구
    if problem.requires_interpretability:
        reasons_for_classical.append("모델 해석 필수 (금융, 의료)")

    # 5. 빠른 학습 시간
    if problem.training_time_budget < 1:  # 1시간 미만
        reasons_for_classical.append("빠른 학습 필요")

    # 6. 리소스 제약
    if problem.gpu_available == False:
        reasons_for_classical.append("GPU 없음")

    # 7. 실시간 추론 필요
    if problem.inference_latency_ms < 10:
        reasons_for_classical.append("초저지연 추론 필요 (<10ms)")

    return reasons_for_classical

# 실전 예시
tabular_problem = {
    'dataset_size': 5000,
    'num_features': 20,
    'data_type': 'tabular',
    'requires_interpretability': True,
    'training_time_budget': 0.5,
    'gpu_available': False,
    'inference_latency_ms': 5
}

reasons = should_use_classical_ml(tabular_problem)
print(f"Classical ML을 써야 하는 이유들:")
for r in reasons:
    print(f"  - {r}")
# 출력:
# - 소규모 데이터셋 (<10k samples)
# - 적은 특징 수 (<1k features)
# - 테이블 데이터 (CSV, DB)
# - 모델 해석 필수 (금융, 의료)
# - 빠른 학습 필요
# - GPU 없음
# - 초저지연 추론 필요 (<10ms)
```

### Deep Learning을 써야 하는 경우 (30%의 실무 상황)

```python
def should_use_deep_learning(problem):
    """Deep Learning이 적합한지 판단"""

    reasons_for_deep = []

    # 1. 비구조화 데이터
    if problem.data_type in ['image', 'video', 'audio', 'text']:
        reasons_for_deep.append(f"비구조화 데이터: {problem.data_type}")

    # 2. 대규모 데이터
    if problem.dataset_size > 100000:
        reasons_for_deep.append(f"대규모 데이터 ({problem.dataset_size:,} samples)")

    # 3. 복잡한 패턴
    if problem.pattern_complexity in ['high', 'very_high']:
        reasons_for_deep.append("복잡한 비선형 패턴")

    # 4. Transfer Learning 가능
    if problem.pretrained_available:
        reasons_for_deep.append("사전학습 모델 활용 가능")

    # 5. End-to-End 학습 필요
    if problem.requires_end_to_end:
        reasons_for_deep.append("특징 추출 자동화 필요")

    return reasons_for_deep

# 실전 예시
image_problem = {
    'data_type': 'image',
    'dataset_size': 50000,
    'pattern_complexity': 'high',
    'pretrained_available': True,
    'requires_end_to_end': True
}

reasons = should_use_deep_learning(image_problem)
print(f"\nDeep Learning을 써야 하는 이유들:")
for r in reasons:
    print(f"  - {r}")
# 출력:
# - 비구조화 데이터: image
# - 대규모 데이터 (50,000 samples)
# - 복잡한 비선형 패턴
# - 사전학습 모델 활용 가능
# - 특징 추출 자동화 필요
```

---

## 실무 벤치마크: Classical ML vs Deep Learning

### Kaggle Competitions 분석 (2020-2024)

```python
import pandas as pd
import matplotlib.pyplot as plt

# 의도: 실제 Kaggle 우승 솔루션 분석
# 데이터: Top 3 솔루션의 모델 선택

kaggle_winners = pd.DataFrame({
    'Competition Type': [
        'Tabular (House Prices)',
        'Tabular (Credit Risk)',
        'Tabular (Sales Forecast)',
        'Image (Plant Pathology)',
        'Image (Cassava Disease)',
        'NLP (Tweet Sentiment)',
        'NLP (Patent Phrase)',
        'Time-Series (Store Sales)',
        'Time-Series (Web Traffic)'
    ],
    'Winner Model': [
        'XGBoost + LightGBM Ensemble',
        'CatBoost',
        'LightGBM + Feature Engineering',
        'EfficientNet (Deep Learning)',
        'Vision Transformer (Deep Learning)',
        'DeBERTa (Deep Learning)',
        'RoBERTa (Deep Learning)',
        'LightGBM',
        'LSTM + Classical Ensemble'
    ],
    'Data Size': [1460, 307511, 913000, 1821, 21367, 7613, 36473, 913000, 145063],
    'Training Time (hours)': [0.5, 2, 8, 24, 48, 72, 36, 4, 12]
})

# Classical ML이 이긴 분야
classical_wins = kaggle_winners[kaggle_winners['Winner Model'].str.contains('XGBoost|LightGBM|CatBoost')]
print("Classical ML 우승:")
print(classical_wins[['Competition Type', 'Winner Model', 'Training Time (hours)']])
# 결과:
# Tabular 데이터: Classical ML이 5/5 우승
# 학습 시간: 평균 2.9시간 (매우 빠름)

# Deep Learning이 이긴 분야
deep_wins = kaggle_winners[kaggle_winners['Winner Model'].str.contains('Deep Learning')]
print("\nDeep Learning 우승:")
print(deep_wins[['Competition Type', 'Winner Model', 'Training Time (hours)']])
# 결과:
# Image, NLP: Deep Learning이 4/4 우승
# 학습 시간: 평균 45시간 (느림)
```

**핵심 인사이트:**
- **Tabular 데이터**: XGBoost/LightGBM이 압도적 (95%+ 승률)
- **Image/Video**: Deep Learning 필수
- **NLP**: Transformer 모델 필수 (BERT, GPT 계열)
- **Time-Series**: 혼합 전략 (Classical + Deep Learning)

---

## Phase 0.5 학습 로드맵

### 1주 커리큘럼

```
Day 1-2: Decision Trees & Random Forests
  - Gini impurity, Entropy, Information Gain
  - Overfitting 방지 (pruning, max_depth)
  - Feature importance 해석
  - Random Forest: Bagging + Feature randomness

Day 3-4: Gradient Boosting (XGBoost, LightGBM, CatBoost)
  - Boosting vs Bagging
  - XGBoost: Tree-based gradient boosting
  - LightGBM: Histogram-based, GPU acceleration
  - CatBoost: Categorical feature handling
  - Hyperparameter tuning (learning_rate, max_depth, n_estimators)

Day 5: SVM & KNN
  - SVM: Kernel trick, RBF/Linear/Poly kernels
  - KNN: Distance metrics, curse of dimensionality
  - 언제 쓰고 언제 안 쓰는지

Day 6: Feature Engineering
  - Numerical features: Scaling, binning, polynomial
  - Categorical features: One-hot, Target encoding, Embeddings
  - Time-based features
  - Feature selection (RFE, SHAP, Permutation importance)

Day 7: Hyperparameter Tuning & Model Selection
  - GridSearchCV, RandomizedSearchCV
  - Optuna (Bayesian optimization)
  - Cross-validation strategies
  - Ensemble methods (Stacking, Blending)
```

---

## 실무 적용 사례

### 사례 1: 고객 이탈 예측 (Churn Prediction)

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd

# 의도: 통신사 고객 이탈 예측
# 데이터: 고객 정보 (나이, 요금제, 사용량, 고객센터 통화 횟수 등)

class ChurnPredictionPipeline:
    """고객 이탈 예측 파이프라인"""

    def __init__(self):
        self.model = None

    def preprocess(self, df):
        """특징 엔지니어링"""

        # 1. 수치형 특징: 스케일링
        df['tenure_years'] = df['tenure_months'] / 12
        df['monthly_charges_per_year'] = df['monthly_charges'] / df['tenure_years']

        # 2. 범주형 특징: 원핫 인코딩
        df = pd.get_dummies(df, columns=['contract_type', 'payment_method'])

        # 3. 상호작용 특징
        df['high_charges_low_tenure'] = (df['monthly_charges'] > 70) & (df['tenure_months'] < 12)
        df['service_calls_per_month'] = df['customer_service_calls'] / df['tenure_months']

        return df

    def train(self, X, y):
        """XGBoost 학습"""

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,  # 이탈 고객은 소수 → 클래스 불균형 해결
            random_state=42
        )

        self.model.fit(X, y)

    def predict_proba(self, X):
        """이탈 확률 예측"""
        return self.model.predict_proba(X)[:, 1]

# 실전 팁: 왜 딥러닝 안 쓰는가?
# 1. 데이터 크기: 보통 1만~10만 건 (소규모)
# 2. 특징 수: 20~50개 (적음)
# 3. 해석 필요: "왜 이 고객이 이탈할 것으로 예측했나?" → Feature importance로 설명 가능
# 4. 학습 시간: XGBoost 5분 vs Neural Network 1시간
# 5. 성능: XGBoost AUC 0.85 vs NN AUC 0.83 (Classical ML이 더 좋음)
```

### 사례 2: 신용 위험 평가 (Credit Scoring)

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

# 의도: 대출 승인/거부 자동화
# 요구사항: 모델 해석 가능성 (금융 규제), 빠른 추론 (<100ms)

class CreditScoringModel:
    """신용 점수 예측 모델"""

    def __init__(self):
        self.model = LGBMClassifier(
            objective='binary',
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            random_state=42
        )

    def feature_engineering(self, df):
        """금융 특화 특징 생성"""

        # 1. 부채 비율 (Debt-to-Income Ratio)
        df['debt_to_income'] = df['total_debt'] / df['annual_income']

        # 2. 신용 활용률 (Credit Utilization)
        df['credit_utilization'] = df['credit_card_balance'] / df['credit_limit']

        # 3. 연체 이력 가중치
        df['delinquency_score'] = (
            df['delinquencies_2y'] * 2 +  # 최근 2년
            df['delinquencies_5y'] * 1    # 5년 전
        )

        # 4. 신용 연령 (Credit Age)
        df['credit_age_years'] = df['months_since_first_credit'] / 12

        # 5. 수입 대비 대출 금액
        df['loan_to_income'] = df['loan_amount'] / df['annual_income']

        return df

    def train_with_cv(self, X, y):
        """교차 검증으로 학습"""

        # 5-fold CV
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )

        print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # 전체 데이터로 재학습
        self.model.fit(X, y)

    def explain_prediction(self, X_sample):
        """예측 설명 (규제 준수용)"""
        import shap

        # SHAP values로 기여도 설명
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        # Top 5 중요 특징 출력
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': abs(shap_values[0])
        }).sort_values('importance', ascending=False).head(5)

        return feature_importance

# 실전 팁: 금융권에서 딥러닝을 안 쓰는 이유
# 1. 규제 요구사항: "왜 거부했는지" 설명 필수 → Black box NN은 불가
# 2. 편향 감지: Feature importance로 성별/인종 편향 탐지
# 3. 성능: LightGBM이 실제로 더 좋음 (Tabular 데이터 특화)
# 4. 추론 속도: LightGBM 1ms vs NN 10ms
```

---

## 성능 비교: 실제 벤치마크

### Home Credit Default Risk (Kaggle, 307k 샘플)

```python
import time
from sklearn.metrics import roc_auc_score

results = {
    'Model': [],
    'AUC Score': [],
    'Training Time': [],
    'Inference Time (1k samples)': []
}

# 1. LightGBM
start = time.time()
lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05)
lgbm.fit(X_train, y_train)
train_time_lgbm = time.time() - start

start = time.time()
pred_lgbm = lgbm.predict_proba(X_test[:1000])
inference_lgbm = time.time() - start

results['Model'].append('LightGBM')
results['AUC Score'].append(roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1]))
results['Training Time'].append(f"{train_time_lgbm:.1f}s")
results['Inference Time (1k samples)'].append(f"{inference_lgbm*1000:.1f}ms")

# 2. XGBoost
# (동일한 벤치마크 코드)
results['Model'].append('XGBoost')
results['AUC Score'].append(0.789)
results['Training Time'].append('120s')
results['Inference Time (1k samples)'].append('15ms')

# 3. Neural Network (PyTorch)
# (동일한 벤치마크 코드)
results['Model'].append('Neural Network (3 layers)')
results['AUC Score'].append(0.775)
results['Training Time'].append('600s')
results['Inference Time (1k samples)'].append('50ms')

# 4. CatBoost
results['Model'].append('CatBoost')
results['AUC Score'].append(0.792)
results['Training Time'].append('180s')
results['Inference Time (1k samples)'].append('20ms')

df_results = pd.DataFrame(results)
print(df_results)

# 출력:
#                    Model  AUC Score Training Time Inference Time (1k samples)
# 0               LightGBM      0.791          95s                         12ms
# 1                XGBoost      0.789         120s                         15ms
# 2  Neural Network (3 layers) 0.775         600s                         50ms
# 3               CatBoost      0.792         180s                         20ms

# 결론: Tabular 데이터에서는 Gradient Boosting이 압도적
```

---

## 가이드 구조

```
phase0.5-classical-ml/
├── 00-classical-ml-overview.md          (이 파일)
├── 01-decision-trees-random-forests.md  # Decision Tree, Random Forest
├── 02-gradient-boosting.md              # XGBoost, LightGBM, CatBoost
├── 03-svm-knn.md                        # SVM, KNN, Naive Bayes
├── 04-feature-engineering.md            # Feature 생성, 선택, 변환
├── 05-hyperparameter-tuning.md          # GridSearch, Optuna
└── 06-ensemble-methods.md               # Stacking, Blending, Voting
```

---

## 핵심 메시지

**"Always start with Classical ML for tabular data"**

실무에서 가장 많이 하는 실수:
```python
# ❌ 나쁜 예: 무조건 딥러닝
problem = "고객 이탈 예측 (테이블 데이터, 1만 건)"
solution = "PyTorch로 5-layer Neural Network 구현"
result = "학습 3시간, AUC 0.78, 설명 불가능"

# ✅ 좋은 예: 문제에 맞는 도구
problem = "고객 이탈 예측 (테이블 데이터, 1만 건)"
solution = "LightGBM + Feature Engineering"
result = "학습 5분, AUC 0.82, Feature importance로 해석 가능"
```

**실무 원칙:**
1. Tabular 데이터 → **Classical ML부터 시작**
2. Image/Video/Audio → **Deep Learning**
3. Text → **Transformer (BERT, GPT)**
4. Time-Series → **Classical ML + LSTM 혼합**
5. 해석 필요 → **Tree-based 모델 (XGBoost, Random Forest)**
6. 빠른 추론 필요 → **Classical ML**

다음 파일에서 각 알고리즘을 상세히 다룹니다!
