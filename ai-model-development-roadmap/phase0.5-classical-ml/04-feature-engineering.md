# Feature Engineering: 실전 완벽 가이드

## 핵심 원칙

**"좋은 Features = 좋은 모델"**

```python
# 의도: Feature Engineering이 모델 성능에 미치는 영향
# 아이디어: 알고리즘보다 Feature가 더 중요

"""
실무 경험칙:
- Raw 데이터 + 복잡한 모델 = 70% 정확도
- 좋은 Features + 단순한 모델 = 90% 정확도

Feature Engineering의 가치:
1. 성능 향상: 10-30%p 개선
2. 학습 속도: 2-10배 빠름
3. 해석 가능성: 비즈니스 인사이트
"""
```

---

## 1. Numerical Features (수치형 특징)

### 1.1 Scaling & Normalization

```python
# 의도: Feature 스케일 조정
# 아이디어: 알고리즘에 따라 필요 여부 다름

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 샘플 데이터
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 100],  # 이상치 포함
    'income': [30000, 45000, 60000, 80000, 120000, 1000000],  # 큰 범위
    'credit_score': [650, 700, 750, 800, 850, 900]
})

print("원본 데이터 스케일:")
print(df.describe())

# 1. StandardScaler (평균 0, 표준편차 1)
scaler_std = StandardScaler()
df_std = pd.DataFrame(
    scaler_std.fit_transform(df),
    columns=df.columns
)

print("\nStandardScaler:")
print(df_std.describe())

# 2. MinMaxScaler (0-1 범위)
scaler_minmax = MinMaxScaler()
df_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(df),
    columns=df.columns
)

print("\nMinMaxScaler:")
print(df_minmax.describe())

# 3. RobustScaler (중앙값, IQR 사용 - 이상치에 강함)
scaler_robust = RobustScaler()
df_robust = pd.DataFrame(
    scaler_robust.fit_transform(df),
    columns=df.columns
)

print("\nRobustScaler:")
print(df_robust.describe())

# 실전 가이드
"""
알고리즘별 Scaling 필요 여부:

Scaling 필요:
✅ Linear Regression, Logistic Regression
✅ SVM, KNN
✅ Neural Networks
✅ PCA, K-Means

Scaling 불필요:
❌ Tree-based (Random Forest, XGBoost, LightGBM)
❌ Naive Bayes

어떤 Scaler?
- 정규분포: StandardScaler
- 범위 지정 필요: MinMaxScaler
- 이상치 많음: RobustScaler
"""
```

### 1.2 Binning (구간화)

```python
# 의도: 연속형 → 범주형 변환
# 아이디어: 비선형 관계 포착, Overfitting 방지

import pandas as pd

# 나이 데이터
ages = pd.Series([18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])

# 1. Equal Width Binning (동일 간격)
age_bins_equal = pd.cut(ages, bins=5, labels=['10s', '20s', '30s', '40s', '50s+'])
print("Equal Width Binning:")
print(age_bins_equal.value_counts())

# 2. Equal Frequency Binning (동일 빈도)
age_bins_freq = pd.qcut(ages, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
print("\nEqual Frequency Binning:")
print(age_bins_freq.value_counts())

# 3. Custom Binning (도메인 지식 활용)
age_bins_custom = pd.cut(
    ages,
    bins=[0, 20, 30, 50, 100],
    labels=['청소년', '청년', '중년', '노년']
)
print("\nCustom Binning:")
print(age_bins_custom.value_counts())

# 실전 예시: 신용카드 사용 금액
amounts = pd.Series([10, 50, 100, 500, 1000, 5000, 10000, 50000])

# Log-scale binning (로그 스케일)
amounts_log_bins = pd.cut(
    np.log10(amounts + 1),
    bins=4,
    labels=['매우 낮음', '낮음', '중간', '높음']
)

print("\nLog-scale Binning (금액):")
for amount, category in zip(amounts, amounts_log_bins):
    print(f"  {amount:>6}원 → {category}")

# 언제 Binning을 쓸까?
"""
✅ 사용하는 경우:
- 비선형 관계 (나이-구매력)
- Tree 모델에도 도움됨 (Split 효율화)
- 도메인 의미 (청소년/청년/중년)
- Overfitting 방지

❌ 사용하지 않는 경우:
- 선형 관계
- 정보 손실 최소화 필요
- 데이터 충분히 많음
"""
```

### 1.3 Polynomial Features (다항 특징)

```python
# 의도: 비선형 관계 학습
# 아이디어: x, y → x, y, x^2, xy, y^2

from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# 비선형 데이터
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x^2

# 1. Linear Model (원본 Features)
from sklearn.linear_model import LinearRegression

model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred_linear = model_linear.predict(X)

# 2. Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("Polynomial Features:")
print(f"원본: {X[0]} → 변환: {X_poly[0]}")  # [1] → [1, 1^2]

model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred_poly = model_poly.predict(X_poly)

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='실제 데이터')
plt.plot(X, y_pred_linear, label='Linear Model', linestyle='--')
plt.plot(X, y_pred_poly, label='Polynomial Model', linewidth=2)
plt.legend()
plt.title('Polynomial Features 효과')
plt.show()

print(f"\nLinear MSE: {np.mean((y - y_pred_linear)**2):.2f}")
print(f"Polynomial MSE: {np.mean((y - y_pred_poly)**2):.2f}")

# 실전 예시: 2D Features
X_2d = np.array([[1, 2], [3, 4]])
poly_2d = PolynomialFeatures(degree=2, include_bias=False)
X_poly_2d = poly_2d.fit_transform(X_2d)

print("\n2D Polynomial Features:")
print(f"원본 shape: {X_2d.shape}")  # (2, 2)
print(f"변환 shape: {X_poly_2d.shape}")  # (2, 5)
print(f"Features: {poly_2d.get_feature_names_out(['x1', 'x2'])}")
# ['x1', 'x2', 'x1^2', 'x1 x2', 'x2^2']

# 주의사항
"""
Polynomial Features 주의점:
1. Feature 수 폭발: degree 높으면 차원 급증
   - degree=2, n=10 → 55 features
   - degree=3, n=10 → 220 features

2. Overfitting 위험: Regularization 필수
   - Ridge, Lasso 함께 사용

3. 계산 비용: degree=3 이상은 느림

실전 사용:
- degree=2: 일반적
- degree=3: 신중하게
- degree=4+: 거의 안 씀
"""
```

### 1.4 Log Transform (로그 변환)

```python
# 의도: 왜도(Skewness) 감소
# 아이디어: 긴 꼬리 분포 → 정규분포

# 왜도가 큰 데이터 (수입, 집값 등)
incomes = np.array([20000, 25000, 30000, 35000, 40000, 50000,
                    60000, 80000, 100000, 500000, 1000000])

# 로그 변환
incomes_log = np.log1p(incomes)  # log(1 + x) - 0 방지

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(incomes, bins=20, edgecolor='black')
axes[0].set_title(f'원본 (Skewness: {pd.Series(incomes).skew():.2f})')
axes[0].set_xlabel('Income')

axes[1].hist(incomes_log, bins=20, edgecolor='black')
axes[1].set_title(f'Log 변환 (Skewness: {pd.Series(incomes_log).skew():.2f})')
axes[1].set_xlabel('Log(Income)')

plt.tight_layout()
plt.show()

# 언제 Log Transform을 쓸까?
"""
적용 대상:
✅ 수입, 집값, 인구 (긴 꼬리 분포)
✅ 클릭 수, 조회 수
✅ 유전자 발현 데이터

효과:
- Skewness 감소 → 정규분포에 가까워짐
- 선형 모델 성능 향상
- 이상치 영향 감소

주의:
- 음수 값 처리: log(x + constant)
- 0 값 처리: log1p(x) = log(1 + x)
"""
```

---

## 2. Categorical Features (범주형 특징)

### 2.1 One-Hot Encoding

```python
# 의도: 범주형 → 이진 벡터
# 아이디어: 각 카테고리를 별도 Feature로

import pandas as pd

# 범주형 데이터
df = pd.DataFrame({
    'city': ['서울', '부산', '대구', '서울', '인천'],
    'education': ['대졸', '석사', '고졸', '박사', '대졸']
})

# 1. pandas get_dummies
df_onehot = pd.get_dummies(df, columns=['city', 'education'], prefix=['city', 'edu'])
print("One-Hot Encoding:")
print(df_onehot)

# 2. sklearn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' → n-1 encoding
encoded = encoder.fit_transform(df[['city']])

print("\nOneHotEncoder (drop first):")
print(f"Categories: {encoder.categories_}")
print(f"Encoded shape: {encoded.shape}")  # (5, 3) - 4개 도시 중 첫 번째 제외

# 장단점
"""
One-Hot Encoding:

장점:
✅ 모든 알고리즘에 사용 가능
✅ 간단하고 명확

단점:
❌ Cardinality 높으면 차원 폭발
   - 1000개 도시 → 1000개 columns
❌ Sparse matrix (메모리 낭비)
❌ Tree 모델에는 비효율적

언제 사용?
- Cardinality < 10-20
- Linear Model, SVM, Neural Network
"""
```

### 2.2 Label Encoding

```python
# 의도: 범주형 → 정수
# 아이디어: 순서 있는 범주 또는 Tree 모델

from sklearn.preprocessing import LabelEncoder

# 순서가 있는 범주
education_levels = pd.Series(['고졸', '대졸', '석사', '박사', '고졸', '대졸'])

# Label Encoding
le = LabelEncoder()
education_encoded = le.fit_transform(education_levels)

print("Label Encoding:")
for original, encoded in zip(education_levels, education_encoded):
    print(f"  {original:4s} → {encoded}")

# Inverse transform
decoded = le.inverse_transform(education_encoded)
print(f"\n복원: {decoded}")

# 주의사항
"""
Label Encoding 사용 시:

✅ 안전한 경우:
- Tree-based 모델 (XGBoost, Random Forest)
- 순서가 있는 범주 (교육수준, 등급)

❌ 위험한 경우:
- Linear Model, SVM (순서 관계 학습 가능성)
- 순서 없는 범주 (도시, 색상)

대안:
- 순서 없음 + Linear Model → One-Hot
- 순서 없음 + Tree Model → Label or One-Hot
- 순서 있음 → Ordinal Encoding
"""
```

### 2.3 Target Encoding (Mean Encoding)

```python
# 의도: Target 평균으로 인코딩
# 아이디어: 각 카테고리의 Target 통계 활용

import numpy as np
import pandas as pd

# 데이터
df = pd.DataFrame({
    'city': ['서울', '부산', '서울', '대구', '서울', '부산', '인천', '대구'],
    'target': [1, 0, 1, 0, 1, 0, 0, 1]
})

# Target Encoding
target_mean = df.groupby('city')['target'].mean()
df['city_target_encoded'] = df['city'].map(target_mean)

print("Target Encoding:")
print(df)

print("\n도시별 Target 평균:")
print(target_mean)

# 문제점: Target Leakage (과적합)
"""
Target Leakage 방지 기법:

1. K-Fold Cross-Validation Encoding:
   - Fold별로 다른 Encoding 사용
   - Train과 Test 분리

2. Smoothing (Regularization):
   - 전체 평균과 가중 평균
   - (n * target_mean + α * global_mean) / (n + α)

3. Noise 추가:
   - Encoding에 Gaussian noise
"""

# Smoothed Target Encoding
class TargetEncoder:
    """Target Encoding with Smoothing"""

    def __init__(self, alpha=10):
        self.alpha = alpha
        self.global_mean = None
        self.encoding_map = {}

    def fit(self, X, y):
        """학습"""
        self.global_mean = y.mean()

        # 각 카테고리별 평균 계산
        df = pd.DataFrame({'category': X, 'target': y})
        grouped = df.groupby('category')['target'].agg(['mean', 'count'])

        for category, row in grouped.iterrows():
            n = row['count']
            category_mean = row['mean']

            # Smoothing
            smoothed = (n * category_mean + self.alpha * self.global_mean) / (n + self.alpha)
            self.encoding_map[category] = smoothed

        return self

    def transform(self, X):
        """변환"""
        return X.map(lambda x: self.encoding_map.get(x, self.global_mean))

# 테스트
encoder = TargetEncoder(alpha=10)
encoder.fit(df['city'], df['target'])
df['city_smoothed'] = encoder.transform(df['city'])

print("\nSmoothed Target Encoding:")
print(df)

# 언제 사용?
"""
Target Encoding 적합:
✅ High Cardinality (100+ categories)
✅ Tree-based 모델 (LightGBM, CatBoost)
✅ Kaggle 경진대회 (성능 우선)

주의사항:
❌ Overfitting 위험 (Smoothing 필수)
❌ Time-series는 leakage 주의
❌ 프로덕션 환경에서는 신중히
"""
```

### 2.4 Frequency Encoding

```python
# 의도: 범주 빈도로 인코딩
# 아이디어: 빈도가 Target과 상관 있을 때

# 데이터
df = pd.DataFrame({
    'brand': ['Apple', 'Samsung', 'Apple', 'Xiaomi', 'Apple',
              'Samsung', 'Huawei', 'Xiaomi', 'Apple', 'Samsung']
})

# Frequency Encoding
freq_map = df['brand'].value_counts()
df['brand_frequency'] = df['brand'].map(freq_map)

print("Frequency Encoding:")
print(df)

print("\nBrand 빈도:")
print(freq_map)

# 정규화 (0-1 범위)
df['brand_frequency_normalized'] = df['brand'].map(freq_map) / len(df)

print("\n정규화된 Frequency:")
print(df[['brand', 'brand_frequency_normalized']])

# 언제 사용?
"""
Frequency Encoding 적합:
✅ 빈도가 Target과 상관
   - 예: 인기 브랜드 → 구매 확률 높음
✅ High Cardinality + One-Hot 불가
✅ 간단하고 빠름

장점:
- Target Leakage 없음
- 계산 빠름
- 새로운 카테고리도 처리 (frequency=0)
"""
```

---

## 3. Datetime Features (시간 특징)

### 3.1 기본 시간 Feature 추출

```python
# 의도: 날짜/시간에서 유의미한 Feature 추출
# 아이디어: 년, 월, 일, 요일, 시간 등

import pandas as pd

# 시계열 데이터
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='6H')
})

# Feature 추출
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['dayofweek'] = df['timestamp'].dt.dayofweek  # 0=월요일, 6=일요일
df['hour'] = df['timestamp'].dt.hour
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['quarter'] = df['timestamp'].dt.quarter

# 주기성 Feature
df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
df['day_of_year'] = df['timestamp'].dt.dayofyear

print("Datetime Features:")
print(df.head(10))

# 실전 예시: 전자상거래
"""
유용한 Datetime Features:

✅ 요일: 주말/평일 구매 패턴
✅ 시간대: 출퇴근 시간, 점심시간
✅ 월: 계절성, 급여일(월말)
✅ 분기: 사업 주기
✅ 공휴일: 별도 처리 필요
"""
```

### 3.2 Cyclical Features (주기적 특징)

```python
# 의도: 주기성을 Sine/Cosine으로 표현
# 아이디어: 12월(12)과 1월(1)이 가까움을 표현

import numpy as np

# 월 데이터 (1-12)
months = np.arange(1, 13)

# 잘못된 방법: 그냥 숫자 사용
print("잘못된 방법: 12월과 1월의 거리 =", abs(12 - 1))  # 11 (멀다고 인식)

# 올바른 방법: Sine/Cosine 변환
month_sin = np.sin(2 * np.pi * months / 12)
month_cos = np.cos(2 * np.pi * months / 12)

print("\n올바른 방법 (Sine/Cosine):")
print(f"12월: sin={month_sin[11]:.3f}, cos={month_cos[11]:.3f}")
print(f" 1월: sin={month_sin[0]:.3f}, cos={month_cos[0]:.3f}")

# Euclidean distance
dist_12_1 = np.sqrt((month_sin[11] - month_sin[0])**2 +
                     (month_cos[11] - month_cos[0])**2)
print(f"12월-1월 거리: {dist_12_1:.3f}")  # 작음 (가깝다고 인식)

# 일반화 함수
def encode_cyclical(data, max_val):
    """주기적 데이터를 Sine/Cosine으로 인코딩"""
    sin_encoded = np.sin(2 * np.pi * data / max_val)
    cos_encoded = np.cos(2 * np.pi * data / max_val)
    return sin_encoded, cos_encoded

# 적용
df['month_sin'], df['month_cos'] = encode_cyclical(df['month'], 12)
df['hour_sin'], df['hour_cos'] = encode_cyclical(df['hour'], 24)
df['dayofweek_sin'], df['dayofweek_cos'] = encode_cyclical(df['dayofweek'], 7)

print("\nCyclical Features:")
print(df[['timestamp', 'month', 'month_sin', 'month_cos']].head())

# 언제 사용?
"""
Cyclical Encoding 필수:
✅ 월 (1-12)
✅ 시간 (0-23)
✅ 요일 (0-6)
✅ 방향 (0-360도)

효과:
- 주기의 연속성 유지
- Tree 모델에도 도움됨
- Linear Model에 필수
"""
```

---

## 4. Interaction Features (상호작용 특징)

```python
# 의도: Feature 간 상호작용 포착
# 아이디어: x1 * x2, x1 / x2 등

import pandas as pd

# 주택 데이터
df = pd.DataFrame({
    'area_sqft': [1000, 1500, 2000, 2500],
    'bedrooms': [2, 3, 4, 5],
    'bathrooms': [1, 2, 2, 3],
    'age_years': [10, 5, 15, 20],
    'distance_to_city_km': [10, 5, 20, 30]
})

# 1. 곱셈 Interaction
df['area_per_bedroom'] = df['area_sqft'] / df['bedrooms']
df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']

# 2. 곱셈
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['room_density'] = df['total_rooms'] / df['area_sqft']

# 3. 도메인 지식 기반
df['value_score'] = (
    df['area_sqft'] * 0.5 +
    df['bedrooms'] * 100 -
    df['age_years'] * 10 -
    df['distance_to_city_km'] * 5
)

print("Interaction Features:")
print(df)

# 자동 Interaction 생성
from itertools import combinations

def create_interactions(df, columns, operations=['multiply', 'divide', 'add', 'subtract']):
    """모든 Feature 쌍에 대해 Interaction 생성"""

    new_features = {}

    for col1, col2 in combinations(columns, 2):
        if 'multiply' in operations:
            new_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]

        if 'divide' in operations and (df[col2] != 0).all():
            new_features[f'{col1}_div_{col2}'] = df[col1] / df[col2]

        if 'add' in operations:
            new_features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]

        if 'subtract' in operations:
            new_features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]

    return pd.DataFrame(new_features)

# 자동 생성
numeric_cols = ['area_sqft', 'bedrooms', 'age_years']
interactions = create_interactions(df, numeric_cols, operations=['multiply', 'divide'])

print(f"\n자동 생성된 Interaction Features: {interactions.shape[1]}개")
print(interactions.columns.tolist())

# 주의사항
"""
Interaction Features:

장점:
✅ 비선형 관계 포착
✅ 도메인 지식 반영
✅ Linear Model 성능 향상

단점:
❌ Feature 수 폭발 (n^2)
❌ Overfitting 위험
❌ 해석 어려움

실전 전략:
1. 도메인 지식 기반 선택 (5-10개)
2. Feature Importance로 검증
3. Regularization (Lasso) 적용
"""
```

---

## 5. Aggregation Features (집계 특징)

```python
# 의도: 그룹별 통계 Feature
# 아이디어: 평균, 합계, 최대, 최소 등

import pandas as pd
import numpy as np

# 거래 데이터
df = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
    'product_category': ['전자', '의류', '전자', '식품', '의류', '전자', '전자', '식품', '의류'],
    'amount': [1000, 500, 1500, 300, 800, 2000, 1200, 400, 600],
    'timestamp': pd.date_range('2024-01-01', periods=9, freq='D')
})

# 1. User별 집계
user_agg = df.groupby('user_id').agg({
    'amount': ['sum', 'mean', 'max', 'min', 'std', 'count'],
    'product_category': lambda x: x.nunique()  # 고유 카테고리 수
}).reset_index()

user_agg.columns = ['user_id', 'total_spent', 'avg_spent', 'max_spent',
                    'min_spent', 'std_spent', 'num_transactions', 'num_categories']

print("User별 집계 Features:")
print(user_agg)

# 2. Category별 집계
category_agg = df.groupby('product_category')['amount'].agg([
    'mean', 'median', 'std', 'count'
]).reset_index()

category_agg.columns = ['product_category', 'category_avg', 'category_median',
                        'category_std', 'category_count']

print("\nCategory별 집계 Features:")
print(category_agg)

# 3. 원본 데이터에 병합
df = df.merge(user_agg, on='user_id', how='left')
df = df.merge(category_agg, on='product_category', how='left')

# 4. Relative Features (상대적 특징)
df['amount_vs_user_avg'] = df['amount'] / df['avg_spent']
df['amount_vs_category_avg'] = df['amount'] / df['category_avg']

print("\n최종 Feature Set:")
print(df[['user_id', 'amount', 'avg_spent', 'amount_vs_user_avg']].head())

# 시계열 Aggregation
"""
시간 윈도우 집계:

- 최근 7일 평균
- 최근 30일 총합
- 전월 대비 증감률
"""

# Rolling Aggregation 예시
df_sorted = df.sort_values('timestamp')
df_sorted['amount_7d_mean'] = df_sorted.groupby('user_id')['amount'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

print("\nRolling Aggregation (window=3):")
print(df_sorted[['user_id', 'timestamp', 'amount', 'amount_7d_mean']])
```

---

## 6. 실전 Feature Engineering 파이프라인

### 6.1 전체 파이프라인

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class FeatureEngineeringPipeline:
    """실전 Feature Engineering 파이프라인"""

    def __init__(self):
        self.pipeline = None

    def build_pipeline(self, numeric_features, categorical_features):
        """파이프라인 구축"""

        # Numeric Pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical Pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        return self

    def fit_transform(self, X):
        """학습 및 변환"""
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        """변환"""
        return self.pipeline.transform(X)

# 사용 예시
# numeric_features = ['age', 'income', 'credit_score']
# categorical_features = ['city', 'education', 'employment']

# fe_pipeline = FeatureEngineeringPipeline()
# fe_pipeline.build_pipeline(numeric_features, categorical_features)
# X_transformed = fe_pipeline.fit_transform(X_train)
```

---

## 핵심 요약

### Feature Engineering 체크리스트

```python
"""
1. Numeric Features:
   □ Scaling (StandardScaler/MinMaxScaler/RobustScaler)
   □ Log transform (왜도 큰 경우)
   □ Polynomial features (비선형 관계)
   □ Binning (구간화)

2. Categorical Features:
   □ One-Hot Encoding (Cardinality < 20)
   □ Target Encoding (Cardinality 높음)
   □ Frequency Encoding
   □ Label Encoding (Tree 모델)

3. Datetime Features:
   □ 년, 월, 일, 요일, 시간 추출
   □ Cyclical encoding (Sine/Cosine)
   □ 공휴일 처리
   □ 시간 경과 (days_since_*)

4. Interaction Features:
   □ 도메인 지식 기반
   □ 곱셈/나눗셈
   □ Feature Importance로 검증

5. Aggregation Features:
   □ 그룹별 통계 (평균, 합계, 표준편차)
   □ Rolling statistics
   □ Relative features (vs 평균)

6. 검증:
   □ Feature Importance 확인
   □ Cross-validation 성능
   □ Overfitting 체크
"""
```

### 실무 우선순위

```
Impact vs Effort:

High Impact, Low Effort:
1. 결측치 처리
2. Scaling (필요 시)
3. One-Hot Encoding
4. 기본 Datetime Features

High Impact, High Effort:
5. Interaction Features (도메인 지식)
6. Aggregation Features
7. Target Encoding (with CV)

Low Impact:
8. Polynomial Features (degree > 2)
9. 자동 Feature 생성 (대부분)
```

다음: Hyperparameter Tuning →
