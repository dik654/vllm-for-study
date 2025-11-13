# Day 1-2: 선형대수 기초 - AI의 언어

## 🎯 왜 선형대수인가?

**"AI는 결국 거대한 행렬 곱셈입니다."**

### AI에서 선형대수가 쓰이는 곳
- **Attention**: Q·Kᵀ (행렬 곱셈)
- **Linear Layer**: W·x + b (행렬 곱셈)
- **Embeddings**: Lookup = 행렬의 한 행 추출
- **Batch Processing**: 여러 입력 = 행렬
- **Image**: 픽셀들의 행렬

→ **Transformer, CNN, GAN 모두 선형대수!**

---

## 1. Vectors (벡터)

### 📖 개념

**Vector**는 크기와 방향을 가진 화살표입니다.

```
v = [2, 3]  →  2차원 평면의 점 or 방향
```

### AI에서의 의미
- **Word Embedding**: "cat" = [0.2, -0.5, 0.8, ...] (768차원 벡터)
- **Image**: 28×28 이미지 = 784차원 벡터로 펼치기
- **Hidden State**: BERT의 한 토큰 = 768차원 벡터

### 💻 실습 1: Vector 기본

```python
# vectors_basics.py
import numpy as np
import matplotlib.pyplot as plt

# Vector 생성
v1 = np.array([2, 3])
v2 = np.array([1, -1])

print(f"v1 = {v1}")
print(f"v2 = {v2}")

# Vector 시각화
plt.figure(figsize=(8, 8))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', width=0.006)
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', width=0.006)
plt.xlim(-1, 4)
plt.ylim(-2, 4)
plt.grid()
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.text(v1[0], v1[1]+0.2, 'v1', fontsize=14, color='r')
plt.text(v2[0], v2[1]+0.2, 'v2', fontsize=14, color='b')
plt.title('Vectors in 2D Space')
plt.savefig('vectors.png', dpi=150)
print("✓ 시각화 저장: vectors.png")

# Vector 덧셈
v_sum = v1 + v2
print(f"\nv1 + v2 = {v_sum}")

# Vector 스칼라 곱
v_scaled = 2 * v1
print(f"2 * v1 = {v_scaled}")

# Vector 크기 (Norm)
v1_norm = np.linalg.norm(v1)
print(f"\n||v1|| = {v1_norm:.3f}")
```

### 🧮 연습 문제

1. 벡터 [3, 4]의 크기는? (손으로 계산 후 코드로 검증)
2. AI에서: Word embedding [0.5, -0.3, 0.2]의 크기는?

---

## 2. 내적 (Dot Product) ⭐ **초중요!**

### 📖 개념

두 벡터의 **내적**:

$$
v \cdot w = v_1 w_1 + v_2 w_2 + ... + v_n w_n
$$

기하학적 의미:
$$
v \cdot w = ||v|| \cdot ||w|| \cdot \cos(\theta)
$$

### AI에서의 의미

**내적 = 유사도!**

- $v \cdot w > 0$: 같은 방향 (유사)
- $v \cdot w = 0$: 직각 (무관)
- $v \cdot w < 0$: 반대 방향 (반대)

### 🔥 AI 응용: Attention의 핵심!

```python
Query = [0.5, 0.3, 0.2]  # "cat"의 query
Key   = [0.4, 0.4, 0.1]  # "animal"의 key

# Attention score = Query · Key
score = np.dot(Query, Key)
# → 높으면 "관련있음", 낮으면 "관련없음"
```

### 💻 실습 2: Dot Product와 유사도

```python
# dot_product_similarity.py
import numpy as np

# 단어 임베딩 (간단한 3차원 예시)
embeddings = {
    "cat":    np.array([1.0, 0.8, 0.1]),
    "dog":    np.array([0.9, 0.7, 0.2]),
    "car":    np.array([0.1, 0.2, 1.0]),
    "truck":  np.array([0.2, 0.3, 0.9]),
}

def cosine_similarity(v1, v2):
    """코사인 유사도 = dot product / (norm * norm)"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# "cat"과 다른 단어들의 유사도
query_word = "cat"
query_vec = embeddings[query_word]

print(f"'{query_word}'와의 유사도:\n")
for word, vec in embeddings.items():
    if word != query_word:
        sim = cosine_similarity(query_vec, vec)
        print(f"  {word:8s}: {sim:.4f}")

# 예상 결과:
# dog와 가장 유사 (동물끼리)
# car/truck과는 낮은 유사도

# AI 실전: Attention Score 계산
print("\n=== Attention Score 계산 ===")
Q = np.array([0.5, 0.3])  # Query
K = np.array([0.4, 0.6])  # Key

score = np.dot(Q, K)
print(f"Q · K = {score:.4f}")
print("→ 이 값이 크면 Query가 Key에 '주목(attend)'합니다!")
```

### 🧮 손 계산 연습

**문제**: Query = [1, 2], Key = [3, 4]일 때, Attention score는?

<details>
<summary>답 보기</summary>

```
Q · K = 1×3 + 2×4 = 3 + 8 = 11
```

</details>

---

## 3. Matrices (행렬)

### 📖 개념

**Matrix**는 숫자를 직사각형으로 배열한 것:

$$
A = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

- **Shape**: (2, 3) = 2행 3열
- **AI**: 모든 것이 행렬!

### AI에서의 행렬

```python
# Batch of embeddings
batch = [[0.1, 0.2, 0.3],   # 문장 1의 토큰 1
         [0.4, 0.5, 0.6],   # 문장 1의 토큰 2
         [0.7, 0.8, 0.9]]   # 문장 1의 토큰 3
# → Shape: (3, 3) = (sequence_length, embedding_dim)

# Linear layer weights
W = [[w11, w12, w13],
     [w21, w22, w23]]
# → Shape: (output_dim, input_dim)
```

### 💻 실습 3: Matrix 기본

```python
# matrices_basics.py
import numpy as np

# Matrix 생성
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Matrix A:")
print(A)
print(f"Shape: {A.shape}")  # (2, 3)

# 전치 (Transpose)
A_T = A.T
print("\nA.T (Transpose):")
print(A_T)
print(f"Shape: {A_T.shape}")  # (3, 2)

# 왜 Transpose?
# Attention에서: Q·K^T를 하려면 K를 transpose!

# 행렬 인덱싱
print(f"\nA[0, 1] = {A[0, 1]}")  # 0행 1열 = 2
print(f"A[1, :] = {A[1, :]}")    # 1행 전체 = [4, 5, 6]
print(f"A[:, 2] = {A[:, 2]}")    # 2열 전체 = [3, 6]
```

---

## 4. Matrix Multiplication ⭐⭐⭐ **가장 중요!**

### 📖 개념

행렬 곱셈 $C = A \times B$:
- A의 shape: (m, n)
- B의 shape: (n, p)
- C의 shape: (m, p)

**핵심**: A의 열 개수 = B의 행 개수

각 원소:
$$
C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}
$$

### 🎨 직관: "행과 열의 내적"

```
A의 i번째 행 · B의 j번째 열 = C[i,j]
```

### 💻 실습 4: Matrix Multiplication (손 계산 → 검증)

```python
# matrix_multiplication.py
import numpy as np

# 작은 예제로 손 계산 연습
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# NumPy로 계산
C = A @ B  # @ = matrix multiplication
print("A @ B =")
print(C)

# 손으로 계산:
# C[0,0] = 1×5 + 2×7 = 5 + 14 = 19
# C[0,1] = 1×6 + 2×8 = 6 + 16 = 22
# C[1,0] = 3×5 + 4×7 = 15 + 28 = 43
# C[1,1] = 3×6 + 4×8 = 18 + 32 = 50

print("\n손 계산 검증:")
print("C[0,0] = 1×5 + 2×7 =", 1*5 + 2*7)
print("C[0,1] = 1×6 + 2×8 =", 1*6 + 2*8)
print("C[1,0] = 3×5 + 4×7 =", 3*5 + 4*7)
print("C[1,1] = 3×6 + 4×8 =", 3*6 + 4*8)
```

### 🔥 실습 5: AI 실전 - Linear Layer

```python
# linear_layer_math.py
import numpy as np

# Linear layer: y = W @ x + b
# x: 입력 (input_dim,)
# W: 가중치 (output_dim, input_dim)
# b: bias (output_dim,)
# y: 출력 (output_dim,)

input_dim = 3
output_dim = 2

# 입력 (예: word embedding)
x = np.array([0.5, -0.3, 0.8])

# 가중치 (학습됨)
W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]])

# Bias
b = np.array([0.1, -0.1])

# Forward pass
y = W @ x + b

print("입력 x:", x)
print("\n가중치 W:")
print(W)
print("\nBias b:", b)
print("\n출력 y = W @ x + b:")
print(y)

# 손 계산 검증
print("\n=== 손 계산 ===")
y0 = W[0,0]*x[0] + W[0,1]*x[1] + W[0,2]*x[2] + b[0]
y1 = W[1,0]*x[0] + W[1,1]*x[1] + W[1,2]*x[2] + b[1]
print(f"y[0] = {y0:.4f}")
print(f"y[1] = {y1:.4f}")

# PyTorch 비교
import torch
import torch.nn as nn

linear = nn.Linear(input_dim, output_dim)
linear.weight.data = torch.tensor(W, dtype=torch.float32)
linear.bias.data = torch.tensor(b, dtype=torch.float32)

x_torch = torch.tensor(x, dtype=torch.float32)
y_torch = linear(x_torch)

print("\n=== PyTorch 결과 ===")
print(y_torch.detach().numpy())
print("✓ NumPy와 일치!")
```

### 🧮 연습 문제

**문제**: 다음 행렬을 곱하시오.

$$
A = \begin{bmatrix}2 & 1\\3 & 4\end{bmatrix}, \quad
B = \begin{bmatrix}1\\2\end{bmatrix}
$$

<details>
<summary>답 보기</summary>

$$
A \times B = \begin{bmatrix}2×1 + 1×2\\3×1 + 4×2\end{bmatrix} = \begin{bmatrix}4\\11\end{bmatrix}
$$

</details>

---

## 5. AI 실전: Attention 수식 완전 분해

### 🎯 목표: "Attention is All You Need"의 핵심 수식 이해

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 단계별 분해

```python
# attention_from_math.py
import numpy as np

# 간단한 예: 3개 토큰, 각각 2차원 임베딩
Q = np.array([[1.0, 0.5],   # Query for token 1
              [0.5, 1.0],   # Query for token 2
              [0.3, 0.7]])  # Query for token 3

K = np.array([[0.8, 0.4],   # Key for token 1
              [0.6, 0.9],   # Key for token 2
              [0.4, 0.6]])  # Key for token 3

V = np.array([[2.0, 1.0],   # Value for token 1
              [1.5, 2.5],   # Value for token 2
              [3.0, 1.8]])  # Value for token 3

d_k = Q.shape[1]  # 2

print("Q shape:", Q.shape)  # (3, 2)
print("K shape:", K.shape)  # (3, 2)
print("V shape:", V.shape)  # (3, 2)

# Step 1: Q @ K^T
scores = Q @ K.T  # (3, 2) @ (2, 3) = (3, 3)
print("\n=== Step 1: Q @ K^T ===")
print(scores)
print(f"Shape: {scores.shape}")
print("→ scores[i,j] = Query i가 Key j에 주목하는 정도")

# Step 2: Scale by sqrt(d_k)
scores_scaled = scores / np.sqrt(d_k)
print("\n=== Step 2: Scaled Scores ===")
print(scores_scaled)
print(f"→ sqrt(d_k) = {np.sqrt(d_k):.3f}로 나눔")

# Step 3: Softmax (각 행마다)
def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

attention_weights = softmax(scores_scaled, axis=-1)
print("\n=== Step 3: Attention Weights (Softmax) ===")
print(attention_weights)
print("→ 각 행의 합 =", attention_weights.sum(axis=1))  # [1, 1, 1]
print("→ 확률 분포!")

# Step 4: Weighted sum of Values
output = attention_weights @ V  # (3, 3) @ (3, 2) = (3, 2)
print("\n=== Step 4: Output = Attention @ V ===")
print(output)
print(f"Shape: {output.shape}")

# 해석
print("\n=== 해석 ===")
print("Token 0의 출력:")
print(f"  = {attention_weights[0,0]:.3f} × V[0] + "
      f"{attention_weights[0,1]:.3f} × V[1] + "
      f"{attention_weights[0,2]:.3f} × V[2]")
print(f"  = {output[0]}")
print("→ 다른 토큰들의 Value를 attention weight로 가중평균!")
```

### 🧠 핵심 직관

```
Attention = "어디에 집중할까?"를 학습

1. Q·K^T: 모든 토큰 쌍의 유사도 계산
2. Softmax: 유사도를 확률로 변환
3. × V: 관련있는 Value를 많이 가져옴
```

### 🧮 손 계산 연습

Q = [1, 0], K = [1, 0], V = [2, 3]일 때,
1. Q·K^T = ?
2. Softmax(1.0) = ? (단일 값이면 1)
3. Output = ?

---

## 6. Transpose의 중요성

### 📖 왜 K^T를 하는가?

```python
# Q shape: (seq_len, d_k)
# K shape: (seq_len, d_k)
# Q @ K는 불가능! (2번째 차원 ≠ 1번째 차원)

# K^T shape: (d_k, seq_len)
# Q @ K^T: (seq_len, d_k) @ (d_k, seq_len) = (seq_len, seq_len) ✓
```

**결과**: (seq_len, seq_len) 행렬 = 모든 토큰 쌍의 attention score!

---

## ✅ Day 1-2 완료 체크리스트

### 이론 이해
- [ ] Vector의 크기(norm) 계산 가능
- [ ] Dot product가 유사도임을 이해
- [ ] 행렬 곱셈의 shape 규칙 이해
- [ ] Transpose의 용도 이해

### 손 계산
- [ ] 2×2 행렬 곱셈을 손으로 계산
- [ ] Vector 내적을 손으로 계산
- [ ] 간단한 Attention score 계산

### 코드 구현
- [ ] NumPy로 dot product 구현
- [ ] 코사인 유사도 계산
- [ ] Matrix multiplication 검증
- [ ] 간단한 Attention 구현

### AI 연결
- [ ] Linear layer가 행렬 곱셈임을 이해
- [ ] Attention 수식의 각 단계 설명 가능
- [ ] Q·K^T의 의미 이해

---

## 🎯 최종 챌린지

**Attention을 완전히 이해했는지 테스트!**

다음을 **손으로 계산**하고 **코드로 검증**:

```
Q = [[1, 2]]  (1개 query)
K = [[3, 4], [5, 6]]  (2개 keys)
V = [[7, 8], [9, 10]]

1. Q @ K^T = ?
2. Softmax 후 attention weights = ?
3. 최종 output = ?
```

<details>
<summary>답 확인</summary>

```python
Q = np.array([[1, 2]])
K = np.array([[3, 4], [5, 6]])
V = np.array([[7, 8], [9, 10]])

scores = Q @ K.T  # [[11, 17]]
weights = softmax(scores)  # 계산해보세요!
output = weights @ V
```

</details>

---

## ⏭️ 다음 단계

선형대수의 기초를 마스터했습니다! 🎉

다음은 더 심화된 내용:

👉 [Day 3-4: 선형대수 심화 (SVD, LoRA)](./02-linear-algebra-advanced.md)

**"이제 Attention 수식이 읽힙니다!"** 🚀
