# Day 3-4: Advanced Linear Algebra for AI

## 🎯 목표

**AI의 핵심 기법들(PCA, SVD, LoRA)의 수학적 기초 완벽 이해!**

```python
# 이 장에서 배울 것:
- Eigenvalues & Eigenvectors → PCA, Stability Analysis
- SVD (Singular Value Decomposition) → Data Compression, LoRA
- Matrix Decomposition → Efficient Computation
- Rank → Low-rank Approximation
```

---

## 📖 1. Eigenvalues & Eigenvectors

### 정의

**"방향은 바뀌지 않고 크기만 변하는 특별한 벡터!"**

```python
A v = λ v

# A: Matrix (n×n)
# v: Eigenvector (방향)
# λ: Eigenvalue (크기)
```

### 직관

**선형 변환 A를 적용했을 때 방향이 그대로인 벡터**

```python
# 일반 벡터: 방향과 크기 모두 변함
x → A·x  # 완전히 다른 방향

# Eigenvector: 방향은 유지, 크기만 변함
v → A·v = λv  # 같은 방향, λ배 늘어남
```

### 예제

```python
import numpy as np

# Matrix
A = np.array([[4, 2],
              [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")
# [5. 2.]

print(f"Eigenvectors:\n{eigenvectors}")
# [[0.89442719 -0.70710678]
#  [0.4472136   0.70710678]]

# Verify: A·v = λ·v
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]

left = A @ v1
right = lambda1 * v1

print(f"A·v = {left}")
print(f"λ·v = {right}")
# Should be equal!
```

---

## 🔍 2. Eigendecomposition

### 공식

```python
A = Q Λ Q⁻¹

# Q: Eigenvectors as columns
# Λ: Diagonal matrix of eigenvalues
# Q⁻¹: Inverse of Q
```

### 왜 유용한가?

**1. 거듭제곱 계산이 쉬워짐!**

```python
# A^k = (Q Λ Q⁻¹)^k = Q Λ^k Q⁻¹

# Λ^k is easy (diagonal!)
Λ = [[λ₁, 0  ],
     [0,  λ₂]]

Λ^k = [[λ₁^k, 0   ],
       [0,    λ₂^k]]
```

**2. 안정성 분석**

```python
# Neural network의 gradient flow:
# ∂L/∂x_{t-1} = W^T ∂L/∂x_t

# Repeated multiplication:
# ∂L/∂x_0 = (W^T)^T ∂L/∂x_T

# Eigenvalue < 1: Vanishing gradients
# Eigenvalue > 1: Exploding gradients
```

### 구현

```python
def eigendecomposition(A):
    """
    Decompose A = Q Λ Q^{-1}
    """
    eigenvalues, Q = np.linalg.eig(A)
    Lambda = np.diag(eigenvalues)
    Q_inv = np.linalg.inv(Q)

    # Verify
    A_reconstructed = Q @ Lambda @ Q_inv
    assert np.allclose(A, A_reconstructed)

    return Q, Lambda, Q_inv

# Usage
A = np.array([[4, 2], [1, 3]])
Q, Lambda, Q_inv = eigendecomposition(A)

# Compute A^10 efficiently
A_10 = Q @ np.linalg.matrix_power(Lambda, 10) @ Q_inv
print(f"A^10 computed via eigendecomposition:\n{A_10}")
```

---

## 🎨 3. Principal Component Analysis (PCA)

### 목표

**고차원 데이터를 저차원으로 압축! (정보 손실 최소화)**

```python
# X: (n_samples, n_features)
# → X_reduced: (n_samples, k) where k << n_features
```

### 수학적 원리

**1. 중심화 (Centering)**

```python
X_centered = X - X.mean(axis=0)
```

**2. Covariance Matrix**

```python
Σ = (1/n) X_centered^T X_centered

# Σ[i,j] = covariance between feature i and j
```

**3. Eigendecomposition**

```python
Σ v_i = λ_i v_i

# v_i: Principal components (PCs)
# λ_i: Variance explained by PC_i
```

**4. Projection**

```python
# Keep top k eigenvectors
V_k = [v_1, v_2, ..., v_k]  # k largest eigenvalues

# Project
X_reduced = X_centered @ V_k
```

### 구현

```python
def pca(X, n_components=2):
    """
    PCA from scratch

    Args:
        X: (n_samples, n_features)
        n_components: Number of components to keep

    Returns:
        X_reduced: (n_samples, n_components)
        components: (n_components, n_features) - principal components
        explained_variance: Variance explained by each component
    """
    # 1. Center
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    # 2. Covariance matrix
    n_samples = X.shape[0]
    Sigma = (X_centered.T @ X_centered) / n_samples

    # 3. Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)

    # 4. Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. Select top k
    components = eigenvectors[:, :n_components].T
    explained_variance = eigenvalues[:n_components]

    # 6. Project
    X_reduced = X_centered @ components.T

    return X_reduced, components, explained_variance

# Example: Dimensionality reduction
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # (150, 4)

X_reduced, components, explained_var = pca(X, n_components=2)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance: {explained_var}")
print(f"Total variance explained: {explained_var.sum() / np.trace(np.cov(X.T)):.2%}")

# Visualize
import matplotlib.pyplot as plt

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=iris.target, cmap='viridis')
plt.xlabel(f'PC1 ({explained_var[0]:.2f})')
plt.ylabel(f'PC2 ({explained_var[1]:.2f})')
plt.title('PCA: Iris Dataset')
plt.colorbar()
plt.show()
```

### AI 응용: Feature Extraction

```python
# High-dimensional data (e.g., images: 784 dimensions)
# → PCA → Low-dimensional (e.g., 50 dimensions)
# → Faster training, less overfitting!
```

---

## 🔧 4. Singular Value Decomposition (SVD)

### 정의

**모든 행렬을 3개 행렬의 곱으로 분해!**

```python
A = U Σ V^T

# A: (m × n)
# U: (m × m) - Left singular vectors
# Σ: (m × n) - Diagonal, singular values
# V^T: (n × n) - Right singular vectors
```

### 특징

- **모든 행렬에 적용 가능** (정방행렬이 아니어도 OK!)
- **Eigendecomposition의 일반화**
- **항상 존재**

### 직관

```python
# A를 3단계 변환으로 분해:

# 1. V^T: Rotate in input space
# 2. Σ: Scale (stretch/compress)
# 3. U: Rotate in output space

X → [V^T rotation] → [Σ scaling] → [U rotation] → Y
```

### 구현

```python
def svd_decomposition(A):
    """
    SVD: A = U Σ V^T
    """
    U, sigma, VT = np.linalg.svd(A, full_matrices=True)

    # Create Σ matrix (m × n)
    m, n = A.shape
    Sigma = np.zeros((m, n))
    Sigma[:min(m,n), :min(m,n)] = np.diag(sigma)

    # Verify
    A_reconstructed = U @ Sigma @ VT
    assert np.allclose(A, A_reconstructed)

    return U, Sigma, VT

# Example
A = np.array([[1, 2, 3],
              [4, 5, 6]])

U, Sigma, VT = svd_decomposition(A)

print(f"U shape: {U.shape}")      # (2, 2)
print(f"Σ shape: {Sigma.shape}")  # (2, 3)
print(f"V^T shape: {VT.shape}")   # (3, 3)
print(f"Singular values: {np.diag(Sigma[:2,:2])}")
```

---

## 🎯 5. Low-Rank Approximation

### 목표

**큰 행렬을 작은 행렬들로 근사!**

```python
# Original: A (m × n)
# Approximation: A ≈ U_k Σ_k V_k^T

# Keep only top k singular values
# → Compression! Storage: mn → k(m+n+1)
```

### 수식

```python
A_k = Σ_{i=1}^k σ_i u_i v_i^T

# σ_i: i-th singular value
# u_i: i-th left singular vector
# v_i: i-th right singular vector
```

### 구현

```python
def low_rank_approximation(A, k):
    """
    Approximate A with rank-k matrix

    Args:
        A: (m, n) matrix
        k: Target rank

    Returns:
        A_k: Rank-k approximation
        compression_ratio: Original size / Compressed size
    """
    U, sigma, VT = np.linalg.svd(A, full_matrices=False)

    # Keep top k
    U_k = U[:, :k]
    sigma_k = sigma[:k]
    VT_k = VT[:k, :]

    # Reconstruct
    A_k = U_k @ np.diag(sigma_k) @ VT_k

    # Compression ratio
    m, n = A.shape
    original_size = m * n
    compressed_size = k * (m + n + 1)
    compression_ratio = original_size / compressed_size

    return A_k, compression_ratio

# Example: Image compression
from PIL import Image

# Load image
img = np.array(Image.open('image.jpg').convert('L'))  # Grayscale
print(f"Original shape: {img.shape}")

# Compress with different ranks
for k in [10, 50, 100]:
    img_compressed, ratio = low_rank_approximation(img, k)

    print(f"Rank {k}: Compression ratio = {ratio:.2f}x")

    # Reconstruction error
    error = np.linalg.norm(img - img_compressed, 'fro') / np.linalg.norm(img, 'fro')
    print(f"  Relative error: {error:.2%}")

    # Save
    plt.imsave(f'compressed_rank_{k}.png', img_compressed, cmap='gray')
```

---

## 🚀 6. LoRA (Low-Rank Adaptation)

### 핵심 아이디어

**거대 모델 fine-tuning을 저렴하게!**

```python
# Original: Update all weights
W_new = W_old + ΔW  # ΔW is huge!

# LoRA: ΔW ≈ B @ A (low-rank!)
W_new = W_old + B @ A

# B: (d × r), A: (r × k)
# r << d, k  (e.g., r=8, d=4096)
# → 99.9% parameter reduction!
```

### 수학적 근거

**Fine-tuning의 변화량은 low-rank!**

```python
# Hypothesis: ΔW has intrinsic low rank

# Therefore:
ΔW = B @ A

# Instead of updating d×k parameters,
# only update r×(d+k) parameters!
```

### 구현

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    LoRA adaptation layer

    W_new = W_frozen + B @ A
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()

        # Frozen original weight
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W.weight.requires_grad = False

        # LoRA matrices
        self.A = nn.Parameter(torch.randn(rank, in_features) / rank)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

    def forward(self, x):
        # Original path
        result = self.W(x)

        # LoRA path
        lora_result = (x @ self.A.T) @ self.B.T
        result += lora_result * self.scaling

        return result

# Example: Fine-tune a large linear layer
original_layer = nn.Linear(4096, 4096)

# Replace with LoRA
lora_layer = LoRALayer(4096, 4096, rank=8)
lora_layer.W.weight.data = original_layer.weight.data.clone()

# Parameter count
original_params = sum(p.numel() for p in original_layer.parameters())
lora_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)

print(f"Original parameters: {original_params:,}")  # 16,777,216
print(f"LoRA trainable parameters: {lora_params:,}")  # 65,536
print(f"Reduction: {original_params / lora_params:.1f}x")  # 256x!
```

### 왜 작동하는가?

```python
# 1. Fine-tuning은 작은 변화
#    → ΔW는 본질적으로 low-rank

# 2. SVD로 확인:
#    ΔW = U Σ V^T
#    → 상위 몇 개 singular values가 대부분의 정보

# 3. LoRA는 이를 명시적으로 모델링!
```

---

## 🎓 학습 목표

- [ ] Eigenvalues와 eigenvectors 계산
- [ ] PCA를 직접 구현
- [ ] SVD의 의미 이해
- [ ] Low-rank approximation 구현
- [ ] LoRA의 원리 이해

---

## 💡 핵심 정리

### Eigendecomposition vs SVD

| Feature | Eigendecomposition | SVD |
|---------|-------------------|-----|
| 적용 | 정방행렬만 | 모든 행렬 |
| 공식 | A = QΛQ⁻¹ | A = UΣV^T |
| 조건 | Symmetric 권장 | 항상 가능 |
| 용도 | Stability, PCA | Compression, LoRA |

### Rank의 의미

```python
# Rank = 독립적인 정보의 차원

# Full rank: All information
# Low rank: Redundancy (압축 가능!)

# AI 응용:
# - 이미지: 저주파 성분 → low rank
# - Fine-tuning: 작은 변화 → low rank
# - LoRA: 명시적 low-rank constraint
```

---

## 📚 실전 예제: PCA로 얼굴 압축

```python
from sklearn.datasets import fetch_lfw_people

# Load face dataset
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = faces.data  # (n_samples, n_features)

print(f"Original: {X.shape}")  # (1288, 1850)

# Apply PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=150)  # Keep 150 components
X_reduced = pca.fit_transform(X)

print(f"Compressed: {X_reduced.shape}")  # (1288, 150)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Reconstruct
X_reconstructed = pca.inverse_transform(X_reduced)

# Visualize
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    # Original
    axes[0, i].imshow(X[i].reshape(50, 37), cmap='gray')
    axes[0, i].set_title('Original')
    axes[0, i].axis('off')

    # Reconstructed
    axes[1, i].imshow(X_reconstructed[i].reshape(50, 37), cmap='gray')
    axes[1, i].set_title('Compressed (150 dims)')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
```

---

## ⏭️ 다음

👉 [Day 5-6: Calculus](./03-calculus.md)

**이제 미적분으로 Backpropagation의 수학을 이해해봅시다!** 📈
