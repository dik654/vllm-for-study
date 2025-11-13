# Day 12-14: Putting It All Together

## 🎯 최종 목표

**모든 수학 개념을 통합하여 Neural Network를 밑바닥부터 완전히 구현!**

```python
# 지금까지 배운 것:
✅ 선형대수: Matrix operations, Dot products
✅ 미적분: Derivatives, Chain rule, Gradients
✅ 확률: Distributions, MLE, Sampling
✅ 정보이론: Entropy, Cross entropy, KL divergence
✅ 최적화: Gradient descent, Adam

# 이제 모두 합쳐서:
→ 2-Layer Neural Network 완전 구현!
```

---

## 🏗️ Part 1: Softmax의 수학 완전 분해

### 1.1 Forward Pass

**선형대수 + 확률**

```python
# Logits (선형 변환)
z = W @ x + b  # Matrix multiplication (선형대수!)

# Softmax (확률로 변환)
p_i = exp(z_i) / Σ_j exp(z_j)  # 확률 분포 (확률론!)

# Properties:
# - p_i ≥ 0 (모두 양수)
# - Σ p_i = 1 (합이 1)
# → Valid probability distribution!
```

**구현 (Numerically Stable)**

```python
import numpy as np

def softmax(z):
    """
    Numerically stable softmax

    수학:
    p_i = exp(z_i) / Σ exp(z_j)

    문제: exp(z_i) overflow!
    해결: exp(z_i - max(z)) / Σ exp(z_j - max(z))
    """
    # Shift for numerical stability
    z_shifted = z - np.max(z, axis=-1, keepdims=True)

    # Softmax
    exp_z = np.exp(z_shifted)
    probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    return probs

# Test
z = np.array([1.0, 2.0, 3.0])
probs = softmax(z)

print(f"Logits: {z}")
print(f"Probabilities: {probs}")
print(f"Sum: {probs.sum()}")  # Should be 1.0
```

### 1.2 Cross Entropy Loss

**정보이론**

```python
# True distribution: y (one-hot)
# Predicted distribution: p (softmax output)

L = -Σ y_i log(p_i)

# For one-hot y (only y_true = 1):
L = -log(p_true)

# 직관: "올바른 클래스의 확률을 최대화!"
```

**구현**

```python
def cross_entropy_loss(y_true, y_pred):
    """
    Cross entropy loss

    Args:
        y_true: (batch, num_classes) - one-hot
        y_pred: (batch, num_classes) - probabilities

    Returns:
        loss: scalar
    """
    # Clip to avoid log(0)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Loss
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    return loss

# Test
y_true = np.array([[0, 1, 0],    # Class 1
                   [1, 0, 0]])    # Class 0
y_pred = np.array([[0.1, 0.7, 0.2],
                   [0.8, 0.1, 0.1]])

loss = cross_entropy_loss(y_true, y_pred)
print(f"Cross Entropy Loss: {loss:.4f}")
```

### 1.3 Gradient (Backpropagation)

**미적분 + Chain Rule**

```python
# Forward:
z = W @ x + b
p = softmax(z)
L = -log(p[y_true])

# Backward (Chain rule!):
∂L/∂z = p - y_true  # 놀랍도록 간단!

# 증명:
# ∂L/∂z_i = ∂L/∂p_j * ∂p_j/∂z_i (chain rule)
#         = Σ_j (-y_j/p_j) * ∂p_j/∂z_i
#         = ... (algebra)
#         = p_i - y_i
```

**구현**

```python
def softmax_cross_entropy_gradient(y_true, y_pred):
    """
    Combined gradient: ∂L/∂z = p - y

    This is the gradient w.r.t. logits (before softmax)
    """
    return y_pred - y_true

# Test
gradient = softmax_cross_entropy_gradient(y_true, y_pred)
print(f"Gradient:\n{gradient}")

# Interpretation:
# - Positive: Predicted too high (reduce)
# - Negative: Predicted too low (increase)
```

---

## 🧠 Part 2: 2-Layer Neural Network 완전 구현

### 2.1 Forward Pass

**전체 수식**

```python
# Layer 1
z1 = W1 @ x + b1        # (선형대수)
a1 = ReLU(z1)           # (activation)

# Layer 2
z2 = W2 @ a1 + b2       # (선형대수)
p = softmax(z2)         # (확률)

# Loss
L = -log(p[y_true])     # (정보이론)
```

**구현**

```python
class TwoLayerNN:
    """
    2-Layer Neural Network from scratch

    Architecture:
    Input → [Linear → ReLU] → [Linear → Softmax] → Output
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize weights (He initialization)
        """
        # Layer 1
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((hidden_dim, 1))

        # Layer 2
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((output_dim, 1))

    def relu(self, z):
        """ReLU activation"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """ReLU derivative"""
        return (z > 0).astype(float)

    def forward(self, X):
        """
        Forward pass

        Args:
            X: (input_dim, batch_size)

        Returns:
            predictions: (output_dim, batch_size)
            cache: Intermediate values for backprop
        """
        # Layer 1
        self.z1 = self.W1 @ X + self.b1
        self.a1 = self.relu(self.z1)

        # Layer 2
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = softmax(self.z2)

        # Cache for backprop
        self.cache = {
            'X': X,
            'z1': self.z1,
            'a1': self.a1,
            'z2': self.z2,
            'a2': self.a2
        }

        return self.a2

    def compute_loss(self, y_true):
        """
        Cross entropy loss
        """
        batch_size = y_true.shape[1]
        loss = -np.sum(y_true * np.log(self.a2 + 1e-12)) / batch_size
        return loss

    def backward(self, y_true):
        """
        Backpropagation (Chain rule!)

        수학:
        ∂L/∂W2 = ∂L/∂z2 @ a1^T
        ∂L/∂b2 = ∂L/∂z2

        ∂L/∂a1 = W2^T @ ∂L/∂z2
        ∂L/∂z1 = ∂L/∂a1 ⊙ ReLU'(z1)

        ∂L/∂W1 = ∂L/∂z1 @ X^T
        ∂L/∂b1 = ∂L/∂z1
        """
        batch_size = y_true.shape[1]

        # Get cached values
        X = self.cache['X']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        a2 = self.cache['a2']

        # Layer 2 gradients
        dz2 = a2 - y_true  # ∂L/∂z2 (softmax + cross entropy!)
        dW2 = (dz2 @ a1.T) / batch_size
        db2 = np.sum(dz2, axis=1, keepdims=True) / batch_size

        # Layer 1 gradients (chain rule!)
        da1 = self.W2.T @ dz2  # ∂L/∂a1
        dz1 = da1 * self.relu_derivative(z1)  # ∂L/∂z1
        dW1 = (dz1 @ X.T) / batch_size
        db1 = np.sum(dz1, axis=1, keepdims=True) / batch_size

        # Store gradients
        self.gradients = {
            'dW2': dW2, 'db2': db2,
            'dW1': dW1, 'db1': db1
        }

    def update_parameters(self, learning_rate):
        """
        Gradient descent update
        """
        self.W2 -= learning_rate * self.gradients['dW2']
        self.b2 -= learning_rate * self.gradients['db2']
        self.W1 -= learning_rate * self.gradients['dW1']
        self.b1 -= learning_rate * self.gradients['db1']

    def train_step(self, X, y_true, learning_rate):
        """
        Single training step

        1. Forward
        2. Compute loss
        3. Backward
        4. Update
        """
        # Forward
        predictions = self.forward(X)

        # Loss
        loss = self.compute_loss(y_true)

        # Backward
        self.backward(y_true)

        # Update
        self.update_parameters(learning_rate)

        return loss

# Create model
model = TwoLayerNN(input_dim=784, hidden_dim=128, output_dim=10)

print(f"W1 shape: {model.W1.shape}")  # (128, 784)
print(f"W2 shape: {model.W2.shape}")  # (10, 128)
```

---

## 🎯 Part 3: MNIST 훈련

### 3.1 데이터 준비

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float32').values
y = mnist.target.astype('int').values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to one-hot
def to_one_hot(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

y_train_onehot = to_one_hot(y_train)
y_test_onehot = to_one_hot(y_test)

print(f"Train: {X_train.shape}, {y_train_onehot.shape}")
print(f"Test: {X_test.shape}, {y_test_onehot.shape}")
```

### 3.2 훈련 루프

```python
def train(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=128, lr=0.01):
    """
    Training loop
    """
    n_samples = X_train.shape[0]
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        n_batches = 0

        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size].T
            y_batch = y_shuffled[i:i+batch_size].T

            # Train step
            loss = model.train_step(X_batch, y_batch, lr)
            epoch_loss += loss
            n_batches += 1

        # Average loss
        avg_loss = epoch_loss / n_batches

        # Validation
        val_loss, val_acc = evaluate(model, X_val, y_val)

        # Record
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss = {avg_loss:.4f}, "
              f"Val Loss = {val_loss:.4f}, "
              f"Val Acc = {val_acc:.2%}")

    return history

def evaluate(model, X, y):
    """
    Evaluate on validation/test set
    """
    X_t = X.T
    y_t = y.T

    # Forward
    predictions = model.forward(X_t)

    # Loss
    loss = -np.sum(y_t * np.log(predictions + 1e-12)) / X.shape[0]

    # Accuracy
    pred_labels = np.argmax(predictions, axis=0)
    true_labels = np.argmax(y_t, axis=0)
    accuracy = np.mean(pred_labels == true_labels)

    return loss, accuracy

# Train model
model = TwoLayerNN(input_dim=784, hidden_dim=128, output_dim=10)

history = train(
    model,
    X_train, y_train_onehot,
    X_test, y_test_onehot,
    epochs=20,
    batch_size=128,
    lr=0.1
)

# Final test accuracy
test_loss, test_acc = evaluate(model, X_test, y_test_onehot)
print(f"\nFinal Test Accuracy: {test_acc:.2%}")
```

### 3.3 시각화

```python
import matplotlib.pyplot as plt

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy
axes[1].plot(history['val_acc'])
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Validation Accuracy')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Visualize predictions
def visualize_predictions(model, X, y, num_samples=10):
    """Show predictions on test samples"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    indices = np.random.choice(X.shape[0], num_samples, replace=False)

    for i, idx in enumerate(indices):
        # Predict
        x_sample = X[idx:idx+1].T
        pred = model.forward(x_sample)
        pred_label = np.argmax(pred)
        true_label = np.argmax(y[idx])

        # Plot
        axes[i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

visualize_predictions(model, X_test, y_test_onehot)
```

---

## 🎓 Part 4: 개념 검증

### 4.1 Gradient Checking

**수치 미분으로 backprop 검증!**

```python
def numerical_gradient(model, X, y, param_name, epsilon=1e-5):
    """
    Compute gradient numerically

    f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)
    """
    param = getattr(model, param_name)
    grad_numerical = np.zeros_like(param)

    # Iterate over all elements
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index

        # Save original value
        original_value = param[idx]

        # f(x + epsilon)
        param[idx] = original_value + epsilon
        model.forward(X)
        loss_plus = model.compute_loss(y)

        # f(x - epsilon)
        param[idx] = original_value - epsilon
        model.forward(X)
        loss_minus = model.compute_loss(y)

        # Gradient
        grad_numerical[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        # Restore
        param[idx] = original_value

        it.iternext()

    return grad_numerical

def gradient_check(model, X, y):
    """
    Check if backprop gradients match numerical gradients
    """
    # Forward + backward
    model.forward(X)
    model.compute_loss(y)
    model.backward(y)

    # Check each parameter
    for param_name in ['W1', 'b1', 'W2', 'b2']:
        grad_backprop = model.gradients[f'd{param_name}']
        grad_numerical = numerical_gradient(model, X, y, param_name)

        # Relative error
        diff = np.linalg.norm(grad_backprop - grad_numerical)
        norm = np.linalg.norm(grad_backprop) + np.linalg.norm(grad_numerical)
        relative_error = diff / norm

        print(f"{param_name}: Relative error = {relative_error:.2e}")

        if relative_error < 1e-5:
            print(f"  ✓ Gradient correct!")
        else:
            print(f"  ✗ Gradient may be wrong!")

# Test
X_sample = X_train[:10].T
y_sample = y_train_onehot[:10].T

gradient_check(model, X_sample, y_sample)
```

### 4.2 Overfitting Test

**단일 배치로 overfitting 가능한지 확인!**

```python
def overfitting_test(model, X, y, steps=1000, lr=0.1):
    """
    Try to overfit a single batch

    If successful: Model has enough capacity
    If failed: Bug in implementation!
    """
    losses = []

    for step in range(steps):
        loss = model.train_step(X, y, lr)
        losses.append(loss)

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")

    # Check if loss decreased significantly
    if losses[-1] < 0.1:
        print("✓ Overfitting successful! Model works correctly.")
    else:
        print("✗ Failed to overfit. Check implementation!")

    return losses

# Test on single batch
X_batch = X_train[:32].T
y_batch = y_train_onehot[:32].T

losses = overfitting_test(model, X_batch, y_batch)

# Plot
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Overfitting Test (Single Batch)')
plt.grid(True)
plt.show()
```

---

## 🎉 축하합니다!

**모든 수학 개념을 통합하여 Neural Network를 완전히 구현했습니다!**

### 사용한 모든 수학:

✅ **선형대수**
- Matrix multiplication (W @ x)
- Dot product (유사도)
- Transpose (W^T)

✅ **미적분**
- Derivatives (∂L/∂W)
- Chain rule (backpropagation!)
- Gradients (모든 파라미터의 변화 방향)

✅ **확률/통계**
- Gaussian (weight initialization)
- Softmax (probability distribution)
- Expectation (loss = average)

✅ **정보이론**
- Cross entropy (loss function)
- Entropy (uncertainty)

✅ **최적화**
- Gradient descent (parameter update)
- Learning rate (step size)

---

## 🚀 다음 단계

이제 당신은:
- ✅ PyTorch/TensorFlow 내부 작동 원리 이해
- ✅ 논문의 모든 수식 읽기 가능
- ✅ 새로운 아키텍처 설계 가능
- ✅ 디버깅 능력 향상

**Phase 0으로 진행하여 HuggingFace로 실전 AI를 배워봅시다!** 🎓

👉 [Phase 0: HuggingFace Ecosystem](../phase0-huggingface-ecosystem/)
