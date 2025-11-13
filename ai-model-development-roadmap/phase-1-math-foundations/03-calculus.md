# Day 5-6: 미적분 - Backpropagation의 수학

## 🎯 왜 미적분인가?

**"AI 학습 = Gradient를 따라 내려가기"**

### AI에서 미적분이 쓰이는 곳
- **Backpropagation**: Chain rule로 모든 gradient 계산
- **Gradient Descent**: ∂Loss/∂W를 구해서 가중치 업데이트
- **Optimizer**: Adam, RMSprop 모두 gradient 기반

---

## 1. Derivative (미분) 기초

### 📖 개념

**미분** = 함수의 순간 변화율

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

**직관**: "x를 조금 증가시키면 f(x)가 얼마나 변하나?"

### AI에서의 의미

```
Loss(W) = 현재 loss 값
dLoss/dW = W를 조금 바꾸면 Loss가 얼마나 변하나?
→ W를 어느 방향으로 업데이트해야 Loss가 줄어드나?
```

### 💻 실습 1: 미분의 직관

```python
# derivative_intuition.py
import numpy as np
import matplotlib.pyplot as plt

# 함수: f(x) = x^2
def f(x):
    return x**2

# 미분: f'(x) = 2x
def f_prime(x):
    return 2*x

# 수치적 미분 (정의에 따라)
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

# 테스트
x = 3.0
analytical = f_prime(x)  # 2 * 3 = 6
numerical = numerical_derivative(f, x)

print(f"x = {x}")
print(f"f(x) = {f(x)}")
print(f"f'(x) 해석적: {analytical}")
print(f"f'(x) 수치적: {numerical:.6f}")
print(f"오차: {abs(analytical - numerical):.10f}")

# 시각화
x_vals = np.linspace(-2, 4, 100)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'b-', label='f(x) = x²', linewidth=2)

# 점 x=3에서 접선
tangent_slope = f_prime(3)
tangent_y = f(3) + tangent_slope * (x_vals - 3)
plt.plot(x_vals, tangent_y, 'r--', label=f'Tangent at x=3 (slope={tangent_slope})')

plt.scatter([3], [f(3)], color='red', s=100, zorder=5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title("Derivative = Slope of Tangent Line")
plt.savefig('derivative.png', dpi=150)
print("\n✓ 시각화 저장: derivative.png")
```

### 🧮 기본 미분 공식 (외워야 함!)

| 함수 | 미분 |
|------|------|
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $\frac{1}{x}$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |

**AI에서 자주 쓰이는 것들**:
- $\frac{d}{dx}(x^2) = 2x$
- $\frac{d}{dx}(e^x) = e^x$ ← Sigmoid, Softmax에서
- $\frac{d}{dx}(\ln(x)) = \frac{1}{x}$ ← Log loss에서

---

## 2. Chain Rule ⭐⭐⭐ **가장 중요!**

### 📖 개념

**Chain Rule**: 합성 함수의 미분

$$
\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)
$$

### 🔥 왜 중요한가?

**Neural Network = 함수의 합성**

```
Input → Layer1 → Layer2 → ... → Loss
  x   →  f₁    →   f₂   →     → L

L = f_n(...f₂(f₁(x))...)
```

Backpropagation = Chain rule을 역방향으로!

### 💻 실습 2: Chain Rule 직접 계산

```python
# chain_rule.py
import numpy as np

# 예: f(x) = (2x + 1)^3
# g(x) = 2x + 1
# f(u) = u^3
# f(g(x)) = (2x + 1)^3

def g(x):
    """Inner function: g(x) = 2x + 1"""
    return 2*x + 1

def f(u):
    """Outer function: f(u) = u^3"""
    return u**3

def composite(x):
    """f(g(x))"""
    return f(g(x))

# 미분들
def g_prime(x):
    """g'(x) = 2"""
    return 2

def f_prime(u):
    """f'(u) = 3u^2"""
    return 3 * u**2

def composite_prime(x):
    """Chain rule: f'(g(x)) * g'(x)"""
    return f_prime(g(x)) * g_prime(x)

# 테스트
x = 2.0
print(f"x = {x}")
print(f"g(x) = {g(x)}")
print(f"f(g(x)) = {composite(x)}")
print(f"\n=== Chain Rule ===")
print(f"f'(g(x)) = f'({g(x)}) = {f_prime(g(x))}")
print(f"g'(x) = {g_prime(x)}")
print(f"d/dx[f(g(x))] = f'(g(x)) × g'(x) = {composite_prime(x)}")

# 수치적으로 검증
h = 1e-6
numerical = (composite(x + h) - composite(x)) / h
print(f"\n수치적 미분: {numerical:.6f}")
print(f"차이: {abs(composite_prime(x) - numerical):.10f}")
```

### 🧮 연습 문제

**문제**: $h(x) = \sin(x^2)$의 미분은?

<details>
<summary>답 보기</summary>

Chain rule 적용:
- 외부: $f(u) = \sin(u)$, $f'(u) = \cos(u)$
- 내부: $g(x) = x^2$, $g'(x) = 2x$

$$
h'(x) = \cos(x^2) \cdot 2x = 2x\cos(x^2)
$$

</details>

---

## 3. Partial Derivatives (편미분)

### 📖 개념

**편미분**: 다변수 함수에서 한 변수만 미분

$$
f(x, y) = x^2 + 3xy + y^2
$$

$$
\frac{\partial f}{\partial x} = 2x + 3y \quad \text{(y는 상수 취급)}
$$

$$
\frac{\partial f}{\partial y} = 3x + 2y \quad \text{(x는 상수 취급)}
$$

### AI에서의 의미

Neural Network의 Loss는 **수백만 개 파라미터의 함수**:

$$
L(w_1, w_2, ..., w_n, b_1, b_2, ...)
$$

각 파라미터에 대한 편미분 = Gradient:

$$
\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, ...
$$

### 💻 실습 3: Partial Derivatives

```python
# partial_derivatives.py
import numpy as np

# 함수: f(x, y) = x^2 + 3xy + y^2
def f(x, y):
    return x**2 + 3*x*y + y**2

# 편미분
def df_dx(x, y):
    """∂f/∂x = 2x + 3y"""
    return 2*x + 3*y

def df_dy(x, y):
    """∂f/∂y = 3x + 2y"""
    return 3*x + 2*y

# 테스트
x, y = 2.0, 3.0
print(f"f({x}, {y}) = {f(x, y)}")
print(f"\n∂f/∂x = {df_dx(x, y)}")
print(f"∂f/∂y = {df_dy(x, y)}")

# 수치적 검증
h = 1e-6
numerical_dx = (f(x + h, y) - f(x, y)) / h
numerical_dy = (f(x, y + h) - f(x, y)) / h

print(f"\n수치적 ∂f/∂x: {numerical_dx:.6f}")
print(f"수치적 ∂f/∂y: {numerical_dy:.6f}")

# AI 예제: Simple Linear Regression
print("\n=== AI 예제: Linear Regression ===")
# y = wx + b
# Loss = (y_pred - y_true)^2

w, b = 2.0, 1.0
x_data, y_true = 3.0, 8.0  # 실제 데이터

y_pred = w * x_data + b
loss = (y_pred - y_true)**2

print(f"예측: y_pred = {w}×{x_data} + {b} = {y_pred}")
print(f"실제: y_true = {y_true}")
print(f"Loss = (y_pred - y_true)² = {loss}")

# Gradient 계산 (손으로!)
# ∂Loss/∂w = 2(y_pred - y_true) × x
# ∂Loss/∂b = 2(y_pred - y_true)

dL_dw = 2 * (y_pred - y_true) * x_data
dL_db = 2 * (y_pred - y_true)

print(f"\n∂Loss/∂w = {dL_dw}")
print(f"∂Loss/∂b = {dL_db}")
print("→ w를 이 방향으로 업데이트하면 Loss 감소!")
```

---

## 4. Gradient (그래디언트)

### 📖 개념

**Gradient** = 모든 편미분을 벡터로 모은 것

$$
\nabla f = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

**의미**: "Loss가 가장 빠르게 증가하는 방향"

→ Gradient의 **반대 방향**으로 가면 Loss 감소!

### 💻 실습 4: Gradient Descent

```python
# gradient_descent.py
import numpy as np
import matplotlib.pyplot as plt

# 함수: f(x) = x^2 + 10sin(x)  (local minima 여러 개)
def f(x):
    return x**2 + 10*np.sin(x)

def f_prime(x):
    """f'(x) = 2x + 10cos(x)"""
    return 2*x + 10*np.cos(x)

# Gradient Descent
x = 8.0  # 시작점
learning_rate = 0.1
history = [x]

print("=== Gradient Descent ===")
for step in range(50):
    grad = f_prime(x)
    x = x - learning_rate * grad  # 핵심 공식!
    history.append(x)

    if step < 5 or step % 10 == 0:
        print(f"Step {step}: x = {x:.4f}, f(x) = {f(x):.4f}, grad = {grad:.4f}")

print(f"\n최종: x = {x:.4f}, f(x) = {f(x):.4f}")

# 시각화
x_vals = np.linspace(-10, 10, 500)
y_vals = f(x_vals)

plt.figure(figsize=(12, 6))
plt.plot(x_vals, y_vals, 'b-', label='f(x) = x² + 10sin(x)', linewidth=2)
plt.plot(history, [f(x) for x in history], 'ro-', label='Gradient Descent Path', markersize=4)
plt.scatter([history[0]], [f(history[0])], color='green', s=200, marker='*',
            label='Start', zorder=5)
plt.scatter([history[-1]], [f(history[-1])], color='red', s=200, marker='*',
            label='End', zorder=5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Optimization')
plt.savefig('gradient_descent.png', dpi=150)
print("\n✓ 시각화 저장: gradient_descent.png")
```

---

## 5. Backpropagation 완전 분해 ⭐⭐⭐

### 🎯 목표: 2-Layer Neural Network의 Gradient 직접 계산

```
Input (x) → Layer1 (W1) → ReLU → Layer2 (W2) → Output (y) → Loss
```

### 💻 실습 5: Backprop from Scratch

```python
# backprop_from_scratch.py
import numpy as np

# 간단한 예제: 1개 데이터, 작은 차원
# Input: x (2,)
# Layer1: W1 (3, 2), b1 (3,)
# ReLU
# Layer2: W2 (1, 3), b2 (1,)
# Loss: MSE

# 초기화
x = np.array([1.0, 2.0])  # 입력
y_true = np.array([1.0])  # 정답

W1 = np.array([[0.1, 0.2],
               [0.3, 0.4],
               [0.5, 0.6]])
b1 = np.array([0.1, 0.1, 0.1])

W2 = np.array([[0.1, 0.2, 0.3]])
b2 = np.array([0.1])

print("=== Forward Pass ===")

# Layer 1
z1 = W1 @ x + b1
print(f"z1 = W1 @ x + b1 = {z1}")

# ReLU
a1 = np.maximum(0, z1)
print(f"a1 = ReLU(z1) = {a1}")

# Layer 2
z2 = W2 @ a1 + b2
print(f"z2 = W2 @ a1 + b2 = {z2}")

# Output (no activation for regression)
y_pred = z2
print(f"y_pred = {y_pred}")

# Loss
loss = 0.5 * (y_pred - y_true)**2
print(f"\nLoss = 0.5 * (y_pred - y_true)² = {loss[0]:.4f}")

print("\n=== Backward Pass (Backpropagation) ===")

# dL/dy_pred
dL_dy = y_pred - y_true
print(f"dL/dy_pred = {dL_dy}")

# dL/dz2 (chain rule: dL/dy × dy/dz2)
# dy/dz2 = 1 (no activation)
dL_dz2 = dL_dy
print(f"dL/dz2 = {dL_dz2}")

# dL/dW2
# z2 = W2 @ a1 + b2
# ∂z2/∂W2 = a1
dL_dW2 = dL_dz2[:, np.newaxis] @ a1[np.newaxis, :]  # Outer product
print(f"dL/dW2 = \n{dL_dW2}")

# dL/db2
dL_db2 = dL_dz2
print(f"dL/db2 = {dL_db2}")

# dL/da1 (chain rule through W2)
dL_da1 = W2.T @ dL_dz2
print(f"\ndL/da1 = W2.T @ dL_dz2 = {dL_da1}")

# dL/dz1 (chain rule through ReLU)
# ReLU'(z) = 1 if z>0 else 0
dReLU = (z1 > 0).astype(float)
dL_dz1 = dL_da1 * dReLU
print(f"dL/dz1 = dL/da1 × ReLU'(z1) = {dL_dz1}")

# dL/dW1
dL_dW1 = dL_dz1[:, np.newaxis] @ x[np.newaxis, :]
print(f"dL/dW1 = \n{dL_dW1}")

# dL/db1
dL_db1 = dL_dz1
print(f"dL/db1 = {dL_db1}")

# 수치적 검증 (W1[0,0]에 대해서만)
h = 1e-7
W1_perturbed = W1.copy()
W1_perturbed[0, 0] += h

z1_new = W1_perturbed @ x + b1
a1_new = np.maximum(0, z1_new)
z2_new = W2 @ a1_new + b2
y_pred_new = z2_new
loss_new = 0.5 * (y_pred_new - y_true)**2

numerical_grad = (loss_new[0] - loss[0]) / h
analytical_grad = dL_dW1[0, 0]

print(f"\n=== Gradient 검증 (W1[0,0]) ===")
print(f"해석적: {analytical_grad:.8f}")
print(f"수치적: {numerical_grad:.8f}")
print(f"차이: {abs(analytical_grad - numerical_grad):.10f}")
print("✓ 일치!" if abs(analytical_grad - numerical_grad) < 1e-6 else "✗ 불일치")

# Gradient Descent로 업데이트
learning_rate = 0.1
print(f"\n=== Weight Update (lr={learning_rate}) ===")
W1_new = W1 - learning_rate * dL_dW1
W2_new = W2 - learning_rate * dL_dW2
print("W1, W2 업데이트 완료!")

# 새로운 loss 계산
z1_new = W1_new @ x + b1
a1_new = np.maximum(0, z1_new)
z2_new = W2_new @ a1_new + b2
y_pred_new = z2_new
loss_new = 0.5 * (y_pred_new - y_true)**2

print(f"이전 Loss: {loss[0]:.6f}")
print(f"새로운 Loss: {loss_new[0]:.6f}")
print(f"감소: {loss[0] - loss_new[0]:.6f}")
print("✓ Loss가 감소했습니다!" if loss_new[0] < loss[0] else "✗ 문제 발생")
```

### 🧠 핵심 직관

**Backpropagation = Chain Rule의 역방향 적용**

```
Forward:  x → z1 → a1 → z2 → y → L
Backward: x ← z1 ← a1 ← z2 ← y ← L

각 단계에서:
dL/d(이전) = dL/d(다음) × d(다음)/d(이전)
```

---

## ✅ Day 5-6 완료 체크리스트

### 이론 이해
- [ ] Derivative가 변화율임을 이해
- [ ] Chain rule의 원리 이해
- [ ] Partial derivative vs Derivative 차이
- [ ] Gradient = Loss가 증가하는 방향

### 손 계산
- [ ] 기본 함수 미분 (x², eˣ 등)
- [ ] Chain rule로 합성함수 미분
- [ ] 편미분 계산
- [ ] 간단한 backprop (2-layer) 손으로 유도

### 코드 구현
- [ ] 수치적 미분 구현 및 검증
- [ ] Gradient descent 구현
- [ ] 2-layer network backprop 완전 구현
- [ ] Gradient 수치적 검증

### AI 연결
- [ ] Backpropagation = Chain rule 이해
- [ ] Gradient descent의 원리 설명 가능
- [ ] ReLU의 미분 이해
- [ ] Loss function의 gradient 계산 가능

---

## 🎯 최종 챌린지

**3-layer Network의 Backprop을 종이에 유도하세요!**

```
x → W1 → ReLU → W2 → ReLU → W3 → MSE Loss
```

모든 dL/dW1, dL/dW2, dL/dW3를 유도할 수 있으면,
**당신은 Backpropagation을 완전히 마스터했습니다!** 🎉

---

## ⏭️ 다음 단계

👉 [Day 7: 최적화 기초 (SGD, Adam)](./04-optimization.md)

**"이제 모든 논문의 미분 수식이 읽힙니다!"** 🚀
