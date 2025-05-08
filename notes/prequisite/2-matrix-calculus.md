## Matrix Calculus

**Concept:** <u>Extends calculus concepts (derivatives, gradients) to functions involving matrices and vectors</u>. Essential for optimizing ML models.

**Why it's important for ML/DL:**
- **Optimization:** Most ML models are trained by minimizing a loss function. Gradient Descent and its variants require computing the gradient of the loss function with respect to model parameters (which are often matrices/vectors).
- **Backpropagation:** The algorithm used to train neural networks is essentially an application of the chain rule from matrix calculus to compute gradients efficiently.

**Key Concepts & Implementations:**

#### a. Gradient

- For a **scalar-valued function** $f(\mathbf{x})$ of a vector $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$, the gradient $\nabla_{\mathbf{x}} f(\mathbf{x})$ is a vector of partial derivatives:
    $$
    \nabla_{\mathbf{x}} f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
    $$
- It points in the direction of the steepest ascent of the function. $-\nabla_{\mathbf{x}} f(\mathbf{x})$ points in the direction of steepest descent.

#### b. Jacobian

- For a **vector-valued function** $\mathbf{f}(\mathbf{x}): \mathbb{R}^n \to \mathbb{R}^m$, where $\mathbf{f} = [f_1, f_2, \dots, f_m]^T$ and $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$.
- The Jacobian matrix $\mathbf{J}$ is an $m \times n$ matrix of all first-order partial derivatives: $J_{ij} = \frac{\partial f_i}{\partial x_j}$
    $$
    \mathbf{J} = \begin{bmatrix}
    \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
    \vdots & \ddots & \vdots \\
    \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
    \end{bmatrix} = \begin{bmatrix}
    (\nabla_{\mathbf{x}} f_1(\mathbf{x}))^T \\
    \vdots \\
    (\nabla_{\mathbf{x}} f_m(\mathbf{x}))^T
    \end{bmatrix}
    $$
- If $m=1$ (scalar function), the Jacobian is the transpose of the gradient vector.

#### c. Hessian

- For a **scalar-valued function** $f(\mathbf{x})$ of a vector $\mathbf{x}$, the Hessian matrix $\mathbf{H}$ is an $n \times n$ matrix of second-order partial derivatives: $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$
    $$
    \mathbf{H} = \begin{bmatrix}
    \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
    \vdots & \ddots & \vdots \\
    \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
    \end{bmatrix}
    $$
- The Hessian describes the local curvature of the function. Used in second-order optimization methods (e.g., Newton's method).

#### d. Chain Rule (for vectors/matrices)

- If $y = f(u)$ and $u = g(\mathbf{x})$, then $\frac{\partial y}{\partial x_i} = \frac{df}{du} \frac{\partial g}{\partial x_i}$.
- If $\mathbf{z} = f(\mathbf{y})$ and $\mathbf{y} = g(\mathbf{x})$, then the Jacobian of $\mathbf{z}$ w.r.t. $\mathbf{x}$ is $\mathbf{J}_{\mathbf{x}}(\mathbf{z}) = \mathbf{J}_{\mathbf{y}}(\mathbf{z}) \mathbf{J}_{\mathbf{x}}(\mathbf{y})$.
- This is the fundamental principle behind backpropagation in neural networks.

**PyTorch Implementation (Autograd):**
PyTorch's `autograd` package is designed for this.

```python
import torch

# Scalar output function example
x_pt = torch.tensor([1., 2., 3.], requires_grad=False) # Input data, no gradient needed for x itself
w_pt = torch.tensor([0.1, 0.2, 0.3], requires_grad=True) # Parameters, we need gradients for these

# Forward pass: y = (w^T x)^2
# PyTorch builds a computation graph
y_pt = (w_pt @ x_pt)**2
print(f"Output y (PyTorch): {y_pt}")

# Backward pass: compute gradients
y_pt.backward() # Computes dy/dw and stores it in w_pt.grad

# Gradients are stored in .grad attribute of tensors with requires_grad=True
print(f"Gradient dy/dw (PyTorch autograd): {w_pt.grad}")

# Jacobian example (output is a vector)
# PyTorch computes "vector-Jacobian products" efficiently via backward()
# For full Jacobian, one might need to call backward() multiple times or use torch.autograd.functional.jacobian

# Let z = [w1*x1, w2*x2, w3*x3]
w_jac_pt = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
x_jac_pt = torch.tensor([1., 2., 3.], requires_grad=False)

z_pt = w_jac_pt * x_jac_pt # Element-wise product
print(f"Vector output z_pt: {z_pt}") # [0.1, 0.4, 0.9]

# To get Jacobian of z w.r.t w_jac_pt:
# dz_i / dw_j. Here, dz_i / dw_j is x_i if i=j, and 0 otherwise.
# So Jacobian should be diag(x_jac_pt)
# Using torch.autograd.functional.jacobian
from torch.autograd.functional import jacobian
J = jacobian(lambda w: w * x_jac_pt, w_jac_pt)
print(f"Jacobian J = dz/dw (PyTorch):\n{J}") # Expected: [[1,0,0],[0,2,0],[0,0,3]]

# Hessian example (for scalar output function)
# Using torch.autograd.functional.hessian
from torch.autograd.functional import hessian
# Consider f(w1, w2) = w1^2 * w2 + w2^3
def func_hessian(w_h):
    return w_h[0]**2 * w_h[1] + w_h[1]**3

w_h_pt = torch.tensor([1.0, 2.0], requires_grad=True)
H = hessian(func_hessian, w_h_pt)
# df/dw1 = 2*w1*w2 -> d2f/dw1^2 = 2*w2 (=4); d2f/dw1dw2 = 2*w1 (=2)
# df/dw2 = w1^2 + 3*w2^2 -> d2f/dw2^2 = 6*w2 (=12); d2f/dw2dw1 = 2*w1 (=2)
# H = [[4, 2], [2, 12]]
print(f"Hessian H (PyTorch):\n{H}")
```
*For the talks, the PyTorch `autograd` part is key. Mentioning that `backward()` computes gradients which are essential for optimization via gradient descent is the main takeaway.*