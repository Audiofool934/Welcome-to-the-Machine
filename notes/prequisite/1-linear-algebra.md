## Linear Algebra

**Concept:** Linear algebra is the branch of mathematics concerning vector spaces and linear mappings between them. In ML, it's fundamental for **<span style='color:lightgreen'>representing data, defining transformations, and solving systems of equations</span>.**

**Why it's important for ML/DL:**
- **Data Representation:** <u>Datasets</u> are often represented as matrices (samples vs. features). Images, videos, and text can be represented as tensors.
- **Model Parameters:** <u>Weights and biases</u> in neural networks are typically stored as matrices and vectors.
- **Transformations:** <u>Linear transformations</u> (matrix multiplications) are core operations in many ML models, especially neural networks.
- **Dimensionality Reduction:** Techniques like PCA rely heavily on linear algebra.
- **Optimization:** Solving linear systems is often a sub-problem in optimization routines.

**Key Concepts & Implementations:**

#### a. Scalars, Vectors, Matrices, Tensors

- **Scalar:** A single number (e.g., `5`, `3.14`).
- **Vector:** An ordered array of numbers. Can be a row or column vector. In ML, often <u>represents a single data point or a feature vector</u>.
- **Matrix:** A <u>2D array of numbers</u>. In ML, often represents a <u>dataset (rows=samples, cols=features)</u> or <u>model parameters</u> (weights of a neural network layer).
- **Tensor:** A generalization of scalars, vectors, and matrices to an arbitrary number of dimensions (or "axes").
  - 0D Tensor: Scalar
  - 1D Tensor: Vector
  - 2D Tensor: Matrix
  - 3D Tensor: e.g., an RGB image (height, width, channels) or a batch of sequences (batch_size, seq_length, features).
  - 4D Tensor: e.g., a batch of RGB images (batch_size, height, width, channels).

**NumPy Implementation:**
```python
import numpy as np

# Scalar
scalar_np = np.array(5)
print(f"Scalar (NumPy): {scalar_np}, shape: {scalar_np.shape}, ndim: {scalar_np.ndim}")

# Vector (1D array)
vector_np = np.array([1, 2, 3])
print(f"Vector (NumPy): {vector_np}, shape: {vector_np.shape}, ndim: {vector_np.ndim}")

# Matrix (2D array)
matrix_np = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Matrix (NumPy):\n{matrix_np}\nshape: {matrix_np.shape}, ndim: {matrix_np.ndim}")

# Tensor (3D array example)
tensor_3d_np = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Tensor (NumPy):\n{tensor_3d_np}\nshape: {tensor_3d_np.shape}, ndim: {tensor_3d_np.ndim}")
```

**PyTorch Implementation:**
```python
import torch

# Scalar
scalar_pt = torch.tensor(5.0) # PyTorch infers dtype, often float32 by default for scalars
print(f"Scalar (PyTorch): {scalar_pt}, shape: {scalar_pt.shape}, ndim: {scalar_pt.ndim}")

# Vector (1D tensor)
vector_pt = torch.tensor([1.0, 2.0, 3.0])
print(f"Vector (PyTorch): {vector_pt}, shape: {vector_pt.shape}, ndim: {vector_pt.ndim}")

# Matrix (2D tensor)
matrix_pt = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"Matrix (PyTorch):\n{matrix_pt}\nshape: {matrix_pt.shape}, ndim: {matrix_pt.ndim}")

# Tensor (3D tensor example)
tensor_3d_pt = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
print(f"3D Tensor (PyTorch):\n{tensor_3d_pt}\nshape: {tensor_3d_pt.shape}, ndim: {tensor_3d_pt.ndim}")

# Common practice to specify dtype for consistency, e.g. torch.float32
vector_pt_float = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"Vector (PyTorch float32): {vector_pt_float}, dtype: {vector_pt_float.dtype}")
```

#### b. Dot Product (Inner Product)

- For two vectors $\mathbf{a}$ and $\mathbf{b}$ of the same length $n$, the dot product is $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$.
- Result is a scalar.
- Geometric interpretation: $\mathbf{a} \cdot \mathbf{b} = ||\mathbf{a}|| \cdot ||\mathbf{b}|| \cos(\theta)$, where $\theta$ is the angle between them.
  - Measures <u>similarity/alignment</u>. If vectors are orthogonal, dot product is 0.

**NumPy Implementation:**
```python
a_np = np.array([1, 2, 3])
b_np = np.array([4, 5, 6])

# Method 1: np.dot()
dot_product_np1 = np.dot(a_np, b_np)
print(f"Dot product (NumPy - np.dot): {dot_product_np1}") # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

# Method 2: @ operator (for Python 3.5+) - preferred for matrix multiplication, but works for dot product
dot_product_np2 = a_np @ b_np
print(f"Dot product (NumPy - @): {dot_product_np2}")

# Method 3: element-wise multiplication and sum
dot_product_np3 = np.sum(a_np * b_np)
print(f"Dot product (NumPy - sum(a*b)): {dot_product_np3}")
```

**PyTorch Implementation:**
```python
a_pt = torch.tensor([1., 2., 3.])
b_pt = torch.tensor([4., 5., 6.])

# Method 1: torch.dot()
dot_product_pt1 = torch.dot(a_pt, b_pt)
print(f"Dot product (PyTorch - torch.dot): {dot_product_pt1}")

# Method 2: torch.matmul() or @ (for 1D vectors, these perform dot product)
dot_product_pt2 = torch.matmul(a_pt, b_pt)
print(f"Dot product (PyTorch - torch.matmul): {dot_product_pt2}")
dot_product_pt3 = a_pt @ b_pt
print(f"Dot product (PyTorch - @): {dot_product_pt3}")

# Method 4: element-wise multiplication and sum
dot_product_pt4 = torch.sum(a_pt * b_pt)
print(f"Dot product (PyTorch - sum(a*b)): {dot_product_pt4}")
```

#### c. Matrix Multiplication

- If $\mathbf{A}$ is an $m \times n$ matrix and $\mathbf{B}$ is an $n \times p$ matrix, their product $\mathbf{C} = \mathbf{A}\mathbf{B}$ is an $m \times p$ matrix.
- $C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$ (dot product of $i$-th row of $\mathbf{A}$ and $j$-th column of $\mathbf{B}$).
- Order matters: $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$ in general.
- Fundamental operation in neural networks: `output = activation(weights @ inputs + bias)`.

**NumPy Implementation:**
```python
A_np = np.array([[1, 2], [3, 4], [5,6]]) # 3x2
B_np = np.array([[7, 8, 9], [10, 11, 12]]) # 2x3

# Method 1: np.matmul()
C_np1 = np.matmul(A_np, B_np)
print(f"Matrix multiplication (NumPy - np.matmul):\n{C_np1}") # Expected shape: 3x3

# Method 2: @ operator
C_np2 = A_np @ B_np
print(f"Matrix multiplication (NumPy - @):\n{C_np2}")

# Note: np.dot() behaves differently for 2D arrays (it's matrix multiplication)
# but np.matmul() or @ is generally preferred for clarity when dealing with matrices.
C_np3 = np.dot(A_np, B_np)
print(f"Matrix multiplication (NumPy - np.dot for 2D):\n{C_np3}")
```

**PyTorch Implementation:**
```python
A_pt = torch.tensor([[1., 2.], [3., 4.], [5.,6.]]) # 3x2
B_pt = torch.tensor([[7., 8., 9.], [10., 11., 12.]]) # 2x3

# Method 1: torch.matmul()
C_pt1 = torch.matmul(A_pt, B_pt)
print(f"Matrix multiplication (PyTorch - torch.matmul):\n{C_pt1}") # Expected shape: 3x3

# Method 2: @ operator
C_pt2 = A_pt @ B_pt
print(f"Matrix multiplication (PyTorch - @):\n{C_pt2}")

# torch.mm() is specifically for 2D matrix multiplication (no broadcasting)
C_pt3 = torch.mm(A_pt, B_pt)
print(f"Matrix multiplication (PyTorch - torch.mm):\n{C_pt3}")
```

#### d. Transpose

- The transpose of a matrix $\mathbf{A}$, denoted $\mathbf{A}^T$, flips the matrix over its main diagonal. Rows become columns and vice-versa.
- If $\mathbf{A}$ is $m \times n$, then $\mathbf{A}^T$ is $n \times m$. $(A^T)_{ij} = A_{ji}$.
- Useful properties: $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T$.

**NumPy Implementation:**
```python
A_np = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3
A_T_np1 = A_np.T
print(f"Original Matrix (NumPy):\n{A_np}")
print(f"Transposed Matrix (NumPy - .T):\n{A_T_np1}") # 3x2

A_T_np2 = np.transpose(A_np)
print(f"Transposed Matrix (NumPy - np.transpose()):\n{A_T_np2}")
```

**PyTorch Implementation:**
```python
A_pt = torch.tensor([[1., 2., 3.], [4., 5., 6.]]) # 2x3
A_T_pt1 = A_pt.T
print(f"Original Matrix (PyTorch):\n{A_pt}")
print(f"Transposed Matrix (PyTorch - .T):\n{A_T_pt1}") # 3x2

A_T_pt2 = torch.transpose(A_pt, 0, 1) # transpose dimensions 0 and 1
print(f"Transposed Matrix (PyTorch - torch.transpose()):\n{A_T_pt2}")
```

#### e. Inverse & Pseudo-inverse

- **Inverse ($\mathbf{A}^{-1}$):** For a square matrix $\mathbf{A}$, its inverse $\mathbf{A}^{-1}$ (if it exists) is such that $\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$ (identity matrix).
    - <span style='color:red'>Only exists if $\mathbf{A}$ is non-singular (determinant $\neq 0$).</span>
    - Used to solve systems of linear equations $\mathbf{Ax} = \mathbf{b} \implies \mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$.
- **Pseudo-inverse ($\mathbf{A}^{+}$):** A generalization of the inverse for non-square or singular matrices.
    - <span style='color:red'>Finds a "best fit" (least squares) solution to $\mathbf{Ax} = \mathbf{b}$.</span>
    - Used in linear regression (Normal Equation: $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$).

**NumPy Implementation:**
```python
from numpy.linalg import inv, pinv

# Inverse
A_inv_np = np.array([[1, 2], [3, 5]]) # Must be square and non-singular
if np.linalg.det(A_inv_np) != 0:
    A_inv_np_inv = inv(A_inv_np)
    print(f"Matrix for inversion (NumPy):\n{A_inv_np}")
    print(f"Inverse (NumPy):\n{A_inv_np_inv}")
    print(f"Check A @ A_inv (should be Identity):\n{A_inv_np @ A_inv_np_inv}")
else:
    print(f"Matrix (NumPy) is singular, cannot invert:\n{A_inv_np}")


# Pseudo-inverse
A_pinv_np = np.array([[1, 2], [3, 4], [5, 6]]) # Non-square
A_pinv_np_plus = pinv(A_pinv_np)
print(f"Matrix for pseudo-inversion (NumPy):\n{A_pinv_np}")
print(f"Pseudo-inverse (NumPy):\n{A_pinv_np_plus}")
# Check: A @ A+ @ A = A
print(f"Check A @ A+ @ A (NumPy):\n{A_pinv_np @ A_pinv_np_plus @ A_pinv_np}")
```

**PyTorch Implementation:**
```python
from torch.linalg import inv as torch_inv, pinv as torch_pinv

# Inverse
A_inv_pt = torch.tensor([[1., 2.], [3., 5.]])
if torch.det(A_inv_pt) != 0:
    A_inv_pt_inv = torch_inv(A_inv_pt)
    print(f"Matrix for inversion (PyTorch):\n{A_inv_pt}")
    print(f"Inverse (PyTorch):\n{A_inv_pt_inv}")
    print(f"Check A @ A_inv (should be Identity):\n{A_inv_pt @ A_inv_pt_inv}")
else:
    print(f"Matrix (PyTorch) is singular, cannot invert:\n{A_inv_pt}")

# Pseudo-inverse
A_pinv_pt = torch.tensor([[1., 2.], [3., 4.], [5., 6.]]) # Non-square
A_pinv_pt_plus = torch_pinv(A_pinv_pt)
print(f"Matrix for pseudo-inversion (PyTorch):\n{A_pinv_pt}")
print(f"Pseudo-inverse (PyTorch):\n{A_pinv_pt_plus}")
# Check: A @ A+ @ A = A
print(f"Check A @ A+ @ A (PyTorch):\n{A_pinv_pt @ A_pinv_pt_plus @ A_pinv_pt}")
```

#### f. Determinant

- A scalar value that can be computed from a square matrix.
- Geometrically, for a 2x2 matrix, it's the signed area of the parallelogram formed by its column (or row) vectors. For 3x3, it's the signed volume.
- **If determinant is 0, the matrix is <span style='color:red'>singular</span> (no inverse, linear transformation collapses space into a lower dimension).**

**NumPy Implementation:**
```python
from numpy.linalg import det

M_np = np.array([[1, 2], [3, 4]]) # det = 1*4 - 2*3 = 4 - 6 = -2
det_M_np = det(M_np)
print(f"Matrix (NumPy):\n{M_np}")
print(f"Determinant (NumPy): {det_M_np}")

M_singular_np = np.array([[1, 2], [2, 4]]) # det = 1*4 - 2*2 = 0
det_M_singular_np = det(M_singular_np)
print(f"Singular Matrix (NumPy):\n{M_singular_np}")
print(f"Determinant of singular matrix (NumPy): {det_M_singular_np}")
```

**PyTorch Implementation:**
```python
from torch.linalg import det as torch_det

M_pt = torch.tensor([[1., 2.], [3., 4.]])
det_M_pt = torch_det(M_pt)
print(f"Matrix (PyTorch):\n{M_pt}")
print(f"Determinant (PyTorch): {det_M_pt}")

M_singular_pt = torch.tensor([[1., 2.], [2., 4.]])
det_M_singular_pt = torch_det(M_singular_pt)
print(f"Singular Matrix (PyTorch):\n{M_singular_pt}")
print(f"Determinant of singular matrix (PyTorch): {det_M_singular_pt}")
```

#### g. Eigenvalues & Eigenvectors

- For a square matrix $\mathbf{A}$, an eigenvector $\mathbf{v}$ is a non-zero vector that, when multiplied by $\mathbf{A}$, only changes in scale (by a factor $\lambda$, the eigenvalue), not direction: $\mathbf{Av} = \lambda\mathbf{v}$.
- Eigenvectors represent the <span style='color:red'>principal axes of the linear transformation defined by $\mathbf{A}$</span>. Eigenvalues indicate the <span style='color:red'>scaling factor along these axes</span>.
- Crucial for Principal Component Analysis (PCA), understanding matrix powers, and stability analysis of dynamical systems.

**NumPy Implementation:**
```python
from numpy.linalg import eig

A_eig_np = np.array([[2, 1], [1, 2]])
eigenvalues_np, eigenvectors_np = eig(A_eig_np)

print(f"Matrix (NumPy):\n{A_eig_np}")
print(f"Eigenvalues (NumPy): {eigenvalues_np}")
print(f"Eigenvectors (NumPy) (columns are eigenvectors):\n{eigenvectors_np}")

# Check: A @ v = lambda * v for the first eigenvector/value
v1_np = eigenvectors_np[:, 0]
lambda1_np = eigenvalues_np[0]
print(f"A @ v1 (NumPy): {A_eig_np @ v1_np}")
print(f"lambda1 * v1 (NumPy): {lambda1_np * v1_np}")
```

**PyTorch Implementation:**
```python
from torch.linalg import eig

A_eig_pt = torch.tensor([[2., 1.], [1., 2.]])
# eig returns complex values by default, even if eigenvalues are real
eigenvalues_pt_complex, eigenvectors_pt_complex = eig(A_eig_pt)

# For real symmetric matrices, use eigh for real eigenvalues/vectors
eigenvalues_pt, eigenvectors_pt = torch.linalg.eigh(A_eig_pt, UPLO='U') # 'U' for upper triangle, 'L' for lower

print(f"Matrix (PyTorch):\n{A_eig_pt}")
print(f"Eigenvalues (PyTorch - eigh for real symmetric): {eigenvalues_pt}")
print(f"Eigenvectors (PyTorch - eigh, columns are eigenvectors):\n{eigenvectors_pt}")

# Check: A @ v = lambda * v
v1_pt = eigenvectors_pt[:, 0]
lambda1_pt = eigenvalues_pt[0]
print(f"A @ v1 (PyTorch): {A_eig_pt @ v1_pt}")
print(f"lambda1 * v1 (PyTorch): {lambda1_pt * v1_pt}")
```

#### [h. Singular Value Decomposition (SVD)](appendix/svd.md)

- Any $m \times n$ matrix $\mathbf{A}$ can be factorized as $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$.
  - $\mathbf{U}$: $m \times m$ orthogonal matrix (columns are left singular vectors).
  - $\mathbf{\Sigma}$: $m \times n$ diagonal matrix (diagonal entries are singular values, $\sigma_i \ge 0$).
  - $\mathbf{V}^T$: $n \times n$ orthogonal matrix (rows are transposes of right singular vectors). (Often $\mathbf{V}$ is returned, so it's $\mathbf{U}\mathbf{\Sigma}\mathbf{V}^*$ or $\mathbf{U}\mathbf{\Sigma}\mathbf{V}^H$ where $H$ is conjugate transpose).
- Singular values are related to eigenvalues of $\mathbf{A}^T\mathbf{A}$ and $\mathbf{A}\mathbf{A}^T$.
- Applications: Dimensionality reduction (PCA can be derived from SVD), matrix approximation (low-rank approximation), pseudo-inverse computation, solving least squares problems.

**NumPy Implementation:**
```python
from numpy.linalg import svd

A_svd_np = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3 matrix
U_np, s_np, Vh_np = svd(A_svd_np) # Vh is V_transpose

print(f"Matrix A (NumPy):\n{A_svd_np}")
print(f"U (NumPy):\n{U_np}\nshape: {U_np.shape}")
print(f"Singular values s (NumPy): {s_np}\nshape: {s_np.shape}") # 1D array of singular values
print(f"Vh (V transpose) (NumPy):\n{Vh_np}\nshape: {Vh_np.shape}")

# Reconstruct A: need to form Sigma matrix from s
Sigma_np = np.zeros(A_svd_np.shape)
Sigma_np[:A_svd_np.shape[0], :A_svd_np.shape[0]] = np.diag(s_np) # Simpler for m<=n
# If m > n, Sigma_np needs to be padded correctly, e.g.
# Sigma_np = np.zeros((A_svd_np.shape[0], A_svd_np.shape[1]))
# Sigma_np[:len(s_np), :len(s_np)] = np.diag(s_np)

# For our 2x3 case, s_np has 2 values. Sigma should be 2x3.
Sigma_reconstruct_np = np.zeros(A_svd_np.shape)
Sigma_reconstruct_np[0,0] = s_np[0]
Sigma_reconstruct_np[1,1] = s_np[1]

A_reconstructed_np = U_np @ Sigma_reconstruct_np @ Vh_np
print(f"Reconstructed A (NumPy):\n{A_reconstructed_np}")
```

**PyTorch Implementation:**
```python
from torch.linalg import svd as torch_svd

A_svd_pt = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
U_pt, S_pt, Vh_pt = torch_svd(A_svd_pt) # Vh is V_transpose in PyTorch >1.8, V before

print(f"Matrix A (PyTorch):\n{A_svd_pt}")
print(f"U (PyTorch):\n{U_pt}\nshape: {U_pt.shape}")
print(f"Singular values S (PyTorch): {S_pt}\nshape: {S_pt.shape}") # 1D tensor of singular values
print(f"Vh (V transpose) (PyTorch):\n{Vh_pt}\nshape: {Vh_pt.shape}")

# Reconstruct A: S_pt is 1D, need to make it a diagonal matrix for matmul
Sigma_pt = torch.diag(S_pt)
# Sigma_pt needs to be padded to A's original shape for correct multiplication
# For a 2x3 matrix, Sigma will be effectively 2x2, then padded for multiplication with Vh (3x3)
# U (2x2) @ Sigma (2x2, effectively) @ Vh (3x3)
# A_reconstructed_pt = U_pt @ Sigma_pt @ Vh_pt[:S_pt.shape[0], :] # If Vh is larger
# The actual reconstruction: U @ diag(S) @ Vh (where diag(S) might need padding to A.shape[1] columns)
# A = U S Vh
# U (m,k), S (k), Vh (k,n) where k=min(m,n)
# Create a full Sigma matrix
Sigma_full_pt = torch.zeros(A_svd_pt.shape[0], A_svd_pt.shape[1], dtype=A_svd_pt.dtype)
Sigma_full_pt.as_strided(S_pt.size(), [Sigma_full_pt.stride(0) + Sigma_full_pt.stride(1), Sigma_full_pt.stride(1)]).copy_(torch.diag(S_pt))


# A_reconstructed_pt = U_pt @ torch.diag_embed(S_pt, offset=0, dim1=-2, dim2=-1)[:, :A_svd_pt.shape[1]] @ Vh_pt
# This is tricky because S_pt is min(m,n).
# Proper reconstruction:
S_diag_pt = torch.diag(S_pt) # This creates a k x k matrix where k = min(m,n)
if A_svd_pt.shape[0] < A_svd_pt.shape[1]: # m < n
    S_mat = torch.cat((S_diag_pt, torch.zeros(S_diag_pt.shape[0], A_svd_pt.shape[1] - S_diag_pt.shape[1])), dim=1)
elif A_svd_pt.shape[0] > A_svd_pt.shape[1]: # m > n
    S_mat = torch.cat((S_diag_pt, torch.zeros(A_svd_pt.shape[0] - S_diag_pt.shape[0], S_diag_pt.shape[1])), dim=0)
else: # m == n
    S_mat = S_diag_pt

A_reconstructed_pt = U_pt @ S_mat @ Vh_pt
print(f"Reconstructed A (PyTorch):\n{A_reconstructed_pt}")
```
