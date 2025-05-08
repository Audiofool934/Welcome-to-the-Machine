## NumPy (`numpy`) 

**What is it?**
  - NumPy stands for "**Numerical Python**".
  - It's the fundamental package for scientific computing in Python.
  - It provides a high-performance **multidimensional array object**, and tools for working with these arrays.

**Core Object: `ndarray`**
  - A fast and memory-efficient multidimensional array providing a container for ***homogeneous* data** (all elements of the same type, e.g., all integers or all floats).
  - Arrays can have any number of dimensions:
    - 1D array: **Vector**
    - 2D array: **Matrix**
    - 3D or higher-D array: **Tensor**

**Key Features:**
  - **Vectorized Operations:** NumPy allows you to perform element-wise operations on entire arrays <u>without writing explicit loops in Python</u>. This is <span style='color:red'>much faster</span> because these operations are implemented in C or Fortran.
  - **Broadcasting:** A powerful mechanism that allows NumPy to work with <u>arrays of different shapes</u> when performing arithmetic operations.
  - **Rich Functionality:** Comprehensive <u>mathematical functions</u>, random number capabilities, linear algebra routines, Fourier transforms, etc.
  - **Interoperability:** Many other Python libraries (like Pandas, SciPy, Scikit-learn, and PyTorch) are built on top of or integrate well with NumPy.

**Why use it?**
  - **Performance:** For numerical computations, it's significantly faster than pure Python lists and loops.
  - **Convenience:** Provides a concise and expressive syntax for numerical operations.
  - **Foundation:** It's the bedrock of the scientific Python stack.

**Simple NumPy Example:**
```python
import numpy as np

# Create a NumPy array from a Python list
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])

# Element-wise addition (vectorized)
c = a + b
print(f"a: {a}")
print(f"b: {b}")
print(f"a + b: {c}")

# Scalar multiplication
d = a * 2
print(f"a * 2: {d}")

# Create a 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Matrix:\n{matrix}")
print(f"Matrix shape: {matrix.shape}")
```

## Pytorch (`torch`)

- **What is it?**
  - An open-source <u>machine learning framework</u> primarily developed by Meta AI. (PyTorch was started by Adam Paszke as an intern project under Facebook Research.)
  - Known for its flexibility, ease of use, and strong Python integration.
  - Widely used for deep learning research and production.

- **Core Object: `Tensor`**
  - Similar to NumPy's `ndarray`, PyTorch tensors are <u>multi-dimensional arrays</u>.
  - **Key Difference:** Tensors can be moved to <u>GPUs for massive parallel computation</u>, which is crucial for training deep learning models.
  - Tensors also form the basis of **Automatic Differentiation (<span style='color:red'>Autograd</span>)**.

- **Key Features:**
  - **GPU Acceleration:** Seamlessly run computations on GPUs (and other accelerators).
  - **Automatic Differentiation (`torch.autograd`):**
    - PyTorch can automatically compute gradients of operations performed on tensors.
    - This is the <u>backbone of training neural networks via backpropagation</u>.
    - It builds a "computational graph" dynamically as operations are performed.
  - **Neural Network Module (`torch.nn`):** Provides <u>pre-defined layers, loss functions, and utilities</u> for building and training neural networks.
  - **Optimization Algorithms (`torch.optim`):** Includes common <u>optimizers</u> like SGD, Adam, etc.
  - **Utilities:** Data loading (`torch.utils.data.DataLoader`), distributed training, and more.
  - **Dynamic Computational Graphs:** Unlike some older frameworks (like early TensorFlow), PyTorch builds the graph "on the fly" as code executes. This makes debugging easier and allows for more flexible model architectures (e.g., with varying structures per iteration).

- **Why use it for Machine Learning / Deep Learning?**
  - **Python-first:** Feels very natural for Python developers.
  - **Flexibility & Control:** Offers a good balance between high-level abstractions and low-level control.
  - **Strong Research Community:** Rapid adoption of new ideas and models.
  - **Ease of Debugging:** Dynamic graphs make it easier to inspect intermediate values.

**Simple PyTorch Example:**
```python
import torch

# Create a PyTorch tensor
a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
b = torch.tensor([6, 7, 8, 9, 10], dtype=torch.float32)

# Element-wise addition
c = a + b
print(f"a: {a}")
print(f"b: {b}")
print(f"a + b: {c}")

# Scalar multiplication
d = a * 2
print(f"a * 2: {d}")

# Create a 2D tensor (matrix)
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(f"Matrix:\n{matrix}")
print(f"Matrix shape: {matrix.shape}")

# Automatic differentiation example
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1
y.backward() # Computes gradients dy/dx
print(f"x: {x}")
print(f"y = x^2 + 3x + 1: {y}")
print(f"dy/dx at x=2: {x.grad}") # Expected: 2*x + 3 = 2*2 + 3 = 7
```

## NumPy vs. PyTorch Tensors

*   PyTorch tensors can be easily converted to NumPy arrays and vice-versa.
*   `tensor.numpy()`: PyTorch Tensor -> NumPy array (shares memory if on CPU)
*   `torch.from_numpy(ndarray)`: NumPy array -> PyTorch Tensor (shares memory)

```python
# Interoperability
numpy_arr = np.array([10, 20, 30])
pytorch_tensor = torch.from_numpy(numpy_arr)
print(f"From NumPy to PyTorch: {pytorch_tensor}")

new_numpy_arr = pytorch_tensor.numpy()
print(f"From PyTorch to NumPy: {new_numpy_arr}")

# If a tensor is on GPU, you need to move it to CPU first
# if torch.cuda.is_available():
#     gpu_tensor = torch.tensor([1,2,3], device="cuda")
#     # cpu_tensor = gpu_tensor.cpu() # Move to CPU
#     # numpy_arr_from_gpu = cpu_tensor.numpy()
```